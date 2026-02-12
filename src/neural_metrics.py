"""Neural evaluation metrics for Bipolar Argumentation Framework extraction.

Reference-free metrics that evaluate a generated BAF against its source essay
(no gold BAF needed).  Two complementary metrics:

1. **BERTScore AF-Coverage**: Token-level semantic overlap between AF content
   and essay.  Measures faithfulness (precision) and coverage (recall).

2. **NLI AF-Faithfulness**: Whether the essay entails the claims made by the AF
   (both arguments and relational structure).

These run as a post-processing step on saved results — no GPU dependency needed
for the main LLM experiment loop.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple

import numpy as np

from .baf import BAF

logger = logging.getLogger("neural_metrics")

# ---------------------------------------------------------------------------
# Natural sort helper
# ---------------------------------------------------------------------------


def _natural_sort_key(s: str) -> list:
    """Sort key that handles embedded numbers: 'a2' < 'a10'."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", s)
    ]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_arguments(baf: BAF) -> str:
    """Serialize argument texts for BERTScore (S1).

    Returns newline-separated argument texts, sorted by argument ID.
    No structural information (relations, IDs, templates).
    """
    if not baf.arguments:
        return ""
    sorted_args = sorted(baf.arguments, key=lambda a: _natural_sort_key(a.id))
    return "\n".join(a.text for a in sorted_args)


def serialize_af_statements(baf: BAF) -> List[str]:
    """Serialize AF into individual statements for NLI (S2).

    Returns a list of statement strings:
    - One per argument: the raw argument text.
    - One per relation: "{source.text}. This {supports/challenges} the argument
      that {target.text}"

    Each statement is individually paired with the essay for NLI inference.
    """
    statements: List[str] = []
    arg_map = {a.id: a for a in baf.arguments}

    # Argument statements (sorted by ID for determinism)
    for arg in sorted(baf.arguments, key=lambda a: _natural_sort_key(a.id)):
        statements.append(arg.text)

    # Relation statements
    for rel in baf.relations:
        src = arg_map.get(rel.source)
        tgt = arg_map.get(rel.target)
        if src is None or tgt is None:
            continue
        if rel.type == "support":
            verb = "supports"
        elif rel.type == "attack":
            verb = "challenges"
        else:
            continue
        statements.append(
            f"{src.text}. This {verb} the argument that {tgt.text}"
        )

    return statements


# ---------------------------------------------------------------------------
# Lazy model loading (singletons)
# ---------------------------------------------------------------------------

_nli_cache: Dict[str, Tuple] = {}


def _get_nli_model(
    model_name: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cpu",
) -> Tuple:
    """Load and cache NLI tokenizer + model.

    Returns (tokenizer, model, device, entailment_idx).
    """
    cache_key = f"{model_name}:{device}"
    if cache_key in _nli_cache:
        return _nli_cache[cache_key]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading NLI model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Derive entailment label index from model config
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    ent_idx = next((v for k, v in label2id.items() if "entail" in k), 2)

    _nli_cache[cache_key] = (tokenizer, model, device, ent_idx)
    return tokenizer, model, device, ent_idx


# ---------------------------------------------------------------------------
# BERTScore AF-Coverage
# ---------------------------------------------------------------------------


def bertscore_af_coverage(
    baf: BAF,
    essay_text: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, float]:
    """BERTScore between concatenated argument texts and essay.

    Returns dict with keys:
        bertscore_precision  — Faithfulness: AF tokens grounded in essay
        bertscore_recall     — Coverage: essay tokens captured in AF
        bertscore_f1         — Balance
        bertscore_per_arg_precision — Mean per-argument BERTScore precision
    """
    from bert_score import score as bert_score_fn

    arg_text = serialize_arguments(baf)

    # Empty BAF: return zeros
    if not arg_text.strip():
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_per_arg_precision": 0.0,
        }

    # Concatenated BERTScore: candidate=arg_text, reference=essay
    P, R, F1 = bert_score_fn(
        [arg_text],
        [essay_text],
        model_type=model_type,
        lang="en",
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=True,
        verbose=False,
    )

    # Per-argument variant: BERTScore(arg.text, essay) for each argument
    sorted_args = sorted(baf.arguments, key=lambda a: _natural_sort_key(a.id))
    arg_texts = [a.text for a in sorted_args]
    essay_refs = [essay_text] * len(arg_texts)

    Pa, _Ra, _Fa = bert_score_fn(
        arg_texts,
        essay_refs,
        model_type=model_type,
        lang="en",
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=True,
        verbose=False,
    )

    return {
        "bertscore_precision": float(P[0]),
        "bertscore_recall": float(R[0]),
        "bertscore_f1": float(F1[0]),
        "bertscore_per_arg_precision": float(Pa.mean()),
    }


# ---------------------------------------------------------------------------
# NLI AF-Faithfulness
# ---------------------------------------------------------------------------


def _nli_entailment_probs(
    premise: str,
    hypotheses: List[str],
    model_name: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cpu",
    batch_size: int = 16,
) -> List[float]:
    """Compute P(entailment) for each (premise, hypothesis) pair.

    The entailment label index is derived from the model's config.label2id
    rather than hardcoded, so this works with any MNLI-style model.
    """
    import torch

    if not hypotheses:
        return []

    tokenizer, model, device, ent_idx = _get_nli_model(model_name, device)

    probs_all: List[float] = []
    for i in range(0, len(hypotheses), batch_size):
        batch_hyps = hypotheses[i : i + batch_size]
        inputs = tokenizer(
            [premise] * len(batch_hyps),
            batch_hyps,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            entailment_probs = probs[:, ent_idx].cpu().tolist()
            probs_all.extend(entailment_probs)

    return probs_all


def nli_af_faithfulness(
    baf: BAF,
    essay_text: str,
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, float]:
    """NLI-based faithfulness evaluation.

    Serializes the AF into statements and checks P(entailment) for each
    against the essay.

    Returns dict with keys:
        nli_faithfulness_mean       — Mean P(entailment) over all statements
        nli_argument_faithfulness   — Mean over argument statements only
        nli_relation_groundedness   — Mean over relation statements only
        nli_faithfulness_min        — Min P(entailment) across all statements
        nli_faithfulness_std        — Std across all statements
        nli_faithfulness_frac_above_50 — Fraction of statements with P(ent) > 0.5
    """
    if not baf.arguments:
        return {
            "nli_faithfulness_mean": 0.0,
            "nli_argument_faithfulness": 0.0,
            "nli_relation_groundedness": 0.0,
            "nli_faithfulness_min": 0.0,
            "nli_faithfulness_std": 0.0,
            "nli_faithfulness_frac_above_50": 0.0,
        }

    statements = serialize_af_statements(baf)
    n_args = len(baf.arguments)
    # First n_args statements are argument statements, rest are relations

    if not statements:
        return {
            "nli_faithfulness_mean": 0.0,
            "nli_argument_faithfulness": 0.0,
            "nli_relation_groundedness": 0.0,
            "nli_faithfulness_min": 0.0,
            "nli_faithfulness_std": 0.0,
            "nli_faithfulness_frac_above_50": 0.0,
        }

    all_probs = _nli_entailment_probs(
        premise=essay_text,
        hypotheses=statements,
        model_name=nli_model,
        device=device,
        batch_size=batch_size,
    )

    arg_probs = all_probs[:n_args]
    rel_probs = all_probs[n_args:]

    all_arr = np.array(all_probs)

    return {
        "nli_faithfulness_mean": float(np.mean(all_arr)),
        "nli_argument_faithfulness": float(np.mean(arg_probs)) if arg_probs else 0.0,
        "nli_relation_groundedness": float(np.mean(rel_probs)) if rel_probs else 0.0,
        "nli_faithfulness_min": float(np.min(all_arr)),
        "nli_faithfulness_std": float(np.std(all_arr)),
        "nli_faithfulness_frac_above_50": float(np.mean(all_arr > 0.5)),
    }


# ---------------------------------------------------------------------------
# Per-essay combined evaluation
# ---------------------------------------------------------------------------


def evaluate_essay_neural(
    pred_baf: BAF,
    gold_baf: BAF,
    essay_text: str,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict:
    """Combined neural evaluation for one essay.

    Evaluates both the predicted BAF and the gold BAF (as ceiling).
    """
    pred_metrics = {}
    pred_metrics.update(
        bertscore_af_coverage(pred_baf, essay_text, bertscore_model, device, batch_size)
    )
    pred_metrics.update(
        nli_af_faithfulness(pred_baf, essay_text, nli_model, device, batch_size)
    )

    gold_metrics = {}
    gold_metrics.update(
        bertscore_af_coverage(gold_baf, essay_text, bertscore_model, device, batch_size)
    )
    gold_metrics.update(
        nli_af_faithfulness(gold_baf, essay_text, nli_model, device, batch_size)
    )

    return {"pred": pred_metrics, "gold": gold_metrics}


# ---------------------------------------------------------------------------
# Dataset-level aggregation with bootstrap CIs
# ---------------------------------------------------------------------------

# Metric keys to aggregate
_METRIC_KEYS = [
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "bertscore_per_arg_precision",
    "nli_faithfulness_mean",
    "nli_argument_faithfulness",
    "nli_relation_groundedness",
    "nli_faithfulness_frac_above_50",
]


def _bootstrap_neural_cis(
    per_essay: Dict[str, Dict],
    metric_keys: List[str],
    prefix: str = "pred",
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict:
    """Bootstrap 95% CIs for neural metric macro-averages."""
    if not per_essay:
        return {}

    rng = np.random.RandomState(seed)
    essay_ids = list(per_essay.keys())
    n = len(essay_ids)

    samples: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        for k in metric_keys:
            vals = [per_essay[essay_ids[i]][prefix][k] for i in indices]
            samples[k].append(float(np.mean(vals)))

    cis = {}
    for k in metric_keys:
        arr = np.array(samples[k])
        cis[k] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
    return cis


def evaluate_dataset_neural(
    predictions: Dict[str, BAF],
    golds: Dict[str, BAF],
    essay_texts: Dict[str, str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli",
    device: str = "cpu",
    batch_size: int = 16,
    bootstrap_n: int = 10000,
    bootstrap_seed: int = 42,
) -> Dict:
    """Aggregate neural evaluation across all essays.

    Parameters
    ----------
    predictions : dict[essay_id -> BAF]
    golds : dict[essay_id -> BAF]
    essay_texts : dict[essay_id -> str]
    """
    per_essay: Dict[str, Dict] = {}
    essay_ids = sorted(golds.keys())

    for i, essay_id in enumerate(essay_ids):
        gold_baf = golds[essay_id]
        essay_text = essay_texts[essay_id]

        if essay_id in predictions:
            pred_baf = predictions[essay_id]
        else:
            # Missing prediction: empty BAF
            pred_baf = BAF()

        logger.info(
            f"  [{i + 1}/{len(essay_ids)}] {essay_id}: "
            f"{len(pred_baf.arguments)} pred args, "
            f"{len(gold_baf.arguments)} gold args"
        )

        per_essay[essay_id] = evaluate_essay_neural(
            pred_baf=pred_baf,
            gold_baf=gold_baf,
            essay_text=essay_text,
            bertscore_model=bertscore_model,
            nli_model=nli_model,
            device=device,
            batch_size=batch_size,
        )

    # Macro-average across essays
    macro_pred = {}
    macro_gold = {}
    for k in _METRIC_KEYS:
        pred_vals = [per_essay[eid]["pred"][k] for eid in per_essay]
        gold_vals = [per_essay[eid]["gold"][k] for eid in per_essay]
        macro_pred[k] = float(np.mean(pred_vals)) if pred_vals else 0.0
        macro_gold[k] = float(np.mean(gold_vals)) if gold_vals else 0.0

    # Bootstrap CIs for prediction metrics
    cis = _bootstrap_neural_cis(
        per_essay, _METRIC_KEYS, prefix="pred",
        n_resamples=bootstrap_n, seed=bootstrap_seed,
    )

    return {
        "config": {
            "nli_model": nli_model,
            "bertscore_model": bertscore_model,
            "device": device,
        },
        "macro": macro_pred,
        "gold_ceiling": macro_gold,
        "bootstrap_95ci": cis,
        "per_essay": {
            eid: per_essay[eid]["pred"] for eid in per_essay
        },
        "per_essay_gold": {
            eid: per_essay[eid]["gold"] for eid in per_essay
        },
        "n_essays": len(golds),
        "n_predicted": sum(1 for eid in golds if eid in predictions),
    }
