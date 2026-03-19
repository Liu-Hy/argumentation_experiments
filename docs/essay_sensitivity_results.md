# Sensitivity Experiment Results

This document summarizes the findings from the sensitivity experiments described in `essay_sensitivity_experiments.md`. All validation-set experiments use 40 held-out training essays (stratified split, seed=42). Test-set experiments use the official 80-essay split.

## Phase 0: CoT Rerun (Test Set)

The original CoT prompt was thesis-centric ("identify the author's main thesis..."), contradicting the BAF formulation where all components are treated uniformly. After fixing the prompt to align with the task definition, results split into two categories.

### Models with 100% parse rate (Claude Haiku 4.5, Claude Sonnet 4.5, Gemini 3 Flash)

| Model | Method | mi.Arg F1 | mi.Sup F1 | mi.Att F1 | mi.RelM F1 | ma.RelM F1 | Parse % |
|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | fs_e2e | **0.868** | 0.332 | 0.083 | 0.208 | 0.238 | 1.000 |
| | fs_cot_e2e | 0.847 | 0.330 | **0.112** | **0.221** | **0.246** | 1.000 |
| Claude Sonnet 4.5 | fs_e2e | **0.861** | **0.358** | **0.191** | **0.275** | **0.299** | 1.000 |
| | fs_cot_e2e | 0.849 | 0.344 | 0.146 | 0.245 | 0.264 | 1.000 |
| Gemini 3 Flash | fs_e2e | **0.869** | **0.341** | 0.098 | **0.220** | **0.248** | 1.000 |
| | fs_cot_e2e | 0.837 | 0.316 | **0.153** | 0.235 | 0.226 | 1.000 |

CoT is roughly neutral to slightly harmful overall. It consistently hurts argument F1 (by 1--3%) and has mixed effects on relations. The best model (Sonnet 4.5) gets worse with CoT: macro Relation Macro F1 drops from 0.299 to 0.264. One partial benefit: CoT sometimes improves attack detection specifically (e.g., +0.05 attack F1 for Gemini Flash), likely because the fixed prompt explicitly draws attention to opposing viewpoints.

### Models with catastrophic parse failure (Gemini 3 Pro, GPT-5.2, GPT-5-mini)

| Model | Method | mi.Arg F1 | ma.RelM F1 | Parse % |
|---|---|---|---|---|
| Gemini 3 Pro | fs_e2e | 0.819 | 0.202 | 0.988 |
| | fs_cot_e2e | 0.172 | 0.040 | **0.125** |
| GPT-5.2 | fs_e2e | 0.841 | 0.260 | 1.000 |
| | fs_cot_e2e | 0.322 | 0.053 | **0.325** |
| GPT-5-mini | fs_e2e | 0.807 | 0.208 | 0.988 |
| | fs_cot_e2e | 0.254 | 0.052 | **0.188** |

These models cannot maintain valid JSON output while also performing chain-of-thought reasoning. This is an instruction-following failure, not a reasoning failure.

### Phase 0 conclusion

The fixed CoT prompt did not rescue the method. CoT hurts or is neutral for models that can handle the formatting, and is catastrophic for those that cannot. The original finding that CoT underperforms fs_e2e is confirmed -- it was not an artifact of the old prompt.

---

## Phase 1: Prompt Variant Selection (Val Set)

Design: 3 prompt variants x 2 models (Gemini 3 Flash, Claude Haiku 4.5) x 5 methods.

### Cross-variant comparison (averaged across 5 methods per model)

| Model | Variant | Avg mi.Arg F1 | Avg mi.RelM F1 | Avg ma.RelM F1 |
|---|---|---|---|---|
| Gemini 3 Flash | **default** | **0.842** | **0.218** | **0.222** |
| | enhanced | 0.820 | 0.198 | 0.202 |
| | minimal | 0.819 | 0.203 | 0.211 |
| Claude Haiku 4.5 | **default** | 0.844 | 0.194 | **0.228** |
| | enhanced | **0.849** | 0.189 | 0.212 |
| | minimal | 0.815 | 0.171 | 0.209 |

### Per-method breakdown (mi.RelM F1 -- micro Relation Macro F1)

| Method | Gemini default | Gemini enhanced | Gemini minimal | Claude default | Claude enhanced | Claude minimal |
|---|---|---|---|---|---|---|
| zs_e2e | **0.204** | 0.172 | 0.152 | **0.193** | 0.182 | 0.167 |
| fs_e2e | **0.245** | 0.203 | 0.236 | **0.219** | 0.184 | 0.205 |
| fs_cot_e2e | 0.217 | 0.207 | **0.224** | 0.209 | **0.212** | 0.187 |
| zs_pipe | 0.203 | **0.215** | 0.186 | **0.151** | 0.174 | 0.132 |
| fs_pipe | **0.222** | 0.191 | 0.218 | **0.200** | 0.193 | 0.164 |

### Phase 1 findings

- **Default wins overall.** Neither enhanced nor minimal consistently improves relation metrics. The enhanced variant's extra instructions appear to confuse rather than help, perhaps over-constraining the models.
- Enhanced hurts attack F1 for Claude (e.g., fs_e2e attack drops from 0.138 to 0.079), despite explicitly including attack awareness hints.
- Minimal performs surprisingly well on few-shot methods (close to default), suggesting that examples matter more than prompt wording when examples are present. But minimal is clearly worst for zero-shot (Gemini zs_e2e argument F1 drops from 0.780 to 0.691).
- Parse success rate is ~1.0 across all variants -- prompt changes do not affect output compliance.

### Phase 1 decision

Keep the default prompt. No variant clears the >= 3% improvement threshold on Relation Macro F1.

---

## Phase 2: N-Shot Ablation (Val Set)

Design: n in {1, 3, 5, 10}, fs_e2e method, default prompt, both models.

### Results

| n | Gemini mi.Arg | Gemini mi.RelM | Gemini ma.RelM | Claude mi.Arg | Claude mi.RelM | Claude ma.RelM |
|---|---|---|---|---|---|---|
| 1 | 0.800 | 0.200 | 0.210 | 0.856 | 0.216 | 0.249 |
| 3 | 0.860 | 0.245 | 0.232 | 0.887 | 0.219 | 0.255 |
| 5 | 0.863 | **0.290** | **0.279** | **0.888** | 0.248 | 0.282 |
| 10 | 0.855 | 0.275 | 0.270 | 0.882 | **0.258** | **0.298** |

### Phase 2 findings

- **More examples consistently improve relation extraction**, especially support F1 and attack F1. This is one of the clearest signals in the experiments.
- Argument F1 plateaus around n=3--5 and does not benefit from additional examples.
- For Gemini, n=5 is the sweet spot (n=10 is slightly lower, possibly due to context length pressure). For Claude, n=10 is best, likely because Haiku handles longer contexts more gracefully.
- The gains are substantial: n=10 yields +4--7% macro Rel-Macro F1 over n=3 for both models.

### Phase 2 takeaway

n=3 is suboptimal for relation extraction. Increasing to n=5 or n=10 provides a meaningful and consistent improvement.

---

## Phase 3: Few-Shot Example Sensitivity (Val Set)

Design: 5 random seeds (0--4) for selecting n=10 examples, fs_e2e, default prompt.

### Gemini 3 Flash (5 seeds)

| Seed | mi.Arg F1 | mi.Sup F1 | mi.Att F1 | mi.RelM F1 | ma.RelM F1 |
|---|---|---|---|---|---|
| 0 | 0.838 | 0.350 | 0.194 | 0.272 | 0.278 |
| 1 | 0.875 | 0.364 | 0.147 | 0.256 | 0.266 |
| 2 | 0.829 | 0.347 | 0.170 | 0.258 | 0.269 |
| 3 | **0.852** | **0.383** | **0.264** | **0.324** | **0.324** |
| 4 | 0.835 | 0.350 | 0.188 | 0.269 | 0.271 |
| **Mean** | **0.846** | **0.359** | **0.192** | **0.276** | **0.282** |
| **Std** | **0.017** | **0.014** | **0.042** | **0.026** | **0.022** |

### Claude Haiku 4.5 (5 seeds)

| Seed | mi.Arg F1 | mi.Sup F1 | mi.Att F1 | mi.RelM F1 | ma.RelM F1 |
|---|---|---|---|---|---|
| 0 | 0.888 | 0.329 | 0.174 | 0.252 | 0.307 |
| 1 | 0.873 | 0.337 | 0.039 | 0.188 | 0.262 |
| 2 | 0.887 | 0.347 | 0.128 | 0.237 | 0.270 |
| 3 | 0.885 | 0.336 | **0.174** | **0.255** | **0.310** |
| 4 | 0.869 | 0.291 | 0.159 | 0.225 | 0.254 |
| **Mean** | **0.880** | **0.328** | **0.135** | **0.231** | **0.281** |
| **Std** | **0.008** | **0.021** | **0.055** | **0.027** | **0.025** |

### Phase 3 findings

- **Argument F1 is very stable** across different example sets (std < 0.02). The specific choice of examples barely matters for span identification.
- **Attack F1 is the most volatile metric** by far (std 0.04--0.06). This is expected: there are only ~17 gold attack relations in the 40-essay val set, so a few swings in attack TP/FP change F1 dramatically.
- Claude seed 1 is a dramatic outlier: attack F1 = 0.039 (vs. mean 0.135), suggesting that particular example set happened to lack attack-heavy essays.
- **Seed 3 is the best for both models** (Gemini ma.RelM = 0.324, Claude ma.RelM = 0.310).
- The original heuristic-selected examples (used in the n=10 non-seeded run) fall within the seed-variation range, confirming the selection was reasonable.

### Phase 3 takeaway

Results are stable enough for argument identification, but attack F1 has high variance due to class rarity. Report mean +/- std across seeds. Note that attack metric instability is inherent to the small attack sample, not a flaw in the method.

---

## Phase 4: Run-to-Run Variance (Test Set)

Design: 3 repeated runs of fs_e2e (temperature=0) on the official test set.

### Claude Haiku 4.5 (original + 3 runs)

| Run | mi.Arg F1 | mi.Sup F1 | mi.Att F1 | ma.Arg F1 | ma.Sup F1 | ma.Att F1 | ma.RelM F1 | Parse % |
|---|---|---|---|---|---|---|---|---|
| Original | 0.868 | 0.332 | 0.083 | 0.866 | 0.337 | 0.057 | 0.238 | 1.000 |
| Run 1 | 0.869 | 0.341 | 0.098 | 0.867 | 0.343 | 0.067 | 0.248 | 1.000 |
| Run 2 | 0.858 | 0.329 | 0.081 | 0.852 | 0.333 | 0.058 | 0.236 | 0.988 |
| Run 3 | 0.866 | 0.328 | 0.083 | 0.864 | 0.332 | 0.058 | 0.238 | 1.000 |
| **Mean** | **0.865** | **0.332** | **0.086** | **0.862** | **0.336** | **0.060** | **0.240** | **0.997** |
| **Std** | **0.004** | **0.006** | **0.007** | **0.006** | **0.004** | **0.004** | **0.005** | **0.006** |

### Gemini 3 Flash (original + 3 runs)

| Run | mi.Arg F1 | mi.Sup F1 | mi.Att F1 | ma.Arg F1 | ma.Sup F1 | ma.Att F1 | ma.RelM F1 | Parse % |
|---|---|---|---|---|---|---|---|---|
| Original | 0.868 | 0.332 | 0.083 | 0.866 | 0.337 | 0.057 | 0.238 | 1.000 |
| Run 1 | 0.838 | 0.354 | 0.190 | 0.837 | 0.347 | 0.119 | 0.266 | 0.988 |
| Run 2 | 0.845 | 0.353 | 0.179 | 0.850 | 0.350 | 0.115 | 0.266 | 1.000 |
| Run 3 | 0.843 | 0.355 | 0.179 | 0.849 | 0.353 | 0.115 | 0.267 | 1.000 |
| **Mean** | **0.849** | **0.349** | **0.158** | **0.851** | **0.347** | **0.102** | **0.259** | **0.997** |
| **Std** | **0.012** | **0.010** | **0.045** | **0.011** | **0.006** | **0.027** | **0.013** | **0.006** |

### Phase 4 findings

- **Claude Haiku 4.5 is highly deterministic**: all metric standard deviations are below 0.01. Temperature=0 effectively gives reproducible outputs.
- **Gemini 3 Flash has ~3x more variance**, especially for attack F1 (std = 0.045 micro, 0.027 macro). The original run appears to be a slight outlier with unusually low attack F1 (0.083) compared to runs 1--3 which cluster around 0.18.
- For both models, core metrics (argument F1, support F1) are stable enough to draw reliable conclusions.

### Phase 4 takeaway

Results are stable across repeated runs. Suitable for a one-sentence note in the paper: "Results were stable across 3 independent runs, with Macro Relation F1 varying by at most +/-1.3%."

---

## Overall Synthesis

| Finding | Evidence | Implication |
|---|---|---|
| Default prompt is adequate | Phase 1: no variant improves >= 3% | No re-running needed |
| CoT does not help (and can be catastrophic) | Phase 0: confirmed with fixed prompt | Robust finding, not a prompt artifact |
| More few-shot examples improve relation extraction | Phase 2: n=5--10 gives +4--7% macro RelM F1 over n=3 | Consider n=5--10 for final runs |
| Example selection has moderate impact | Phase 3: std ~0.025 on Rel-Macro F1 | Report mean +/- std; heuristic selection is reasonable |
| Run-to-run variance is low | Phase 4: std < 0.013 on Rel-Macro F1 | Single-run results are trustworthy |
| Attack F1 is inherently noisy | Phases 3--4: std 0.03--0.06 on attack F1 | Acknowledge in paper; due to class rarity, not method instability |

### Open decision

The main actionable choice: whether to re-run the full test-set experiments with n=5 or n=10 (since Phase 2 showed clear gains over n=3), or keep n=3 for the main results and report the n-shot ablation as supplementary analysis.
