# Experiment Plan: AF-Based Legal Judgment Prediction on JTD

## 1. Overview & Research Questions

We evaluate whether LLM-constructed argumentation frameworks (AFs) can serve as effective intermediate reasoning structures for legal judgment prediction on the Japanese Tort-Case Dataset (JTD). The LLM constructs the AF from structured case data, formal argumentation semantics compute on the AF, and predictions for downstream tasks are derived from the computed results.

**Core Research Questions:**

1. Does structuring legal reasoning as an AF improve judgment prediction over direct LLM prediction?
2. How does the choice of AF type (Dung vs. Bipolar) affect prediction quality?
3. Does quantitative argumentation (graded strength) improve over standard binary acceptability?
4. Which semantics (grounded vs. preferred) is more suitable for legal judgment tasks?
5. What is the value of formal semantic computation beyond structured LLM elicitation?

## 2. Dataset & Tasks

**Japanese Tort-Case Dataset (JTD):** 6,508 annotated civil tort cases under Japanese Civil Code Article 709, annotated by 41 legal experts.

| Statistic | Value |
|---|---|
| Total cases | 6,508 (training only; test held private) |
| Tort affirmed / denied | 2,569 (39.5%) / 3,939 (60.5%) |
| Plaintiff claims accepted | 12,960 / 25,231 (51.4%) |
| Defendant claims accepted | 11,431 / 22,458 (50.9%) |
| Avg plaintiff claims per case | 3.9 |
| Avg defendant claims per case | 3.5 |
| Avg undisputed facts per case | 1.3 |

**Each case contains:**
- `undisputed_facts` (U): facts agreed by both parties
- `plaintiff_claims` (P): arguments by plaintiff, each labeled `is_accepted`
- `defendant_claims` (D): arguments by defendant, each labeled `is_accepted`
- `court_decision` (T): boolean, whether the tort claim is affirmed

**Two tasks:**

1. **Rationale Extraction (RE):** Predict which claims from each party the court accepted (per-claim boolean).
2. **Tort Prediction (TP):** Predict whether the tort claim is affirmed (per-case boolean).

**Key structural observation:** The arguments (claims) are already extracted and given as structured input. The AF construction task is therefore about **identifying relations** between given arguments and (for quantitative variants) **assigning strengths**, not about extracting arguments from raw text.

## 3. Design Space

The baselines span a 2x2 factorial design over two dimensions, plus a non-AF control and an LLM-reasoning-with-AF condition:

| | **Standard (binary)** | **Quantitative (graded)** |
|---|---|---|
| **Dung AF (attack-only)** | B2 (grounded), B3 (preferred) | B5 (DF-QuAD) |
| **Bipolar AF (attack + support)** | B4 (grounded via reduction) | B6 (DF-QuAD) |

Plus:
- **B0:** Direct LLM prediction (no AF) — ablation control
- **B1:** AF-guided LLM prediction (AF as structured chain-of-thought) — isolates the value of formal computation
- **BH:** Heuristic QBAF + DF-QuAD (no LLM relation extraction) — isolates the value of LLM-targeted relations

**Key comparisons enabled:**

| Comparison | Tests |
|---|---|
| B0 vs. B1 | Value of AF-structured elicitation |
| B0 vs. B2-B6 | Value of AF + formal semantics |
| B1 vs. B2-B6 | Value of formal computation over LLM intuition |
| B2 vs. B4 | Dung vs. Bipolar (standard) |
| B5 vs. B6 | Dung vs. Bipolar (quantitative) |
| B2 vs. B5 | Standard vs. quantitative (Dung) |
| B4 vs. B6 | Standard vs. quantitative (Bipolar) |
| B2 vs. B3 | Grounded vs. preferred semantics |
| BH vs. B6 | Heuristic vs. LLM-targeted relations (QBAF) |
| BH vs. B0 | Heuristic AF computation vs. direct LLM |

## 4. Baseline Methods

### B0: Direct LLM Prediction (No AF)

**Pipeline:** Input (U, P, D) → LLM → RE predictions + TP prediction

The LLM is given the case data and directly asked to predict: (a) which plaintiff and defendant claims are accepted, and (b) whether the tort is affirmed. No AF is constructed. This serves as the ablation control to measure the value added by AF-based reasoning.

### B1: AF-Guided LLM Prediction (AF as Structured CoT)

**Pipeline:** Input → LLM (Step 1: construct AF) → AF → LLM (Step 2: predict with AF) → RE + TP

The LLM first constructs an AF (Dung-style with attack relations) from the case data. The constructed AF is then provided back to the LLM as additional context, and the LLM is asked to reason with this structured representation to predict RE and TP. No formal argumentation semantics are applied — the LLM performs the reasoning over the AF structure using its own judgment.

This baseline isolates whether the structured elicitation (making the LLM think in AF terms) provides value, independently of formal semantic computation.

### B2: Standard Dung AF + Grounded Semantics

**Pipeline:** Input → LLM → Dung AF → Grounded extension → RE + TP

**AF type:** Dung AF (attack-only). Arguments are all plaintiff and defendant claims. The LLM identifies directed attack relations between claims (typically cross-party: defendant claims attacking plaintiff claims and vice versa). Undisputed facts are provided as context for identifying attacks but not included as arguments in the AF (since in Dung AF without support, they cannot contribute to the reasoning).

**Semantics:** Grounded semantics — the unique minimal complete extension. An argument is acceptable iff it belongs to the grounded extension. Grounded semantics is skeptical: arguments involved in unresolved conflicts remain undecided (out of the extension). In legal cases where plaintiff and defendant claims mutually attack each other, this may result in a conservative (small) extension.

**Prediction mapping:** See Section 7.

### B3: Standard Dung AF + Preferred Semantics

**Pipeline:** Input → LLM → Dung AF → Preferred extensions → RE + TP

Same AF construction as B2. Uses preferred semantics instead: maximal admissible sets. Multiple preferred extensions may exist.

**Acceptance criteria:** Both credulous and skeptical acceptance are reported:
- *Credulous:* a claim is accepted if it belongs to at least one preferred extension. More permissive; may better capture the legal intuition that the court "chooses" between competing positions.
- *Skeptical:* a claim is accepted only if it belongs to every preferred extension. More conservative; only accepts claims that survive under all maximal admissible interpretations.

### B4: Standard Bipolar AF + Grounded Semantics (via Reduction)

**Pipeline:** Input → LLM → BAF → Reduction to Dung → Grounded extension → RE + TP

**AF type:** Bipolar Argumentation Framework. Arguments include all plaintiff claims, defendant claims, AND undisputed facts. The LLM identifies both **attack** and **support** relations:
- Attack: cross-party claims rebutting each other
- Support: undisputed facts supporting claims, within-party claims reinforcing each other

**Semantics:** BAF is reduced to a standard Dung AF using the Cayrol & Lagasquie-Schiex coalition reduction:
1. Keep all original attack relations.
2. For each support relation (a supports b): any argument c that attacks a also attacks b (secondary attack). Intuition: undermining a supporter weakens the supported claim.
3. Compute grounded extension on the resulting Dung AF.

This tests whether explicitly modeling support relationships (especially from undisputed facts to claims) improves over attack-only reasoning.

### B5: Quantitative AF (QuAF) + DF-QuAD

**Pipeline:** Input → LLM → QuAF (arguments with strengths, weighted attacks) → DF-QuAD → final strengths → RE + TP

**AF type:** Quantitative Argumentation Framework (attack-only). Arguments are plaintiff and defendant claims, each with an LLM-assigned **base strength** s₀(a) ∈ [0,1] reflecting how convincing the argument is in isolation (prior to considering interactions). The LLM identifies attack relations between claims.

**Semantics:** DF-QuAD algorithm (Rago et al., 2016). Iteratively computes final argument strengths by aggregating the effect of attackers:

```
For each argument a:
  att_agg(a) = 1 - ∏(1 - strength(aᵢ)) for all attackers aᵢ of a

  if att_agg(a) <= 0:
    strength(a) = s₀(a)
  else:
    strength(a) = s₀(a) * (1 - att_agg(a))
```

Iterate until convergence (strengths stabilize). Final strengths in [0,1].

This tests whether graded assessment of argument strength and continuous aggregation improve over binary in/out decisions.

### B6: Quantitative Bipolar AF (QBAF) + DF-QuAD

**Pipeline:** Input → LLM → QBAF (arguments with strengths, attacks + supports) → DF-QuAD → final strengths → RE + TP

**AF type:** Quantitative Bipolar AF. Same as B5 but with both attack and support relations. Arguments include plaintiff claims, defendant claims, and undisputed facts, each with LLM-assigned base strengths.

**Semantics:** DF-QuAD with support (Baroni et al., 2019):

```
For each argument a:
  att_agg(a) = 1 - ∏(1 - strength(aᵢ)) for all attackers aᵢ of a
  sup_agg(a) = 1 - ∏(1 - strength(sⱼ)) for all supporters sⱼ of a

  if sup_agg(a) >= att_agg(a):
    strength(a) = s₀(a) + (1 - s₀(a)) * (sup_agg(a) - att_agg(a))
  else:
    strength(a) = s₀(a) * (1 - (att_agg(a) - sup_agg(a)))
```

This is the most expressive variant: it captures both support/attack structure and graded argument strength.

### BH: Heuristic QBAF + DF-QuAD (No LLM Relation Extraction)

**Pipeline:** Input → Deterministic AF construction (no LLM) → QBAF → DF-QuAD → final strengths → RE + TP

**AF type:** Quantitative Bipolar AF constructed via simple structural heuristics, with no LLM call for relation identification:
- **Arguments:** All P-claims, D-claims, and U-facts.
- **Base strengths:** Uniform s₀ = 0.5 for all P-claims and D-claims; s₀ = 1.0 for U-facts (axioms).
- **Attacks:** Complete bipartite — every P-claim attacks every D-claim, and every D-claim attacks every P-claim.
- **Supports:** Every U-fact supports every P-claim and every D-claim.

**Semantics:** DF-QuAD with support (same as B6).

This baseline requires **zero LLM calls** for AF construction. It tests whether LLM-targeted relation identification (as in B5/B6) adds value over a naive structural prior where all cross-party claims oppose each other. Under this heuristic with uniform strengths, the outcome is primarily driven by the count of arguments on each side, making it an informative lower bound. Note: this baseline is only meaningful for the quantitative variant (DF-QuAD), since a complete bipartite attack graph would produce an empty grounded extension in standard Dung semantics (all arguments in mutual attack cycles).

## 5. AF Construction via LLM

Since arguments (claims) are given in the dataset, the LLM's task is to identify **relations** between them and (for quantitative variants) assign **strengths**.

### 5.1 Prompt Structure

All AF construction prompts follow this structure:

1. **System prompt:** Defines the AF type, explains the relation semantics (attack, support), and specifies the output format.
2. **User prompt:** Provides:
   - Undisputed facts (U) as context
   - Plaintiff claims (P) with IDs
   - Defendant claims (D) with IDs
   - Instruction to identify relations (and strengths)
3. **Output format:** Structured JSON

### 5.2 Relation Definitions (in Prompts)

**Attack (for all AF types):**
> Argument A attacks argument B if A provides a reason, evidence, or legal basis that undermines, contradicts, or rebuts the conclusion of B. Attacks are typically cross-party (defendant claims rebut plaintiff claims, and vice versa), but may also occur within the same party if claims are logically inconsistent.

**Support (for BAF types only):**
> Argument A supports argument B if A provides evidence, reasoning, or factual basis that strengthens or substantiates B. Support commonly occurs: (1) from undisputed facts to claims they substantiate, and (2) between claims within the same party that reinforce each other.

### 5.3 Output Schemas

**Standard Dung (B2, B3):**
```json
{
  "attacks": [
    {"source": "p0", "target": "d2"},
    {"source": "d1", "target": "p0"}
  ]
}
```

**Standard BAF (B4):**
```json
{
  "attacks": [...],
  "supports": [
    {"source": "u0", "target": "p1"},
    {"source": "p0", "target": "p2"}
  ]
}
```

**Quantitative Dung (B5):**
```json
{
  "argument_strengths": {
    "p0": 0.8, "p1": 0.6, "d0": 0.7
  },
  "attacks": [...]
}
```

**Quantitative BAF (B6):**
```json
{
  "argument_strengths": {
    "u0": 0.9, "p0": 0.8, "d0": 0.7
  },
  "attacks": [...],
  "supports": [...]
}
```

Argument IDs use the convention: `u{i}` for undisputed facts, `p{i}` for plaintiff claims, `d{i}` for defendant claims.

### 5.4 Treatment of Undisputed Facts (U)

Undisputed facts are agreed upon by both parties and accepted by the court. They are treated as **axioms** in all AF types:

- **Status:** Always accepted. In standard AF (B4), U-arguments are unattackable — no attack may target them. In quantitative AF (B6), U-arguments have fixed base strength s₀ = 1.0, and no attacks may target them.
- **Relation constraints:** U-arguments may only serve as **sources** of support relations (U → claim). They do not attack other arguments, and other arguments do not attack them.
- **Dung AF (B2, B3, B5):** Since Dung AF has no support relations and U cannot participate in attacks, undisputed facts are provided as **context** in the LLM prompt but are not included as arguments in the AF.
- **BAF (B4, B6):** U-arguments are included in the AF and may support claims from either party. Because U is unattackable, the coalition reduction (Section 6.3) will not generate secondary attacks through U (since no argument c attacks any u ∈ U). However, the support from U to claims still strengthens those claims in the quantitative variant (B6) via the DF-QuAD support aggregation.

## 6. Argumentation Semantics & Computation

### 6.1 Grounded Semantics (B2, B4)

The grounded extension is computed via iterative fixpoint:

```
IN₀ = {a ∈ A : a has no attackers}
repeat:
  OUT = {a ∈ A : ∃ attacker b of a such that b ∈ IN}
  IN = IN ∪ {a ∈ A : all attackers of a are in OUT}
until no change
```

Arguments in `IN` are accepted; arguments in `OUT` are rejected; all others are undecided.

### 6.2 Preferred Semantics (B3)

A set S ⊆ A is **admissible** if:
1. S is conflict-free (no two arguments in S attack each other)
2. S defends all its members (for every attacker of a ∈ S, some b ∈ S attacks that attacker)

A **preferred extension** is a maximal (w.r.t. set inclusion) admissible set. Computed via standard backtracking algorithms.

Both acceptance modes are reported:
- **Credulous acceptance:** a is accepted if a ∈ E for some preferred extension E.
- **Skeptical acceptance:** a is accepted if a ∈ E for every preferred extension E.

### 6.3 BAF Coalition Reduction (B4)

Given BAF = (A, att, sup):
1. Initialize Dung AF' = (A, att' = att)
2. For each support (a, b) ∈ sup:
   - For each attack (c, a) ∈ att:
     - Add (c, b) to att' (secondary attack: attacking a's supporter threatens a)
3. Return AF' = (A, att')

Then apply grounded semantics on AF'.

### 6.4 DF-QuAD (B5, B6)

See formulas in Section 4 (B5 and B6 descriptions). Iterate until max change in any argument's strength < ε = 0.001, or a maximum of 100 iterations.

## 7. Prediction Mapping (AF → RE/TP)

### 7.1 Rationale Extraction (RE)

**Standard AF (B2, B3, B4):**
- Claim is predicted as **accepted** iff the corresponding argument is in the computed extension.
- Claims that are "undecided" (not in IN or OUT for grounded semantics) are predicted as **rejected** under the default convention. As a sensitivity check, results are also reported under an "undecided-excluded" convention where undecided claims are omitted from the RE scoring (metrics computed only over IN/OUT arguments). This prevents the undecided-to-rejected mapping from systematically penalizing grounded semantics.

**Quantitative AF (B5, B6):**
- Claim is predicted as **accepted** iff final strength > τ.
- The threshold τ is **tuned per CV fold**: a grid search over τ ∈ {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7} is performed on each fold's training split, selecting the τ that maximizes RE Micro-F1, and evaluating on the held-out split. The full τ-vs-performance curve is also reported as a sensitivity analysis (Section 10.3).

### 7.2 Tort Prediction (TP)

Two strategies are evaluated as equal alternatives; both are reported and compared:

**Strategy A — Meta-argument:**
Introduce a meta-argument T representing "the tort is affirmed." Connect it to the AF:
- *Standard Dung (B2, B3):* Each defendant claim attacks T. Each plaintiff claim attacks each defendant claim that attacks T (defense of T). T's membership in the extension determines the TP prediction.
- *Standard BAF (B4):* Each plaintiff claim supports T, each defendant claim attacks T.
- *Quantitative (B5, B6):* T has base strength 0.5 (neutral). P-claims support T, D-claims attack T. T's final DF-QuAD strength > 0.5 → affirm.

**Strategy B — Aggregation over RE:**
- *Standard AF:* Tort affirmed if (# accepted P-claims) / (# total P-claims) > 0.5.
- *Quantitative AF:* Tort affirmed if mean final strength of P-claims > 0.5.

Strategy A leverages AF structure more directly; Strategy B is simpler and more transparent. Neither is assumed superior a priori — the comparison itself is an analysis target (Section 10.4).

## 8. Evaluation Metrics

### 8.1 Rationale Extraction (RE)

| Metric | Description |
|---|---|
| Claim-level Micro-F1 | Pool all claims across cases, compute F1 for "accepted" class |
| Case-averaged F1 | Per-case F1, then average across cases |
| Per-party Micro-F1 | Separate F1 for plaintiff claims and defendant claims |
| Claim-level Accuracy | % of individual claims correctly predicted |

### 8.2 Tort Prediction (TP)

| Metric | Description |
|---|---|
| Accuracy | % of cases with correct tort prediction |
| Macro-F1 | Average of F1 for "affirmed" and "denied" classes |
| Precision / Recall | For the "affirmed" class |

### 8.3 Joint Evaluation

| Metric | Description |
|---|---|
| Joint Accuracy | % of cases where BOTH TP and ALL RE predictions are correct |
| RE→TP Consistency | % of cases where TP prediction derived from RE matches direct TP prediction |

## 9. Experiment Protocol

### 9.1 Data Split

Since only the training set (6,508 cases) is available, we use stratified 5-fold cross-validation, stratified on `court_decision`, to evaluate all methods. Results are reported as mean ± std across folds.

### 9.2 LLM Models

Test with 2-3 models to assess sensitivity to model choice:
- **Frontier:** Claude Sonnet 4.6 or GPT-4o
- **Strong reasoning:** DeepSeek-R1 or Qwen-3 (strong at reasoning, cost-effective)
- **Smaller/faster:** Claude Haiku 4.5 or GPT-4o-mini (cost comparison)

All accessed via OpenRouter (already supported by codebase in `src/llm_client.py`).

### 9.3 Sampling

Given the dataset size (6,508 cases) and per-case API cost, run on a **stratified subsample** for development and initial comparison:
- **Development:** 200 cases (stratified)
- **Full evaluation:** All 6,508 cases (for final results, or as budget allows)

### 9.4 Reproducibility

- Fixed random seeds for all sampling
- All LLM outputs (raw JSON), constructed AFs, computed extensions/strengths, and predictions saved per case
- Temperature = 0 for LLM calls (deterministic decoding where supported)
- Experiment configuration (model, method, prompt variant, seed) saved with results

## 10. Expected Analyses

### 10.1 Main Results Table

Table with all baselines (B0-BH) x metrics (RE Micro-F1, RE Case-averaged F1, TP Accuracy, TP F1). Statistical significance via paired bootstrap test.

### 10.2 Ablation: AF Structure Quality

- Number of relations per case (attacks, supports) across methods
- Relation density: how many cross-party vs within-party relations
- Consistency: do different LLM models produce similar AF structures?

### 10.3 Ablation: Threshold Sensitivity (Quantitative Methods)

For B5 and B6, vary the acceptance threshold τ and report RE/TP metrics as a function of τ.

### 10.4 Ablation: TP Prediction Strategy

Compare Strategy A (meta-argument) vs Strategy B (RE aggregation) across all methods.

### 10.5 Case Complexity Analysis

Stratify results by:
- Number of claims (small: ≤4, medium: 5-8, large: >8)
- Presence/absence of undisputed facts
- Class balance within case (ratio of accepted to total claims)

### 10.6 Error Analysis

- Cases where AF-based methods outperform direct LLM (B0): what AF structure features correlate with improvement?
- Cases where direct LLM outperforms AF-based: what goes wrong in AF construction or semantics?
- Qualitative examples of well-constructed vs poorly-constructed AFs

### 10.7 AF Semantics Behavior

- For grounded semantics: how many claims end up "undecided"? Does this hurt RE performance?
- For preferred semantics: how many preferred extensions exist per case? How do credulous vs. skeptical acceptance differ in precision/recall trade-off?
- For DF-QuAD: distribution of final strengths (well-separated or clustered near 0.5?)

## 11. Implementation Notes

### 11.1 AF Solver

For grounded and preferred semantics, either use an existing Python library (e.g., `py-arg`) or implement directly (grounded fixpoint is straightforward; preferred extension enumeration via backtracking). DF-QuAD is implemented as iterative strength updates (~20 lines of code).

**Preferred semantics timeout:** Although JTD cases are typically small (~7 claims), long-tail cases with many claims may cause preferred extension enumeration to be expensive. A 30-second per-case timeout is enforced; if exceeded, fall back to grounded semantics for that case and flag it in the results.

### 11.2 Codebase Integration

The existing codebase (`src/`) provides infrastructure for LLM client, output parsing, and evaluation. Key extensions needed:
- **Data loader** for JTD format (JSONL with claims + decision)
- **Prompt templates** for AF construction (per AF type)
- **AF computation module** (grounded, preferred, BAF reduction, DF-QuAD)
- **Prediction mapping module** (extension/strength → RE/TP predictions)
- **Evaluation module** for RE and TP metrics

### 11.3 Language Considerations

All case data is in Japanese. The LLM must process Japanese legal text. Frontier LLMs (Claude, GPT-4o) have strong Japanese language capabilities. Prompts can be in English (with Japanese case content) or fully in Japanese; we use English prompts for reproducibility and readability.

## 12. Summary of Baselines

| ID | Method | AF Construction (LLM task) | Computation | Key Test |
|---|---|---|---|---|
| B0 | Direct LLM | None | None | Ablation control |
| B1 | AF-Guided LLM | Identify attacks | LLM reasons with AF | Value of formal computation |
| B2 | Dung + Grounded | Identify attacks | Grounded extension | Standard AF baseline |
| B3 | Dung + Preferred | Identify attacks | Preferred extensions | Skeptical vs credulous |
| B4 | BAF + Grounded | Identify attacks + supports | Coalition reduction + grounded | Value of support relations |
| B5 | QuAF + DF-QuAD | Identify attacks + assign strengths | DF-QuAD | Value of quantification |
| B6 | QBAF + DF-QuAD | Identify attacks + supports + assign strengths | DF-QuAD with support | Full expressiveness |
| BH | Heuristic QBAF | None (complete bipartite, uniform strengths) | DF-QuAD with support | Value of LLM relation extraction |

## References

- Baroni, P., Rago, M., & Toni, F. (2019). From fine-grained properties to broad principles for gradual argumentation: A principled spectrum. International Journal of Approximate Reasoning.
- Cayrol, C., & Lagasquie-Schiex, M.C. (2005). On the acceptability of arguments in bipolar argumentation frameworks. ECSQARU.
- Dung, P.M. (1995). On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial Intelligence.
- Rago, M., Toni, F., Aurisicchio, M., & Baroni, P. (2016). Discontinuity-free decision support with quantitative argumentation debates. KR.
- Yamada, H., et al. (2024). Japanese Tort-case Dataset for Rationale-supported Legal Judgment Prediction. Artificial Intelligence and Law.
