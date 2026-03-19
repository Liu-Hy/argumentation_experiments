# Pairwise Relation Classification: Results

## Key Finding

Pairwise decomposition **does not improve** relation extraction over the bulk baseline. Both the 3-pair few-shot and graph-context variants suffer from severe over-prediction, though showing the full training graph provides a measurable improvement.

## 1. Results Summary (Gemini 3 Flash)

All methods use gold arguments. The 3-pair and graph-context pairwise results are evaluated on the same 4 essays for apples-to-apples comparison; the bulk baseline is from the full 80-essay archived run.

| Metric | gold_fs (bulk, 80 ess.) | gold_fs_pairwise (3-pair) | gold_fs_pw_graph |
|--------|------------------------|--------------------------|------------------|
| **Support P / R / F1** | 0.468 / 0.601 / **0.526** | 0.167 / 0.824 / **0.277** | 0.208 / 0.882 / **0.337** |
| **Attack P / R / F1** | 0.186 / 0.429 / **0.259** | 0.000 / 0.000 / **0.000** | 0.182 / 0.667 / **0.286** |
| **Relation micro F1** | **0.506** | **0.257** | **0.333** |
| **Relation macro F1** | **0.392** | **0.139** | **0.311** |
| **Over-prediction ratio** | ~1.4x gold | 4.9x gold | 4.2x gold |

## 2. Analysis

### 2.1 Graph context helps, but over-prediction persists

Showing the full training graph (all arguments + all relations from one training essay) provides clear improvements over the 3-pair variant:

- **Macro Relation F1 more than doubles**: 0.139 → 0.311 (+0.172)
- **Attack F1 restored from zero**: 0.000 → 0.286. The 3-pair prompt showed no attack examples strong enough to learn from; the full graph shows attacks in structural context.
- **Over-prediction reduced**: 4.9x → 4.2x gold, though still far above bulk's ~1.4x.

The improvement confirms that **prior bias was a real confound** in the 3-pair prompt (67% relation rate in examples vs. ~8% true base rate). The graph context explicitly shows the true density (~10/105 ≈ 9.5%), which the model partially incorporates.

### 2.2 Still far below the bulk baseline

Despite the improvement, graph-context pairwise (Micro F1 = 0.333) still substantially underperforms the bulk approach (Micro F1 = 0.506). The core problem remains: **the model over-predicts relations by 4.2x** even when shown a sparsely-annotated training example.

The per-essay breakdown reveals consistent over-prediction across all essay sizes:

| Essay | Args | Pairs | Gold Rels | 3-pair Pred | Graph Pred | Graph Ratio |
|-------|------|-------|-----------|-------------|------------|-------------|
| essay004 | 11 | 55 | 6 | 28 | 26 | 4.3x |
| essay005 | 12 | 66 | 6 | 19 | 16 | 2.7x |
| essay006 | 19 | 171 | 13 | 70 | 59 | 4.5x |
| essay021 | 18 | 153 | 12 | 64 | 54 | 4.5x |

### 2.3 Recall is high, precision is the bottleneck

Both pairwise variants achieve high recall (0.86–0.88 for support), meaning they *do find* most true relations. But precision remains low (~0.21) because the model labels too many non-relations as relations. The bulk approach achieves higher precision (0.47) by forcing the model to commit to a selective set.

## 3. Interpretation

The original hypothesis was that relation extraction performance is limited by **combinatorial complexity**—the model struggles to produce all relations simultaneously. The pairwise experiments tested this by decomposing the task into individual pair classifications.

**Results refute this hypothesis.** The bottleneck is **calibration and selectivity**, not combinatorial complexity:

1. **Even with explicit density calibration** (full training graph showing ~9.5% relation density), the model still predicts relations for ~35% of pairs—roughly 4x the true rate.

2. **The bulk format provides an implicit structural constraint** that pairwise classification cannot replicate. When the model must output a complete relation set at once, it naturally produces a sparser, more tree-like structure. In pairwise mode, each pair is judged independently, and the model defaults to "yes" too often.

3. **Graph context is the right direction but insufficient**. The improvement from 3-pair → graph (0.139 → 0.311 Macro F1) shows that calibration matters. But a single training graph is not enough to override the model's tendency to find argumentative connections between most argument pairs in a persuasive essay.

This finding is consistent with the AAEC annotation scheme, where relations follow a specific tree structure (premises → claims → major claims) rather than a dense graph. The bulk prompt implicitly conveys this structural expectation, while pairwise classification strips it away.

