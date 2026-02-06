# LLM argumentation framework evaluation
_Exported on 2/6/2026 at 12:07:34 CST from Cursor (2.4.28)_

---

**User**

I want to do experiments to evaluate how well current LLMs can construct argumentation frameworks (AFs) from text, especially bipolar AFs. The folder @PersuasiveEssaysV2 contains the "Argument Annotated Essays (version 2)" dataset, which is one of the best benchmarks for this purpose. I have (with the help of AIs) figured out an experiment plan in @EXPERIMENT_PLAN.md , an implementation plan based on that in @IMPLEMENTATION_PLAN.md , and implemented the code in @src , @requirements.txt , and @run_experiment.py . Please (1) do quick checks on whether the experiment plan is reasonable w.r.t. my goal, and whether the implementation plan is reasonable w.r.t. the experiment plan ; (2) carefully check the code to make sure the implementation is correct and should work as expected. 
DO NOT edit any of the existing files in the codebase, although you are encouraged to propose code edits.

---

**Cursor**

Interpretation: you want a quick reasonableness check of the experiment plan vs your goal, a check that the implementation plan matches the experiment plan, and a careful code review for correctness without editing files.

## Findings (ordered by severity)

- **Critical: gold-argument conditions will drop all relations because IDs are remapped in the prompt but not remapped in parsing.** The gold-argument prompt reassigns gold argument IDs to sequential `a1/a2/...` before sending to the model, so model outputs those IDs, but `parse_relations_only()` only accepts relations that reference the original gold IDs (e.g., `T1`, `T2`). This means every relation is filtered out, so gold-argument results will be near-zero regardless of model quality.  
  ```358:399:src/prompts.py
  def gold_arg_relations(
      essay_text: str,
      gold_baf: BAF,
      examples: List[Dict] | None = None,
  ) -> List[Dict[str, str]]:
      ...
      for i, arg in enumerate(gold_baf.arguments, 1):
          new_id = f"a{i}"
          id_map[arg.id] = new_id
          args.append({"id": new_id, "text": arg.text})
      ...
  ```
  ```249:262:src/output_parser.py
  def parse_relations_only(raw_output: str, gold_baf: BAF) -> BAF:
      ...
      gold_ids = {a.id for a in gold_baf.arguments}
      relations = _parse_relations(data.get("relations", []), gold_ids)
      return BAF(arguments=list(gold_baf.arguments), relations=relations)
  ```
  **Proposed fix:** either (a) keep original gold IDs in the prompt (no remap), or (b) return the `id_map` from `gold_arg_relations()` and map model relation IDs back to gold IDs before `_parse_relations`.

- **High: parse success is always `True` for gold-argument runs, even when JSON is invalid.** This makes `parse_success_rate` meaningless for gold conditions and mismatches your plan’s definition.  
  ```169:183:run_experiment.py
      elif is_gold:
          ...
          return {
              ...
              "parse_success": True,  # gold args always present
          }
  ```
  **Proposed fix:** set `parse_success` based on whether JSON could be parsed and contained a `relations` list, even if empty.

- **Medium: few-shot examples may violate the “verbatim substring” rule when BRAT annotations are discontinuous.** In `parse_brat_ann`, discontinuous spans are collapsed to outermost boundaries and `text` is taken directly from the `.ann` line. This `text` may not be a contiguous substring of the essay and will be used in few-shot examples, potentially teaching the model to output non-verbatim spans.  
  ```40:56:src/data_loader.py
          if line.startswith("T"):
              ...
              # Handle discontinuous spans: take outermost boundary
              start = int(type_and_offsets[1])
              end = int(type_and_offsets[-1])
              text = parts[2]
              arguments.append(
                  Argument(id=entity_id, text=text, start=start, end=end)
              )
  ```
  **Proposed fix:** for discontinuous spans, either (a) reconstruct `text` from `essay_text[start:end]` (contiguous approximation) and flag it, or (b) skip discontinuous spans for few-shot examples to keep them strictly verbatim.

## Quick checks: experiment plan vs goal

Overall the plan is well aligned with your stated goal (prompting-based BAF extraction on AAEC/Persuasive Essays). The main design choices look reasonable:

- Good: end-to-end vs pipeline vs gold-argument diagnostics directly probe span vs relation bottlenecks.
- Good: macro relation F1 as primary metric properly handles extreme class imbalance.
- Good: fixed few-shot examples for reproducibility.
- Good: temperature 0 for determinism and easier error analysis.

Two mild gaps to consider:
- The plan assumes verbatim span extraction; discontinuous spans in AAEC can complicate this (see medium issue above).
- The plan’s “parse success” definition is tighter than the current implementation for gold conditions.

## Quick checks: implementation plan vs experiment plan

The implementation plan matches the experiment plan closely:

- All method variants in the plan are implemented in `src/prompts.py` and `run_experiment.py`.
- Evaluation matches the plan: IoU ≥ 0.5, Hungarian matching, macro/micro relation F1, bootstrap CIs.
- Few-shot example selection and persistence to `results/meta` match the plan.

The main mismatch is the gold-argument ID remapping bug (critical). The parse success definition mismatch for gold conditions is also a plan-implementation divergence.

## Suggested edits (no changes made)

1) **Gold-argument ID mapping**
   - Easiest: keep original gold IDs in `gold_arg_relations()` (no `a1/a2` remap).
   - Alternative: return `id_map` and map model output IDs back to gold IDs before parsing relations.

2) **Gold-argument parse_success**
   - Mark success only if `_extract_json` returns a dict and `relations` parses cleanly.

3) **Discontinuous spans in few-shot examples**
   - When building examples, replace `arg.text` with `essay_text[arg.start:arg.end]` to ensure verbatim substrings (or skip discontinuous spans if you can detect them).

If you want, I can draft concrete code changes as patch suggestions without applying them.

