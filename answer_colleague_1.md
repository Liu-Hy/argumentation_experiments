# LLM argumentation framework evaluation
_Exported on 2/6/2026 at 12:07:06 CST from Cursor (2.4.28)_

---

**User**

I want to do experiments to evaluate how well current LLMs can construct argumentation frameworks (AFs) from text, especially bipolar AFs. The folder @PersuasiveEssaysV2 contains the "Argument Annotated Essays (version 2)" dataset, which is one of the best benchmarks for this purpose. I have (with the help of AIs) figured out an experiment plan in @EXPERIMENT_PLAN.md , an implementation plan based on that in @IMPLEMENTATION_PLAN.md , and implemented the code in @src , @requirements.txt , and @run_experiment.py . Please (1) do quick checks on whether the experiment plan is reasonable w.r.t. my goal, and whether the implementation plan is reasonable w.r.t. the experiment plan ; (2) carefully check the code to make sure the implementation is correct and should work as expected. 
DO NOT edit any of the existing files in the codebase, although you are encouraged to propose code edits.

---

**Cursor**

Got it. I’ll (a) read `EXPERIMENT_PLAN.md` and sanity-check it against your stated goal (AF/BAF-from-text evaluation), (b) read `IMPLEMENTATION_PLAN.md` and verify it matches the experiment plan, then (c) do a careful pass through `run_experiment.py` + everything in `src/` (data loading → prompting → LLM calls → parsing → AF/BAF construction → evaluation → outputs), flagging any correctness or “will break at runtime” issues and proposing concrete code edits (without changing files).

Next I’m going to sanity-check the dataset file formats against what `data_loader.py` expects, and then I’ll trace the gold/pipeline prompting → parsing → evaluation paths to catch any ID/offset mismatches (I already see one likely serious ID mismatch in the gold-argument setting).

Now I’ll inspect a couple concrete `.ann`/`.txt` pairs to verify the BRAT parsing assumptions (argument lines, relation lines, discontinuous spans). This will also tell us whether the “verbatim substring” constraint holds for the gold examples you embed in prompts.

Next I’ll walk through the remaining `src/` modules (and any `__init__.py` / helpers) to ensure there aren’t hidden inconsistencies—especially around ID normalization, span text handling, and what exactly gets counted in metrics.

### (1) Quick checks: experiment plan vs your goal

Your `EXPERIMENT_PLAN.md` is **well-aligned** with “evaluate how well current LLMs can construct (bi)polar AFs from text”:

- **Task formulation is appropriate**: converting AAEC’s component taxonomy into **nodes + {support, attack} edges** is a clean way to focus on BAF structure rather than corpus-specific labels.
- **Method matrix is sensible**: E2E vs 2-step pipeline, zero-shot vs few-shot, plus a **gold-argument diagnostic** is exactly the right set of comparisons to localize bottlenecks.
- **Metrics are mostly on-point**: IoU-based span matching + Hungarian alignment is a reasonable relaxed span metric; relation scoring conditioned on matched endpoints is standard; **macro relation F1** as the primary metric is important given the extreme class imbalance (attacks are rare).

Main practical risks (not “wrong”, but likely to affect results/interpretation):
- **Output size / truncation**: full-graph output for long essays can exceed completion budgets (esp. few-shot with full essay examples). This will show up as parse failures or systematically missing edges/arguments.
- **Verbatim-span requirement is hard for LLMs**: you’re evaluating both “argument mining” and “exact quoting”. That’s fine, but you should expect the parser/metric design to strongly shape outcomes.

### (1) Quick checks: implementation plan vs experiment plan

`IMPLEMENTATION_PLAN.md` broadly matches `EXPERIMENT_PLAN.md` (modules, data flow, metrics, bootstrap CIs, OpenRouter usage). However there is **one major mismatch**: the **gold-argument diagnostic condition as implemented cannot work correctly** due to argument-ID renaming (details below). That means H6 (“bottleneck is relation extraction vs span identification”) would be invalid with current code.

---

### (2) Careful code check (correctness + “will it work?”)

#### What looks correct / matches the plan

- **Dataset loading + conversion**: `load_persuasive_essays()` correctly reads `train-test-split.csv` (semicolon-delimited) and parses BRAT `.ann`/`.txt`. I also smoke-tested loading: it yields **402 total / 322 train / 80 test** and global counts **6089 args / 3613 supports / 219 attacks**, matching your plan.
- **Experiment loop + resume**: `run_experiment.py` writes per-essay raw outputs and then aggregates metrics; `--resume` reuses cached `pred_baf` objects and skips API calls.
- **Evaluation core**: span matching via Hungarian and relation scoring “TP iff endpoints match + typed edge exists” matches your plan’s definitions.

#### High-severity bug: gold-argument setting ID mismatch (breaks diagnostic experiments)

In `src/prompts.py`, `gold_arg_relations()` **renumbers** gold arguments to IDs `a1`, `a2`, … in the prompt:

```358:399:/home/techt/Desktop/argumentation_experiments/src/prompts.py
def gold_arg_relations(
    essay_text: str,
    gold_baf: BAF,
    examples: List[Dict] | None = None,
) -> List[Dict[str, str]]:
    ...
    # Format gold arguments with sequential IDs
    id_map = {}
    args = []
    for i, arg in enumerate(gold_baf.arguments, 1):
        new_id = f"a{i}"
        id_map[arg.id] = new_id
        args.append({"id": new_id, "text": arg.text})
```

But `parse_relations_only()` only accepts relation endpoints that are in the **original gold IDs** (AAEC uses `T1`, `T2`, …), so any model output that follows the prompt and uses `a1`/`a2` will be discarded:

```249:262:/home/techt/Desktop/argumentation_experiments/src/output_parser.py
def parse_relations_only(raw_output: str, gold_baf: BAF) -> BAF:
    ...
    gold_ids = {a.id for a in gold_baf.arguments}
    relations = _parse_relations(data.get("relations", []), gold_ids)
    return BAF(arguments=list(gold_baf.arguments), relations=relations)
```

And `run_experiment.py` routes gold methods through this path:

```169:183:/home/techt/Desktop/argumentation_experiments/run_experiment.py
elif is_gold:
    msgs = build_messages(method, essay_text, examples, gold_baf=gold_baf)
    resp: LLMResponse = client.generate(msgs, model_key)
    pred_baf = parse_relations_only(resp.content, gold_baf)
```

**Effect**: `gold_zs` / `gold_fs` will produce **near-zero relations**, making gold-argument scores meaningless.

**Proposed fix (no file edits made here, just recommendation)**:
- Either **don’t renumber** in the gold-argument prompt (use `T1`, `T2`, … as IDs), **or**
- Keep renumbering in the prompt, but in `parse_relations_only()` accept `a1..aN` and **map them back** to the underlying `T*` IDs before returning the `BAF` (mapping is deterministic by order).

#### Methodology/implementation gotchas that can bias metrics upward

These are not syntax/runtime bugs, but they affect whether the evaluation is “as expected”.

1) **Unresolvable predicted arguments are dropped (not counted as FP).**  
If the model outputs an argument text that can’t be located/fuzzy-located, `_parse_arguments()` simply skips it:

```112:154:/home/techt/Desktop/argumentation_experiments/src/output_parser.py
resolved = _fuzzy_find(text, essay_text)
if resolved is not None:
    ...
else:
    logger.debug(f"Could not locate argument text in essay: {text[:60]}...")
return arguments
```

That means these “bad predictions” are **not counted** in `len(pred_baf.arguments)`, so argument precision can be inflated (and relations involving those args never even enter evaluation).

**Suggestion**: track and count “dropped/unlocatable” predicted arguments (and invalid relations) as FP in metrics, or treat them as parse failures in a way that penalizes predictions rather than ignoring them.

2) **Invalid/unknown relations are dropped (not counted as FP).**  
`_parse_relations()` filters out unknown types and relations that reference unknown argument IDs; those predictions disappear rather than being penalized.

**Suggestion**: keep a counter of filtered relations and include it in results; optionally count them as FP for a stricter evaluation.

#### Robustness issues (likely to cause avoidable parse failures)

- **CoT E2E prompt increases chance of non-JSON text**: `fs_cot_e2e` encourages reasoning; `_extract_json()` can latch onto the wrong `{...}` block if the model uses braces in reasoning. Strongly consider changing prompts to require the final JSON inside a single ```json code fence (and nothing else after it), and/or making `_extract_json()` search for a JSON object containing `"arguments"`/`"relations"` rather than “first braces”.
- **Fuzzy matcher’s candidate start positions are whitespace-based** (suboptimal and can misalign matches):

```183:202:/home/techt/Desktop/argumentation_experiments/src/output_parser.py
words_starts = [0] + [m.start() for m in re.finditer(r"\s+", text)]
for ws in words_starts:
    ...
    candidate = text[ws : ws + wlen]
```

Using `m.start()` puts the window start on whitespace. Using `m.end()` (start of next token) is usually better.

#### Minor correctness/engineering notes

- `parse_brat_ann()` uses the `.ann` “text” field directly rather than slicing `essay_text[start:end]`. It appears verbatim in the samples I checked, but slicing from offsets would **guarantee** examples are true substrings (important because your prompts insist on verbatim copying).
- Gold-method `parse_success` is hardcoded `True` in `run_experiment.py`, so “parse success rate” is not meaningful for gold runs.

---

### Summary: is it reasonable + should it work?

- **Experiment plan**: yes, it’s a strong and publishable design for prompted BAF extraction on AAEC, with good diagnostic comparisons and appropriate metrics for the attack imbalance.
- **Implementation plan**: mostly consistent, but the **gold-argument diagnostic is currently broken** in code due to ID renumbering.
- **Codebase**: runs, loads data, and computes metrics as designed; but you should fix the gold-ID bug and (ideally) address the “drop invalid predictions” behavior if you want metrics that reflect *all* model outputs rather than only the parseable subset.

