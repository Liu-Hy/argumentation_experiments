# LLM Bipolar Argumentation Framework Extraction

This project evaluates how well current LLMs can extract bipolar argumentation frameworks (BAFs) from natural language text using prompting alone (no fine-tuning). Given a persuasive essay, the task is to identify argument spans and classify the support/attack relations between them.

We use the [Argument Annotated Essays v2](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) dataset (402 essays, ~6k arguments, ~3.8k relations) and test 10 LLMs via OpenRouter across five prompting strategies that vary along two axes: task decomposition (end-to-end vs. two-step pipeline) and prompting style (zero-shot vs. few-shot), plus chain-of-thought reasoning. A gold-argument diagnostic condition isolates relation extraction performance from span identification errors.

Evaluation uses character-level IoU span matching with Hungarian alignment, and reports macro-averaged F1 separately for support and attack relations, with bootstrap confidence intervals.

## Usage

```bash
pip install -r requirements.txt
python run_experiment.py --method e2e_zs --model <model_key>
```

Set `OPENROUTER_API_KEY` in a `.env` file or as an environment variable. See `EXPERIMENT_PLAN.md` and `IMPLEMENTATION_PLAN.md` for full details on the experimental design.
