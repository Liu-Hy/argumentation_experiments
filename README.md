# LLM Bipolar Argumentation Framework Extraction

This project evaluates how well current LLMs can extract bipolar argumentation frameworks (BAFs) from natural language text using prompting alone (no fine-tuning). Given a persuasive essay, the task is to identify argument spans and classify the support/attack relations between them.

We use the [Argument Annotated Essays v2](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) dataset (402 essays, ~6k arguments, ~3.8k relations) and test 10 LLMs via OpenRouter across five prompting strategies that vary along two axes: task decomposition (end-to-end vs. two-step pipeline) and prompting style (zero-shot vs. few-shot), plus chain-of-thought reasoning. A gold-argument diagnostic condition isolates relation extraction performance from span identification errors.

Evaluation uses character-level IoU span matching with Hungarian alignment, and reports macro-averaged F1 separately for support and attack relations.

## Project Structure

```
data/                   Input datasets (PersuasiveEssaysV2, CAIL2023, japanese-tort-case)
results/                Experiment results organized by dataset
src/                    Core Python modules (BAF, data loader, evaluation, prompts, etc.)
scripts/                Shell scripts for running experiment pipelines
docs/                   Experiment plans, reviews, and reports
```

## Usage

```bash
pip install -r requirements.txt
python run_experiment.py --model <model_key> --dataset ./data/PersuasiveEssaysV2
```

Set `OPENROUTER_API_KEY` in a `.env` file or as an environment variable. See `docs/EXPERIMENT_PLAN.md` and `docs/IMPLEMENTATION_PLAN.md` for full details on the experimental design.
