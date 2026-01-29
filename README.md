# GPT-OSS Hallucination Detection for Coding & Typesetting Tasks

## Overview
Purpose: extract activations from transformer layers and train linear probes to detect hallucinated code/markup produced by OpenAI's `gpt-oss-20b` model.

Hallucination in this context refers to the LLM inventing nonexistent names of functions, classes, macros, etc. and referring to them when generating a response to a coding task. This is a commonly observed behavior of LLMs, especially when the coding task involves libraries or languages which have few occurrences in a model's training data, although it is declining in frequency as the capabilities of the frontier models increase.

## Repository Layout
- `linear_probe.ipynb`: original experiment notebook (kept as reference), which was tested on a T4 GPU through Google Colab.

**The rest of the repo aims to copy the experiments in `linear_probe.ipynb` into a full Python package structure. It likely has bugs because it has not been tested on a GPU yet due to compute limitations. It will be tested as soon as possible, and this file will be updated when that happens.**

- `hallucination_tools/`: language-specific helpers and data utilities.
  - `hallucination_tools/python.py`, `typst.py`, `latex.py`: prompt generators, response extractors, and hallucination-checking helpers.
  - `hallucination_tools/data_loader.py`: centralized CSV loading/saving and caching for prompt files and hallucination frequency outputs.
- `interp_tools/linear_probe.py`: activation extraction and linear-probe runner used to train logistic regressions on activations.
- `data/`:
  - `data/prompts/`: prompt CSVs (templates, tasks, and generated prompt lists).
  - `data/hallucination_frequencies/`: per-language cached hallucination frequency CSVs (one file per language, e.g. `latex.csv`).
  - `data/linear_probe_results/`: saved activations, regression weights, and `accuracy.csv` summarizing probe metrics.
- `experiments/`: experiment drivers that orchestrate full runs using the modules above (e.g. `hallucination_detection_latex.py`).
- Demo scripts: `run_probe_dummy_model_demo.py`, `run_typst_dummy_model_demo.py` (use dummy models/tokenizers to test the pipeline without downloading large models).
- `requirements.txt`: project dependencies for running real experiments.

Quick guide
- Check that the file I/O and imports are working:

```bash
python run_typst_dummy_model_demo.py
python run_probe_dummy_model_demo.py
```

- Run a full experiment on LaTeX task prompts:

```bash
python experiments/hallucination_detection_latex.py
```

  Edit the top-level constants in the experiment file to change `N_SAMPLES`, `RECOMPUTE`, `SAVE_ACTIVATIONS`, `SAVE_REGRESSION`, and `LAYERS`.

## Notes
- Prompts live in `data/prompts/` and are consumed by the `compute_hallucination_frequencies` functions in each `hallucination_tools` module. Results are stored in `data/hallucination_frequencies/<language>.csv` and are cached to avoid re-running the same evaluation by default.
- The probe runner writes activations and regression weights under `data/linear_probe_results/` when enabled, and always writes `data/linear_probe_results/accuracy.csv` with probe accuracy metrics.
- Can be easily adapted to other models.
