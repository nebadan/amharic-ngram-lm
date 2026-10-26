# NLP Assignment 1 — Amharic N-gram Language Models

Addis Ababa University · School of Information Technology and Engineering

N-gram language models (n = 1–4) trained on the **GPAC** Amharic corpus: corpus statistics, n-gram
probabilities, stop-word analysis, word clouds, sentence probabilities, text generation,
intrinsic (perplexity) and extrinsic evaluation, and an Amharic next-word suggestion app.

[![Open the reproducibility check in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nebadan/amharic-ngram-lm/blob/main/notebooks/Reproduce_Assignment1.ipynb)

Group: Neibyu Daniel · Yonas Tadesse · Selamawit Siferh — Instructor: Fantahun B. (PhD)

| Deliverable | Where |
|---|---|
| Code + all answers with outputs (Tasks 1–4) | [notebooks/Assignment1_Amharic_Ngram.ipynb](notebooks/Assignment1_Amharic_Ngram.ipynb) |
| **Reproducibility check** (re-runs everything, PASS/FAIL table) | [notebooks/Reproduce_Assignment1.ipynb](notebooks/Reproduce_Assignment1.ipynb) |
| Presentation study guide (LaTeX) | [study_guide/study_guide.pdf](study_guide/study_guide.pdf) |
| Written report | [report/NLP_Assignment1_Report.docx](report/NLP_Assignment1_Report.docx) / [.pdf](report/NLP_Assignment1_Report.pdf) |
| Figures and raw results | [outputs/figures/](outputs/figures/), [outputs/results.json](outputs/results.json) |
| Predictive-text app (Task 4) | Live (static, in-browser): https://huggingface.co/spaces/nebadan/amharic-next-word · exact local Gradio version: [app/app.py](app/app.py) |
| Presentation slides | https://claude.ai/artifact/7GWdMZbbCQVZHLUkSAQSwa (download as .pptx/PDF from the deck) |

## Reproducing the results

`notebooks/Reproduce_Assignment1.ipynb` downloads GPAC from the course Google Drive link,
verifies its SHA-256, re-runs all four tasks with the same random seeds and compares every
headline number with `outputs/results.json`, ending in a PASS/FAIL table.

- **Full mode** (machine with ≥ 24 GB RAM, ~20 min): reproduces every reported number exactly.
- **Quick mode** (chosen automatically with less RAM, e.g. free Colab, ~5 min): runs on a fixed
  25% sample and checks the conclusions (orderings and trends of every task).

Locally:

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute Reproduce_Assignment1.ipynb --output reproduced.ipynb
```

## Layout

```
amharic_lm/          n-gram library
  text.py            normalization, sentence splitting, tokenization, stop-words
  corpus.py          streaming preprocessing, id encoding, train/test split
  ngram.py           hashed n-gram tables, MLE / add-k / Kneser-Ney, prediction, generation
  evaluate.py        next-word prediction, keystroke savings, corrupted-sentence discrimination
notebooks/
  assignment1.py     notebook source (jupytext percent format)
  Assignment1_Amharic_Ngram.ipynb   executed notebook with all outputs
  reproduce.py / Reproduce_Assignment1.ipynb   reproducibility check
app/                 Gradio app (exact Kneser-Ney, runs locally)
static_app/          browser version for the free Hugging Face Space: export_tables.py, space/, deploy.sh
report/              report generator (build_report.js) and the report
study_guide/         in-depth presentation study guide (LaTeX source + PDF)
outputs/             results.json and all figures
data/                GPAC.txt goes here (not committed: 1 GB, from the course link)
```

## Run it locally

Needs ~22 GB of free RAM for the full 77M-token corpus (the full run takes ~16 minutes).

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Download `GPAC.txt` from the [course Google Drive link](https://drive.google.com/file/d/1jEOAORylTBAiNKes_JerD6gFF7cnLFyr/view) into `data/` (or run `.venv/bin/gdown 1jEOAORylTBAiNKes_JerD6gFF7cnLFyr -O data/GPAC.txt`), then execute the notebook:

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Assignment1_Amharic_Ngram.ipynb
```

Run the suggestion app locally (uses `app/model/amharic_4gram_kn.npz`, written by the main notebook; it is not in git because of its size):

```bash
.venv/bin/python app/app.py
```

Rebuild the browser app's suggestion tables (`static_app/space/tables.bin`, not in git) and deploy the static Hugging Face Space:

```bash
.venv/bin/python static_app/export_tables.py
```

```bash
static_app/deploy.sh nebadan/amharic-next-word
```

Rebuild the report (Word + PDF):

```bash
report/make_report.sh
```
