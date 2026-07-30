---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Smart MCQ Solver

A fine-tuned transformer that reads a multiple-choice question and ranks its five
options. Built for the Smart MCQ Solver Challenge.

## The model

A transformer with a **multiple-choice head**, fully fine-tuned on 2,000
labelled questions, and selected as the best of the models compared in the
project notebook. The currently deployed checkpoint is `roberta-base`;
the app loads whatever checkpoint the model repo holds, so swapping in a
different encoder needs no code change.

Validation scores from the notebook's model comparison:

| model | validation MAP@3 |
|---|---|
| **Fine-tuned transformer (deployed here)** | **0.8371** |
| Ensemble (weighted Borda) | 0.8367 |
| From-scratch BiLSTM | 0.8213 |
| RAG + Qwen2.5-0.5B | 0.8208 |
| TF-IDF + Logistic Regression | 0.7733 |
| Sentence embeddings (MiniLM, zero-shot) | 0.4846 |
| Most-frequent-letter baseline | 0.4088 |

Validation uses a **leakage-free split**: near-duplicate questions are clustered
by sentence-embedding similarity and whole clusters are assigned to one side of
the split, so no validation question has a near-identical twin in training.
Those numbers therefore measure genuine generalisation rather than memorisation.

## How it scores options

The model is not a 5-way classifier over the letters A-E. Each option is paired
with the prompt and scored independently by the same shared encoder:

```
(prompt, A) -> score
(prompt, B) -> score        softmax over the 5 scores
(prompt, C) -> score   ->   gives the probability of each option
(prompt, D) -> score
(prompt, E) -> score
```

Two consequences worth knowing:

- **Order invariance.** Shuffling the options does not change which one wins,
  because the letter is only an index applied after scoring.
- **Data efficiency.** Each question yields five `(prompt, option)` training
  signals against one shared scoring function, which matters a great deal with
  only 2,000 labelled questions.

## Metric

The competition scores **MAP@3**: you submit three ranked letters and earn 1 for
a correct first guess, 1/2 for second, 1/3 for third, 0 otherwise. The app shows
the full top-3 alongside the per-option probabilities.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Set `MODEL_ID` in `app.py` to your fine-tuned Hub repo before deploying.
