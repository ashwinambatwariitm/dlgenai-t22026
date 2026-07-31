---
license: mit
library_name: transformers
pipeline_tag: multiple-choice
base_model: bert-base-uncased
tags:
  - multiple-choice
  - question-answering
  - bert
---

# dlgenai-t22026-mcq

`bert-base-uncased` with a multiple-choice head, fully fine-tuned to answer 5-option
multiple-choice questions (Smart MCQ Solver Challenge).

## How it works

This is **not** a 5-way classifier over the letters A-E. Each option is paired
with the prompt and scored independently by the same shared encoder:

```
(prompt, option A) -> score
(prompt, option B) -> score      softmax over the 5 scores
(prompt, option C) -> score  ->  gives a probability per option
(prompt, option D) -> score
(prompt, option E) -> score
```

Two consequences: the prediction is **invariant to the order** the options are
listed in (the letter is only an index applied after scoring), and each question
yields five `(prompt, option)` training signals against one shared scoring
function - which matters with only 2,000 labelled questions.

## Usage

```python
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

model_id = "ashwinambatwariitm/dlgenai-t22026-mcq"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMultipleChoice.from_pretrained(model_id).eval()

prompt = "What is the capital of France?"
options = ["Berlin", "Madrid", "Paris", "Rome", "Lisbon"]

enc = tok([prompt] * 5, options, truncation=True, max_length=256,
          padding=True, return_tensors="pt")
batch = {k: v.unsqueeze(0) for k, v in enc.items()}       # (1, 5, seq_len)

with torch.no_grad():
    probs = model(**batch).logits[0].softmax(-1)

best = int(probs.argmax())
print("ABCDE"[best], options[best])
```

## Training

| setting | value |
|---|---|
| base model | `bert-base-uncased` |
| head | `BertForMultipleChoice` |
| fine-tuning | full (all parameters, not LoRA) |
| optimizer | AdamW, lr 2e-5 |
| schedule | linear warmup 10% then linear decay |
| effective batch | 16 questions (8 x 2 grad accumulation) |
| max sequence length | 256, dynamic per-batch padding |
| precision | fp16 mixed precision |
| epochs | 10 trained, epoch 6 selected by validation |
| training data | 2,000 labelled questions |

Epoch count was chosen by validation rather than fixed in advance: the model is
evaluated after every epoch and the best-scoring epoch (6 of 10) is used for the
final full-data retrain.

## Evaluation

Metric is **MAP@3** - you rank three options and score 1 for a correct first
guess, 1/2 for second, 1/3 for third, 0 otherwise.

Validation uses a **leakage-free split**. The dataset contains many near-
duplicate questions, so a naive random split leaves 79.5% of validation questions
with a near-identical twin in training, which inflates scores badly. Instead,
questions are clustered by sentence-embedding similarity (cosine >= 0.95) and
whole clusters are assigned to one side of the split, bringing cross-boundary
near-duplicates to **0%**. Reported numbers therefore reflect genuine
generalisation.

### Results

Leakage-free validation split (400 held-out questions):

| model | MAP@3 |
|---|---|
| Ensemble (weighted Borda over all models) | 0.8858 |
| **`bert-base-uncased` fine-tuned (this model)** | **0.8729** |
| From-scratch BiLSTM | 0.8633 |
| From-scratch mini transformer | 0.8471 |
| From-scratch text-CNN | 0.8462 |
| RAG (retrieval + Qwen2.5-0.5B-Instruct) | 0.8208 |
| **Answer-text artifact (reads no question)** | **0.7954** |
| TF-IDF + Logistic Regression | 0.7733 |
| TF-IDF + Linear SVM | 0.7596 |
| Sentence embeddings (MiniLM, zero-shot) | 0.4846 |
| Most-frequent-letter baseline | 0.4088 |
| Constant "A B C" baseline | 0.3937 |

Validation peaked at epoch 6 of 10, so the final full-data model was retrained
for exactly 6 epochs. Top-1 accuracy on the validation split is 0.7925.

**Read that table against 0.7954, not against 0.3937.** The row in bold is not a
model: it ranks the five options solely by whether an option's text was ever a
correct answer during training, never reading the prompt. It works because this
dataset draws correct answers and distractors from pools that never overlap (313
answer texts, 1,271 distractor texts, zero in both roles). So the portion of this
model's score attributable to actually reading the question is about **0.078**
(0.8729 minus 0.7954), rather than the 0.479 that measuring against the constant
baseline would suggest. Note that TF-IDF falls *below* the artifact line entirely.

Worth noting honestly: the from-scratch BiLSTM (0.8633) lands within 0.010 of
this fine-tuned model. Its own validation score swings between 0.8633 and 0.8942
across the last two epochs, and two runs of the identical pipeline moved this
model between 0.8650 and 0.8729, so that gap is inside the run-to-run noise and
should not be read as the two being equivalent - only as not separable at this
sample size (400 questions).

Note the gap between a pretrained encoder used **as-is** (MiniLM, 0.4846 - barely
above the 0.4088 baseline) and the same class of model **fine-tuned** on this task
(0.8729). Pretrained knowledge alone does not solve this; adapting it does.

The three from-scratch encoders isolate architecture from pretraining. The mini
transformer (0.8471) uses the same self-attention mechanism as this model but
with every weight randomly initialised, so the distance from it to 0.8729 is
roughly what pretraining bought on this task.

### Held-out leaderboard score

The full pipeline (this model inside a weighted-Borda ensemble, plus a
retrieval layer) scores **MAP@3 0.75664** on the competition's hidden test set
(Kaggle public leaderboard), against **0.8858** on the leakage-free validation
split. The submission it produced differed from this model's raw predictions on
only 0 of 500 questions on the submitted run, so the retrieval layer set
rank-1 for 466 of 500 and the trained model supplied the rest.

That ~0.13 gap is not a bug - it is the most interesting property of this
dataset, and the two numbers measure genuinely different things.

The cause is that **half the competition test set is literally present in the
training data.** Comparing the 500 test questions against `train.csv` by exact
string match:

| relationship to train.csv | share of test | recoverable by memorisation? |
|---|---|---|
| **identical question** (same prompt *and* same 5 options) | **51.4%** (257/500) | yes |
| same prompt, different options | 2.2% (11/500) | partly |
| different prompt, same option set | 38.8% (194/500) | no - a different question |
| no match at all | 7.6% (38/500) | no |

All 257 identical twins carry a single consistent answer, so that half of the
test set is answerable by lookup alone. The 38.8% that merely reuse an option set
are genuinely different questions - the dataset recycles distractor pools across
many prompts, which makes naive embedding-similarity matching look far more
promising than it is.

This composition explains the validation/leaderboard gap. Our validation split
removes near-duplicates by design (0%), so it measures reasoning on unseen
questions. The test set is ~51% recall and ~49% reasoning, so a leaderboard score
blends the two. It also explains why the model with the best validation score is
**not** the best leaderboard model: a simple retrieval baseline (nearest training
question, copy its answer text) scores 0.75436 despite only 0.8208 on validation.

If one assumes the memorisable half is answered essentially perfectly, the
implied MAP@3 on the remaining ~49% is roughly **0.50**, against a
random-guessing baseline of 0.3937 - i.e. only modestly above chance on
genuinely novel questions. That figure is an inference from the leaderboard
score and the composition above, not a directly measured value, since the test
labels are not public.

Treat 0.8858 as the optimistic estimate of generalisation, 0.75664 as the
observed score on the competition's test distribution, and expect noticeably
weaker performance on questions unlike this dataset.

The decoupling is worth stating plainly: across the models in the table,
validation MAP@3 ranges from 0.82 to 0.89, yet every submission lands within
about 0.01 of 0.75 on the leaderboard. Because retrieval fixes rank-1 for 93% of
test questions, the choice of trained model has little leverage on the score. Two
runs of the identical pipeline make this concrete: validation rose 0.0116 while
the leaderboard fell 0.0027.

## Limitations

- **English only**, and tuned to one dataset's question style. Expect weaker
  results on other MCQ formats.
- **The training data contains a strong labelling artifact**: across 2,000
  questions there are only 313 distinct correct-answer texts, and none of them
  ever appears as a wrong option. A model trained on it can partly learn "which
  text looks like an answer" instead of genuinely reasoning about the question.
  This is why any validation split drawn from this data is optimistic.
- **Roughly half of this model's leaderboard score is recall, not reasoning.**
  51.4% of the competition test set is an exact duplicate of a training question
  (see Evaluation). Do not read 0.75664 as an estimate of reasoning ability on
  new questions - the evidence suggests that is closer to 0.50 MAP@3, against a
  0.3937 random baseline.
- Always exactly **5 options**; the code above assumes that shape.
