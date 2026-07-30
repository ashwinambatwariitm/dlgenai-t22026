---
license: mit
library_name: transformers
pipeline_tag: multiple-choice
base_model: roberta-base
tags:
  - multiple-choice
  - question-answering
  - roberta
---

# dlgenai-t22026-mcq

`roberta-base` with a multiple-choice head, fully fine-tuned to answer 5-option
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
| base model | `roberta-base` |
| head | `RobertaForMultipleChoice` |
| fine-tuning | full (all parameters, not LoRA) |
| optimizer | AdamW, lr 2e-5 |
| schedule | linear warmup 10% then linear decay |
| effective batch | 16 questions (8 x 2 grad accumulation) |
| max sequence length | 256, dynamic per-batch padding |
| precision | fp16 mixed precision |
| training data | 2,000 labelled questions |

Epoch count was chosen by validation rather than fixed in advance: the model is
evaluated after every epoch and the best-scoring epoch is used for the final
full-data retrain.

## Evaluation

Metric is **MAP@3** - you rank three options and score 1 for a correct first
guess, 1/2 for second, 1/3 for third, 0 otherwise.

Validation uses a **leakage-free split**. The dataset contains many near-
duplicate questions, so a naive random split leaves ~83% of validation questions
with a near-identical twin in training, which inflates scores badly. Instead,
questions are clustered by sentence-embedding similarity (cosine >= 0.95) and
whole clusters are assigned to one side of the split, bringing cross-boundary
near-duplicates to **0%**. Reported numbers therefore reflect genuine
generalisation.

## Limitations

- **English only**, and tuned to one dataset's question style. Expect weaker
  results on other MCQ formats.
- **The training data contains a strong labelling artifact**: across 2,000
  questions there are only 313 distinct correct-answer texts, and none of them
  ever appears as a wrong option. A model trained on it can partly learn "which
  text looks like an answer" instead of genuinely reasoning about the question.
  This is why any validation split drawn from this data is optimistic, and why
  held-out leaderboard performance is meaningfully lower than local validation.
- Always exactly **5 options**; the code above assumes that shape.
