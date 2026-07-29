# =========================================================
# 0. Install (run once in notebook)
# =========================================================
# !pip install -q transformers datasets accelerate scikit-learn sentencepiece

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split

# =========================================================
# 1. Load data
# =========================================================
DATA_DIR = "/home/studinstru/dlGenAI/dlgenai-t22026/dataset/smart-mcq-solver-challenge"
df = pd.read_csv(f"{DATA_DIR}/train.csv")   # id, prompt, A, B, C, D, E, answer

LABEL2ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

df["label"] = df["answer"].map(LABEL2ID)

def build_text(row):
    # Pack prompt + all 5 options into one input.
    # The model sees all options at once and predicts the class index (0-4).
    return (
        f"Question: {row['prompt']}\n"
        f"A) {row['A']}\n"
        f"B) {row['B']}\n"
        f"C) {row['C']}\n"
        f"D) {row['D']}\n"
        f"E) {row['E']}"
    )

df["text"] = df.apply(build_text, axis=1)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["label"]
)

# =========================================================
# 2. Dataset class
# =========================================================
class MCQDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = int(self.labels[idx])
        return enc

# =========================================================
# 3. Metric
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# =========================================================
# 4. Generic train function (works for both checkpoints)
# =========================================================
def train_model(model_name, output_dir, epochs=3, lr=2e-5, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5,use_safetensors=True
    )

    train_ds = MCQDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds   = MCQDataset(val_df["text"], val_df["label"], tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, tokenizer

# =========================================================
# 5. Train DeBERTa-v3-small
# =========================================================
# Point this at the LOCAL folder you downloaded, e.g. "./models/deberta-v3-small"
# (the folder must contain config.json, pytorch_model.bin/model.safetensors,
#  tokenizer.json / spm.model, etc. - i.e. exactly what snapshot_download or
#  git clone from HF gives you)
REPO_ROOT = "/home/studinstru/dlGenAI/dlgenai-t22026"
DEBERTA_LOCAL_PATH = f"{REPO_ROOT}/models/deberta-v3-small"
ROBERTA_LOCAL_PATH = f"{REPO_ROOT}/models/roberta-base"

# Save fine-tuned outputs with ABSOLUTE paths so they can be found later
# regardless of which directory the notebook's kernel starts in.
DEBERTA_OUTPUT_DIR = f"{REPO_ROOT}/models/deberta-mcq"
ROBERTA_OUTPUT_DIR = f"{REPO_ROOT}/models/roberta-mcq"

deberta_trainer, deberta_tok = train_model(
    model_name=DEBERTA_LOCAL_PATH,
    output_dir=DEBERTA_OUTPUT_DIR,
    epochs=3,
    lr=2e-5,
    batch_size=8,
)

# =========================================================
# 6. Train RoBERTa-base
# =========================================================
roberta_trainer, roberta_tok = train_model(
    model_name=ROBERTA_LOCAL_PATH,
    output_dir=ROBERTA_OUTPUT_DIR,
    epochs=3,
    lr=1e-5,
    batch_size=8,
)

# =========================================================
# 7. Inference / ensemble on any dataframe with same columns
# =========================================================
@torch.no_grad()
def get_logits(model_dir, texts, batch_size=16, max_len=512):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt"
        ).to(device)
        out = model(**enc)
        all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def predict_ensemble(df_in, deberta_dir=DEBERTA_OUTPUT_DIR, roberta_dir=ROBERTA_OUTPUT_DIR):
    texts = df_in.apply(build_text, axis=1).tolist()

    logits_deberta = get_logits(deberta_dir, texts)
    logits_roberta = get_logits(roberta_dir, texts)

    # softmax then average (simple, robust ensembling)
    probs_deberta = torch.softmax(torch.tensor(logits_deberta), dim=-1).numpy()
    probs_roberta = torch.softmax(torch.tensor(logits_roberta), dim=-1).numpy()

    avg_probs = (probs_deberta + probs_roberta) / 2
    pred_ids = np.argmax(avg_probs, axis=-1)
    pred_labels = [ID2LABEL[i] for i in pred_ids]
    return pred_labels, avg_probs

# Example usage on validation set:
preds, probs = predict_ensemble(val_df)
val_acc = (np.array(preds) == val_df["answer"].values).mean()
print("Ensemble validation accuracy:", val_acc)

# Example usage on a fresh test.csv (if you have one, same columns minus 'answer'):
# test_df = pd.read_csv("test.csv")
# test_preds, _ = predict_ensemble(test_df)
# submission = pd.DataFrame({"id": test_df["id"], "answer": test_preds})
# submission.to_csv("submission.csv", index=False)
