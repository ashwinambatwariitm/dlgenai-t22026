"""Smart MCQ Solver - Streamlit demo.

Loads the fine-tuned multiple-choice model from the Hugging Face Hub and ranks
the five options of a question.

The model is NOT a 5-way classifier over the letters A-E. Each option is paired
with the prompt and scored on its own by the same shared encoder, then a softmax
over the five scores gives a probability per option. That makes the prediction
independent of the order the options are listed in.

Run locally:  streamlit run streamlit_app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

MODEL_ID = "ashwinambatwariitm/dlgenai-t22026-mcq"
LETTERS = ["A", "B", "C", "D", "E"]
MAX_LEN = 256

st.set_page_config(page_title="Smart MCQ Solver", page_icon="brain", layout="centered")


@st.cache_resource(show_spinner="Loading the fine-tuned model (first run only)...")
def load_model():
    """Cached so the 500 MB model loads once per container, not once per click."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
    mdl.eval()
    return tok, mdl


@torch.no_grad()
def solve(tok, mdl, prompt, options):
    """Score the five options; returns (ranked_indices, probabilities)."""
    filled = [i for i, o in enumerate(options) if o and o.strip()]
    texts = [o if (o and o.strip()) else "" for o in options]

    enc = tok([prompt] * 5, texts, truncation=True, max_length=MAX_LEN,
              padding=True, return_tensors="pt")
    batch = {k: v.unsqueeze(0) for k, v in enc.items()}     # (1, 5, seq_len)

    logits = mdl(**batch).logits[0]                          # (5,)
    mask = torch.full_like(logits, float("-inf"))
    mask[filled] = 0.0                                       # blank options cannot win
    probs = torch.softmax(logits + mask, dim=-1)

    ranked = sorted(filled, key=lambda i: float(probs[i]), reverse=True)
    return ranked, probs


st.title("Smart MCQ Solver")
st.markdown(
    "A fine-tuned transformer that reads a multiple-choice question and ranks "
    "the five options. Each option is scored **independently** against the "
    "prompt by a shared encoder, so the answer does not depend on the order "
    "the options are listed in."
)

EXAMPLE_PROMPT = (
    "Pick the best possible answer: What is Martin Heidegger's view on the "
    "relationship between time and human existence? among the listed options."
)
EXAMPLE_OPTIONS = [
    "Martin Heidegger believes that humans exist within a time continuum that is "
    "infinite and does not have a defined beginning or end.",
    "Martin Heidegger believes that humans do not exist inside time, but that they "
    "are time. The relationship to the past is a present awareness of having been.",
    "Martin Heidegger does not believe in the existence of time or that it has any "
    "effect on human consciousness.",
    "Martin Heidegger believes that the relationship between time and human "
    "existence is cyclical and the future is predetermined.",
    "Martin Heidegger believes that time is an illusion, and the past, present and "
    "future are all happening simultaneously.",
]

if st.button("Load an example question"):
    st.session_state["prompt"] = EXAMPLE_PROMPT
    for letter, text in zip(LETTERS, EXAMPLE_OPTIONS):
        st.session_state[f"opt_{letter}"] = text

prompt = st.text_area("Question", key="prompt", height=90,
                      placeholder="Type the question here...")
options = [
    st.text_area(f"Option {letter}", key=f"opt_{letter}", height=68)
    for letter in LETTERS
]

if st.button("Solve", type="primary"):
    if not prompt or not prompt.strip():
        st.warning("Please enter a question.")
    elif sum(1 for o in options if o and o.strip()) < 2:
        st.warning("Please fill in at least two options.")
    else:
        tok, mdl = load_model()
        ranked, probs = solve(tok, mdl, prompt, options)

        best = ranked[0]
        st.success(f"Predicted answer: **{LETTERS[best]}**")
        st.write(options[best])

        top3 = " ".join(LETTERS[i] for i in ranked[:3])
        st.caption(f"Top-3 in the competition's MAP@3 format: `{top3}`")

        st.subheader("Confidence per option")
        for i in ranked:
            label = options[i].strip()
            st.write(f"**{LETTERS[i]}** - {label[:90]}{'...' if len(label) > 90 else ''}")
            st.progress(float(probs[i]), text=f"{float(probs[i]):.1%}")

st.divider()
st.caption(
    f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) - roberta-base with a "
    "multiple-choice head, fully fine-tuned on 2,000 questions."
)
