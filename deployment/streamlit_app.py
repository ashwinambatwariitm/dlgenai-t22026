"""Smart MCQ Solver - Streamlit demo.

Loads the fine-tuned multiple-choice model from the Hugging Face Hub and ranks
the five options of a question.

The model is NOT a 5-way classifier over the letters A-E. Each option is paired
with the prompt and scored on its own by the same shared encoder, then a softmax
over the five scores gives a probability per option. That makes the prediction
independent of the order the options are listed in.

Run locally:  streamlit run streamlit_app.py
"""

import random

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

# Five worked examples taken VERBATIM from train.csv - prompt and all five
# options exactly as the model saw them in training, one per correct answer
# letter (A to E). The exact wording matters: abbreviating an option costs the
# model the answer-text signal it partly relies on, and confidence drops even
# when the ranking stays right. Covering all five letters also lets a reviewer
# check the model is not simply favouring one position.
EXAMPLES = [
    {
        "id": 5,
        "answer": "A",
        "prompt": "Identify the correct statement: What is the concept of simultaneity "
            "in Einstein's book, Relativity? carefully.",
        "options": [
            "Simultaneity is relative, meaning that two events that appear "
            "simultaneous to an observer in a particular inertial reference frame "
            "need not be judged as simultaneous by a second observer in a "
            "different inertial frame of reference.",
            "Simultaneity is relative, meaning that two events that appear "
            "simultaneous to an observer in a particular inertial reference frame "
            "will always be judged as simultaneous by a second observer in a "
            "different inertial frame of reference.",
            "Simultaneity is absolute, meaning that two events that appear "
            "simultaneous to an observer in a particular inertial reference frame "
            "will always be judged as simultaneous by a second observer in a "
            "different inertial frame of reference.",
            "Simultaneity is a concept that applies only to Newtonian theories "
            "and not to relativistic theories.",
            "Simultaneity is a concept that applies only to relativistic theories "
            "and not to Newtonian theories.",
        ],
    },
    {
        "id": 12,
        "answer": "B",
        "prompt": "Choose the correct answer: What is the throttling process, and why "
            "is it essential? from the following choices.",
        "options": [
            "The throttling process is a steady flow of a fluid through a flow "
            "resistance, such as a valve or porous plug, and is responsible for "
            "the pressure raise in domestic refrigerators. This process is key "
            "because it is at the heart of the refrigeration cycle.",
            "The throttling process is a steady adiabatic flow of a fluid through "
            "a flow resistance, such as a valve or porous plug, and is "
            "responsible for the temperature drop in domestic refrigerators. This "
            "process is key because it is at the heart of the refrigeration "
            "cycle.",
            "The throttling process is a steady adiabatic flow of a fluid through "
            "a flow resistance, such as a valve or porous plug, and is "
            "responsible for the pressure drop in domestic refrigerators. This "
            "process is key because it is at the heart of the refrigeration "
            "cycle.",
            "The throttling process is a steady flow of a fluid through a flow "
            "resistance, such as a valve or porous plug, and is responsible for "
            "the temperature raise in domestic refrigerators. This process is key "
            "because it is at the heart of the refrigeration cycle.",
            "The throttling process is a steady adiabatic flow of a fluid through "
            "a flow resistance, such as a valve or porous plug, and is "
            "responsible for the temperature drop in domestic refrigerators. This "
            "process is not significant because it is not used in the "
            "refrigeration cycle.",
        ],
    },
    {
        "id": 11,
        "answer": "C",
        "prompt": "Select the most accurate option: What is the Peierls bracket in "
            "canonical quantization? based on the given context.",
        "options": [
            "The Peierls bracket is a mathematical symbol used to represent the "
            "Poisson algebra in the canonical quantization approach.",
            "The Peierls bracket is a mathematical tool used to generate the "
            "Hamiltonian in the canonical quantization technique.",
            "The Peierls bracket is a Poisson bracket derived from the action in "
            "the canonical quantization procedure that converts the quotient "
            "algebra into a Poisson algebra.",
            "The Peierls bracket is a mathematical symbol used to represent the "
            "quotient algebra in the canonical quantization approach.",
            "The Peierls bracket is a mathematical tool used to generate the "
            "Euler-Lagrange equations in the canonical quantization technique.",
        ],
    },
    {
        "id": 28,
        "answer": "D",
        "prompt": "Identify the correct statement: What is a uniform tiling in the "
            "hyperbolic plane? based on the given context.",
        "options": [
            "A uniform tiling in the hyperbolic plane is a tessellation of the "
            "hyperbolic plane with irregular polygons as faces. These are not "
            "vertex-transitive and isogonal.",
            "A uniform tiling in the hyperbolic plane is a tessellation of the "
            "hyperbolic plane with regular polygons as faces. These are not "
            "vertex-transitive and isogonal.",
            "A uniform tiling in the hyperbolic plane is a tessellation of the "
            "hyperbolic plane with irregular polygons as faces. These are "
            "vertex-transitive and isogonal.",
            "A uniform tiling in the hyperbolic plane is an edge-to-edge filling "
            "of the hyperbolic plane, with regular polygons as faces. These are "
            "vertex-transitive and isogonal.",
            "A uniform tiling in the hyperbolic plane is an edge-to-edge filling "
            "of the hyperbolic plane, with irregular polygons as faces. These are "
            "vertex-transitive and isogonal.",
        ],
    },
    {
        "id": 7,
        "answer": "E",
        "prompt": "Which of the following is correct? What is a \"coffee ring\" in "
            "physics? among the listed options.",
        "options": [
            "A type of coffee that is made by boiling coffee grounds in water.",
            "A pattern left by a particle-laden liquid after it is spilled, named "
            "for the characteristic ring-like deposit along the perimeter of a "
            "spill of coffee or red wine.",
            "A type of coffee that is made by mixing instant coffee with hot "
            "water.",
            "A type of coffee that is made by pouring hot water over coffee "
            "grounds in a filter.",
            "A pattern left by a particle-laden liquid after it evaporates, named "
            "for the characteristic ring-like deposit along the perimeter of a "
            "spill of coffee or red wine.",
        ],
    },
]


# Five questions from test.csv that appear NOWHERE in train.csv - not the same
# prompt, not the same option set. Only 38 of the 500 test questions qualify
# (7.6%); the rest reuse training material. The deployed model was trained on
# all 2,000 training rows, so these are the only genuinely unseen questions
# available. Their labels are not public, so there is nothing to check against -
# the point is the confidence spread, which is what MAP@3's top-3 exists for.
UNSEEN_EXAMPLES = [
    {
        "id": 6,
        "prompt": "Choose the correct answer: What is the Evans balance? from the "
            "following choices.",
        "options": [
            "The Evans balance is a mechanism used to measure the change in "
            "weight of a sample when an electromagnet is turned on, which is "
            "proportional to the susceptibility.",
            "The Evans balance is a framework used to measure the dependence of "
            "the NMR frequency of a liquid sample on its shape or orientation to "
            "determine its susceptibility.",
            "The Evans balance is a mechanism used to measure the magnetic field "
            "distortion around a sample immersed in water inside an MR scanner to "
            "determine its susceptibility.",
            "The Evans balance is a structure used to measure the susceptibility "
            "of a sample by measuring the force change on a strong compact magnet "
            "upon insertion of the sample.",
            "The Evans balance is a structure used to measure the magnetic "
            "susceptibility of most crystals, which is not a scalar quantity.",
        ],
    },
    {
        "id": 75,
        "prompt": "Determine the correct option: What is the reason for heating metals "
            "to a temperature just above the upper critical temperature? from the "
            "following choices.",
        "options": [
            "To prevent the grains of solution from growing too large, which "
            "minimizes mechanical properties such as toughness, shear strength, "
            "and tensile strength.",
            "To boost the size of the grains of solution, which enhances "
            "mechanical properties such as toughness, shear strength, and tensile "
            "strength.",
            "To prevent the grains of solution from growing too large, which "
            "enhances mechanical properties such as toughness, shear strength, "
            "and tensile strength.",
            "To prevent the grains of solution from growing too small, which "
            "enhances mechanical properties such as toughness, shear strength, "
            "and tensile strength.",
            "To raise the size of the grains of solution, which reduces "
            "mechanical properties such as toughness, shear strength, and tensile "
            "strength.",
        ],
    },
    {
        "id": 80,
        "prompt": "Which of the following is correct? What is the reason for heating "
            "metals to a temperature just above the upper critical temperature? "
            "among the listed options.",
        "options": [
            "To prevent the grains of solution from growing too large, which "
            "reduces mechanical properties such as toughness, shear strength, and "
            "tensile strength.",
            "To boost the size of the grains of solution, which enhances "
            "mechanical properties such as toughness, shear strength, and tensile "
            "strength.",
            "To prevent the grains of solution from growing too large, which "
            "enhances mechanical properties such as toughness, shear strength, "
            "and tensile strength.",
            "To prevent the grains of solution from growing too small, which "
            "enhances mechanical properties such as toughness, shear strength, "
            "and tensile strength.",
            "To raise the size of the grains of solution, which minimizes "
            "mechanical properties such as toughness, shear strength, and tensile "
            "strength.",
        ],
    },
    {
        "id": 126,
        "prompt": "Which of the following is correct? What happens to an initially "
            "inhomogeneous physical framework that is isolated by a thermodynamic "
            "operation? based on the given context.",
        "options": [
            "It will change its internal state only if it is composed of a single "
            "submechanism and has internal walls.",
            "It will change its internal state only if it is composed of several "
            "substructures separated from each other by walls.",
            "It will remain in its initial state indefinitely.",
            "It will generally change its internal state over time.",
            "It will change its internal state only if it is composed of a single "
            "subframework.",
        ],
    },
    {
        "id": 145,
        "prompt": "Determine the correct option: Which of the following statements is "
            "true about the categorization of planetary structures according to "
            "their orbital dynamics?",
        "options": [
            "Planetary frameworks cannot be categorized based on their orbital "
            "dynamics.",
            "Planetary mechanisms can be categorized as resonant, "
            "non-resonant-interacting, hierarchical, or some combination of "
            "these, but only based on the number of planets in the mechanism.",
            "Planetary frameworks can only be categorized as resonant or "
            "non-resonant-interacting.",
            "Planetary frameworks can be categorized as resonant, "
            "non-resonant-interacting, hierarchical, or some combination of "
            "these.",
            "Planetary frameworks can only be categorized as hierarchical or "
            "non-hierarchical.",
        ],
    },
]

def _load(example, source, answer=None):
    """Put an example into the input boxes and record what it was."""
    pool_key = "example_id"
    st.session_state[pool_key] = example["id"]
    st.session_state["example_source"] = source
    st.session_state["example_answer"] = answer
    st.session_state["prompt"] = example["prompt"]
    for letter, text in zip(LETTERS, example["options"]):
        st.session_state[f"opt_{letter}"] = text


def _pick(pool):
    """Random choice that never repeats the previous example."""
    fresh = [e for e in pool if e["id"] != st.session_state.get("example_id")]
    return random.choice(fresh or pool)


left, right = st.columns(2)
with left:
    if st.button("Load a seen question", use_container_width=True):
        chosen = _pick(EXAMPLES)
        _load(chosen, "seen", chosen["answer"])
with right:
    if st.button("Load an unseen question", use_container_width=True):
        _load(_pick(UNSEEN_EXAMPLES), "unseen")

if st.session_state.get("example_source") == "seen":
    st.caption(
        f"train.csv row {st.session_state['example_id']}, verbatim. The model was "
        f"trained on this question, so expect a confident answer. Its labelled "
        f"answer is **{st.session_state['example_answer']}** - press Solve and check."
    )
elif st.session_state.get("example_source") == "unseen":
    st.caption(
        f"test.csv row {st.session_state['example_id']}. This question appears "
        "nowhere in the training data - neither the prompt nor the option set - so "
        "the model is genuinely reasoning rather than recalling. Only 38 of the 500 "
        "test questions qualify. Its label is not public, so watch the confidence "
        "spread instead: that is what the top-3 of MAP@3 exists for."
    )

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
# Deliberately does not name the backbone: the deployed checkpoint can be swapped
# (BERT / RoBERTa / DeBERTa) without this line going stale. The model card on the
# Hub carries the specific architecture and scores.
st.caption(
    f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) - a transformer with a "
    "multiple-choice head, fully fine-tuned on 2,000 questions. "
    "See the model card for the backbone, training recipe and evaluation."
)
