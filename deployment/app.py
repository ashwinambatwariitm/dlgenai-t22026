"""Smart MCQ Solver - Hugging Face Space demo.

Loads the fine-tuned multiple-choice model selected by the project notebook and
ranks the five options of a question. The app is backbone-agnostic: it goes
through AutoTokenizer / AutoModelForMultipleChoice, so BERT, RoBERTa and
DeBERTa all work with no code change - only MODEL_ID below points at a
specific checkpoint.

How the model reads a question
------------------------------
It is NOT a 5-way classifier over the letters A-E. Each option is paired with
the prompt and scored on its own by the same shared encoder:

    (prompt, A) -> score        (prompt, D) -> score
    (prompt, B) -> score        (prompt, E) -> score
    (prompt, C) -> score        ->  softmax over the 5 scores

That makes the prediction independent of the order the options appear in, and
it is why the app reports a probability for every option rather than one label.

Set MODEL_ID below to the Hub repo you pushed the fine-tuned model to.
"""

import gradio as gr
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# The fine-tuned model pushed from the notebook's Step 10. The repo name is
# deliberately backbone-agnostic: whichever encoder wins in the notebook (BERT,
# RoBERTa, DeBERTa, ...) gets pushed here, and nothing below needs to change.
MODEL_ID = "ashwinambatwariitm/dlgenai-t22026-mcq"

LETTERS = ["A", "B", "C", "D", "E"]
MAX_LEN = 256

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
model.eval()
print("Model ready.")


@torch.no_grad()
def solve(prompt, option_a, option_b, option_c, option_d, option_e):
    """Score the five options and return them ranked best-first."""
    options = [option_a, option_b, option_c, option_d, option_e]

    if not prompt or not prompt.strip():
        return "Please enter a question.", {}
    filled = [i for i, o in enumerate(options) if o and o.strip()]
    if len(filled) < 2:
        return "Please fill in at least two options.", {}

    # Blank options must not win by accident, so remember which ones are real.
    texts = [o if (o and o.strip()) else "" for o in options]

    encoded = tokenizer(
        [prompt] * 5,
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    # AutoModelForMultipleChoice wants (batch, num_choices, seq_len).
    batch = {k: v.unsqueeze(0) for k, v in encoded.items()}

    logits = model(**batch).logits[0]                      # (5,)
    mask = torch.full_like(logits, float("-inf"))
    mask[filled] = 0.0                                     # ignore blank options
    probs = torch.softmax(logits + mask, dim=-1)

    ranked = sorted(filled, key=lambda i: float(probs[i]), reverse=True)
    top3 = " ".join(LETTERS[i] for i in ranked[:3])

    answer = (
        f"### Predicted answer: **{LETTERS[ranked[0]]}**\n\n"
        f"{texts[ranked[0]]}\n\n"
        f"---\n"
        f"Top-3 (the competition's MAP@3 format): `{top3}`"
    )
    confidences = {
        f"{LETTERS[i]}: {texts[i][:60]}{'...' if len(texts[i]) > 60 else ''}": float(probs[i])
        for i in ranked
    }
    return answer, confidences


EXAMPLE = [
    "Pick the best possible answer: What is Martin Heidegger's view on the "
    "relationship between time and human existence? among the listed options.",
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

with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown(
        "# Smart MCQ Solver\n"
        "A fine-tuned transformer that reads a multiple-choice question and ranks "
        "the five options. Each option is scored **independently** against the "
        "prompt by a shared encoder, so the prediction does not depend on the "
        "order the options are listed in.\n\n"
        "Enter a question and five options, then press **Solve**."
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Question", lines=3,
                                placeholder="Type the question here...")
            opt_a = gr.Textbox(label="Option A", lines=2)
            opt_b = gr.Textbox(label="Option B", lines=2)
            opt_c = gr.Textbox(label="Option C", lines=2)
            opt_d = gr.Textbox(label="Option D", lines=2)
            opt_e = gr.Textbox(label="Option E", lines=2)
            run = gr.Button("Solve", variant="primary")
        with gr.Column():
            answer = gr.Markdown(label="Answer")
            confidence = gr.Label(label="Confidence per option", num_top_classes=5)

    inputs = [prompt, opt_a, opt_b, opt_c, opt_d, opt_e]
    run.click(solve, inputs=inputs, outputs=[answer, confidence])
    gr.Examples(examples=[EXAMPLE], inputs=inputs)

if __name__ == "__main__":
    demo.launch()
