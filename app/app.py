"""Amharic predictive text — next-word suggestion with an n-gram (Kneser-Ney) model.

Type an incomplete Amharic sentence; the app takes the last one or two words,
retrieves candidate next words from the language model, scores them, and shows
the top five. Click a suggestion to add it and keep writing.
"""
import os
import re
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from amharic_lm.ngram import NgramLM, EOS  # noqa: E402
from amharic_lm.text import normalize, tokenize, SENT_END  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "model" / "amharic_4gram_kn.npz"))
print(f"Loading {MODEL_PATH} ...", flush=True)
LM = NgramLM.load(MODEL_PATH)
print(f"Loaded: vocabulary {len(LM.vocab):,} words, "
      + ", ".join(f"{k}-grams {len(t.keys):,}" for k, t in LM.tables.items()), flush=True)

K = 5
ORDERS = {"1 word (bigram)": 2, "2 words (trigram)": 3, "3 words (4-gram)": 4}
END_MARK = "።"


def analyse(text, order_label):
    """Return (context words, partial word, suggestions)."""
    n = ORDERS[order_label]
    norm = normalize(text or "")
    # only the sentence currently being written matters
    parts = SENT_END.split(norm)
    current = parts[-1] if parts else ""
    toks = tokenize(current)
    prefix = ""
    if current and not current[-1].isspace() and toks:
        prefix = toks.pop()            # last word is still being typed
    context = toks[-(n - 1):]
    sugg = LM.topk_next(context, n=n, k=K, prefix=prefix, allow_eos=not prefix)
    return context, prefix, sugg


def show(word):
    return END_MARK if word == EOS else word


def update(text, order_label):
    context, prefix, sugg = analyse(text, order_label)
    words = [show(w) for w, _ in sugg]
    buttons = [gr.update(value=w, visible=True) for w in words]
    buttons += [gr.update(value="", visible=False)] * (K - len(buttons))
    rows = [[i + 1, show(w), f"{p:.4f}", "█" * max(1, round(40 * p / sugg[0][1]))]
            for i, (w, p) in enumerate(sugg)]
    ctx = " ".join(context) if context else "⟨sentence start⟩"
    info = f"**Context used:** {ctx}"
    if prefix:
        info += f" · **completing:** `{prefix}…`"
    return [*buttons, rows, info]


def accept(text, word, order_label):
    """Insert the chosen suggestion (replacing a partly typed word)."""
    text = text or ""
    if word == END_MARK:
        new = text.rstrip() + END_MARK + " "
    else:
        _, prefix, _ = analyse(text, order_label)
        if prefix:
            m = re.search(r"\S+$", text)
            text = text[:m.start()] if m else text
        elif text and not text[-1].isspace():
            text += " "
        new = text + word + " "
    return new


def complete(text, order_label, max_words=12):
    """Greedily follow the top suggestion until the sentence ends."""
    for _ in range(max_words):
        _, prefix, sugg = analyse(text, order_label)
        if not sugg:
            break
        w = show(sugg[0][0])
        text = accept(text, w, order_label)
        if w == END_MARK:
            break
    return text


EXAMPLES = ["ኢትዮጵያ ውስጥ የቡና ", "ኢትዮጵያ በአፍሪካ ", "አዲስ አበባ ", "ጠቅላይ ሚኒስትር ",
            "የኢትዮጵያ ህዝብ ", "በዚህ ዓመት ", "ኢትዮጵያ ውስጥ የቡና ም"]

CSS = """
.gradio-container {max-width: 860px !important; margin: auto;}
#txt textarea {font-size: 1.35rem; line-height: 1.6;}
.sugg button {font-size: 1.2rem !important; min-height: 3rem;}
"""

with gr.Blocks(title="Amharic Next-Word Suggestion") as demo:
    gr.Markdown(
        "# የአማርኛ ቃል ጠቋሚ · Amharic Next-Word Suggestion\n"
        "Type an incomplete Amharic sentence. The model reads the last one or two words, finds "
        "the words that followed them in the **GPAC** corpus (~78M words), scores each with an "
        "interpolated **Kneser-Ney 4-gram** model, and shows the five most likely next words. "
        "Click one to add it and keep writing. If you stop in the middle of a word, the "
        "suggestions complete that word instead.")
    with gr.Row():
        order = gr.Radio(list(ORDERS), value="2 words (trigram)", label="Context length")
    txt = gr.Textbox(label="Your sentence", placeholder="ኢትዮጵያ ውስጥ የቡና …", lines=3,
                     elem_id="txt", autofocus=True)
    info = gr.Markdown()
    with gr.Row(elem_classes="sugg"):
        btns = [gr.Button(visible=False, variant="primary" if i == 0 else "secondary")
                for i in range(K)]
    table = gr.Dataframe(headers=["#", "word", "P(word | context)", ""], interactive=False,
                         datatype=["number", "str", "str", "str"], label="Top-5 with probabilities")
    with gr.Row():
        auto = gr.Button("Complete the sentence (greedy)")
        clear = gr.Button("Clear")
    gr.Examples(EXAMPLES, inputs=txt)
    gr.Markdown(
        "<small>NLP Assignment 1 — Addis Ababa University. Text is normalized like the corpus "
        "(e.g. ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ), so suggestions use the normalized spelling. "
        "`።` means the model predicts the end of the sentence.</small>")

    outs = [*btns, table, info]
    txt.change(update, [txt, order], outs, trigger_mode="always_last", show_progress="hidden")
    order.change(update, [txt, order], outs)
    for b in btns:
        b.click(accept, [txt, b, order], txt)
    auto.click(complete, [txt, order], txt)
    clear.click(lambda: "", None, txt)
    demo.load(update, [txt, order], outs)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
