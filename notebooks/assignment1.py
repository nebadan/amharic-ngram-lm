# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NLP Assignment 1 — Amharic N-gram Language Models
# Addis Ababa University · Corpus: GPAC (Gereme et al., *Information* 12(1):20, 2021)
#
# | Task | Part |
# |---|---|
# | #1 N-gram model (1.1–1.8) | 1 |
# | #2 Intrinsic evaluation | 2 |
# | #3 Extrinsic evaluation | 3 |
# | #4 Next-word suggestion | 4 |
#
# Code: `amharic_lm/` (n-grams stored as 64-bit hashes in sorted numpy arrays).

# %%
import gc, json, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display, Markdown

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from amharic_lm.text import normalize, sentences, tokenize, STOPWORDS
from amharic_lm.ngram import NgramLM, Vocab, BOS, EOS, UNK
from amharic_lm import corpus as C
from amharic_lm import evaluate as E

DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Ethiopic"]  # Latin/math first, Ethiopic as fallback
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_colwidth", 80)
RESULTS = {}
T0 = time.time()

def save_results():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=1, default=float)

# %% [markdown]
# ---
# # Part 1 — N-gram language model
# ## 1.1 Preprocessing and corpus statistics
# NFC → homophone normalization (ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ) → sentence split on `።` `፡፡` `፧` `?` `!` → tokenize (punctuation removed, numbers → `<NUM>`, Latin dropped).

# %%
RAW, CLEAN = DATA / "GPAC.txt", DATA / "gpac_clean.txt"
if not CLEAN.exists() or not (OUT / "raw_stats.json").exists():
    raw_stats = C.preprocess_file(RAW, CLEAN)
    json.dump(raw_stats, open(OUT / "raw_stats.json", "w"))
raw_stats = json.load(open(OUT / "raw_stats.json"))

# example of the pipeline on raw text
sample_raw = open(RAW, encoding="utf-8").read(700)
print("RAW :", sample_raw[:300], "...\n")
for s in list(sentences(sample_raw))[:4]:
    print("SENT:", s)

# %%
word_counts = C.count_words(CLEAN)
sent_lens = np.array([len(l.split()) for l in open(CLEAN, encoding="utf-8")])
n_tokens, n_types = int(sent_lens.sum()), len(word_counts)
n_unique_sents = len({hash(l) for l in open(CLEAN, encoding="utf-8")})
hapax = sum(1 for c in word_counts.values() if c == 1)
char_lens = np.array([len(w) for w in word_counts for _ in range(1)])
avg_word_len_tok = sum(len(w) * c for w, c in word_counts.items()) / n_tokens

stats = {
    "Raw file size (MB)": os.path.getsize(RAW) / 1e6,
    "Raw characters": raw_stats["raw_chars"],
    "Raw whitespace tokens": raw_stats["raw_ws_tokens"],
    "Sentences": len(sent_lens),
    "Unique sentences": n_unique_sents,
    "Repeated sentences (%)": 100 * (1 - n_unique_sents / len(sent_lens)),
    "Tokens (after cleaning)": n_tokens,
    "Vocabulary size (types)": n_types,
    "Type/token ratio": n_types / n_tokens,
    "Hapax legomena (freq = 1)": hapax,
    "Hapax % of vocabulary": 100 * hapax / n_types,
    "Mean sentence length (tokens)": sent_lens.mean(),
    "Median sentence length": float(np.median(sent_lens)),
    "Max sentence length": int(sent_lens.max()),
    "Mean word length (chars, by token)": avg_word_len_tok,
    "Mean word length (chars, by type)": char_lens.mean(),
    "<NUM> tokens": word_counts["<NUM>"],
    "Stop-word tokens (%)": 100 * sum(word_counts[w] for w in STOPWORDS) / n_tokens,
}
RESULTS["corpus_stats"] = stats
display(pd.DataFrame({"value": stats}).style.format("{:,.3f}"))

# %%
top_words = word_counts.most_common(25)
display(pd.DataFrame(top_words, columns=["word", "count"]).assign(
    pct=lambda d: 100 * d["count"] / n_tokens).T)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
freqs = np.array(sorted(word_counts.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)
ax[0].loglog(ranks, freqs, lw=1.5)
slope = np.polyfit(np.log(ranks[10:100000]), np.log(freqs[10:100000]), 1)[0]
ax[0].set(title=f"Zipf plot (slope ≈ {slope:.2f})", xlabel="rank", ylabel="frequency")
ax[1].hist(np.clip(sent_lens, 0, 80), bins=80, color="#4C72B0")
ax[1].set(title="Sentence length", xlabel="tokens (clipped at 80)", ylabel="sentences")
w, c = zip(*top_words[:15])
ax[2].barh(range(15)[::-1], c, color="#55A868")
ax[2].set_yticks(range(15)[::-1], w)
ax[2].set(title="Top 15 words", xlabel="count")
plt.tight_layout(); plt.savefig(FIG / "corpus_stats.png", bbox_inches="tight"); plt.show()

# vocabulary growth (Heaps' law) on the first 20M tokens
seen, growth, n = set(), [], 0
for toks in C.iter_clean(CLEAN):
    for t in toks:
        seen.add(t); n += 1
        if n % 200_000 == 0:
            growth.append((n, len(seen)))
    if n >= 20_000_000:
        break
g = np.array(growth)
b, logk = np.polyfit(np.log(g[:, 0]), np.log(g[:, 1]), 1)
RESULTS["corpus_stats"].update({"Zipf slope": slope, "Heaps beta": b, "Heaps K": float(np.exp(logk))})
plt.figure(figsize=(5.5, 3.5))
plt.plot(g[:, 0] / 1e6, g[:, 1] / 1e3)
plt.title(f"Vocabulary growth (Heaps: V = {np.exp(logk):.1f}·N^{b:.2f})")
plt.xlabel("tokens (millions)"); plt.ylabel("types (thousands)")
plt.tight_layout(); plt.savefig(FIG / "heaps.png", bbox_inches="tight"); plt.show()
del seen; gc.collect()

# %% [markdown]
# Large vocabulary and many hapaxes: Amharic attaches prepositions and suffixes to words. About a quarter of sentences are repeats; they are removed before evaluation (Part 2).
#
# ## 1.2 Creating n-grams (n = 1–4)
# Sentences are padded `<s> <s> <s> w1 … wn </s>`; n-grams never cross sentences.

# %%
vocab_full = Vocab.from_counter(word_counts, min_count=1)
full = C.encode(CLEAN, vocab_full)
t = time.time()
lm_full = NgramLM.build(full, vocab_full, order=4, kn=True)
print(f"built in {time.time() - t:.0f}s")

first = open(CLEAN, encoding="utf-8").readline().split()
print("Sentence:", " ".join(first), "\n")
padded = [BOS] * 3 + first + [EOS]
for n in range(1, 5):
    grams = [tuple(padded[i:i + n]) for i in range(len(padded) - n + 1)]
    grams = [g for g in grams if g.count(BOS) <= n - 1 and EOS not in g[:-1]]
    print(f"{n}-grams ({len(grams)}):", ["_".join(g) for g in grams[:8]], "...")

ngram_counts = pd.DataFrame({
    "n": [1, 2, 3, 4],
    "unique n-grams": [len(lm_full.tables[k].keys) for k in range(1, 5)],
    "total n-grams": [int(lm_full.tables[k].count.sum()) for k in range(1, 5)],
    "singletons (%)": [100 * (lm_full.tables[k].count == 1).mean() for k in range(1, 5)],
})
RESULTS["ngram_counts"] = ngram_counts.to_dict("records")
display(ngram_counts.style.format({"unique n-grams": "{:,}", "total n-grams": "{:,}", "singletons (%)": "{:.1f}"}))

# %% [markdown]
# ## 1.3 N-gram probabilities and top-10 n-grams
# $P(w_1..w_n) = C(w_1..w_n) / \sum C(\cdot)$

# %%
def top_table(lm, n, k=10, skip=(BOS, EOS)):
    rows = lm.top_ngrams(n, k, skip=skip)
    return pd.DataFrame([(" ".join(g), c, p) for g, c, p in rows], columns=[f"{n}-gram", "count", "P"])

RESULTS["top_ngrams"] = {}
for n in range(1, 5):
    df = top_table(lm_full, n)
    RESULTS["top_ngrams"][n] = df.values.tolist()
    display(Markdown(f"**Top 10 {n}-grams**"))
    display(df.style.format({"count": "{:,}", "P": "{:.2e}"}))

# %% [markdown]
# ## 1.4 Conditional probability (bigrams)
# $P(w_i \mid w_{i-1}) = C(w_{i-1} w_i) / C(w_{i-1}\,\cdot)$

# %%
def cond_prob(prev, w, lm=lm_full):
    V = lm.vocab
    ids, cnt, _ = lm.followers([V.index.get(normalize(prev), V.unk)], 2)
    tot = cnt.sum()
    wid = V.index.get(normalize(w), -1)
    c = int(cnt[ids == wid].sum()) if tot else 0
    return c, int(tot), (c / tot if tot else 0.0)

pairs = [("ኢትዮጵያ", "ውስጥ"), ("አዲስ", "አበባ"), ("ጠቅላይ", "ሚኒስትር"), ("በአፍሪካ", "ቀንድ"),
         ("የቡና", "ምርት"), ("የኢትዮጵያ", "ህዝብ"), ("ብሔራዊ", "ባንክ"), ("ቀንድ", "ትገኛለች")]
df = pd.DataFrame([(a, b, *cond_prob(a, b)) for a, b in pairs],
                  columns=["previous", "word", "C(prev w)", "C(prev ·)", "P(word | previous)"])
RESULTS["conditional"] = df.values.tolist()
display(df.style.format({"P(word | previous)": "{:.4f}"}))

def top_followers(prev, k=6, lm=lm_full):
    V = lm.vocab
    ids, cnt, _ = lm.followers([V.index[normalize(prev)]], 2)
    o = np.argsort(-cnt)[:k]
    return [(V.words[ids[i]], round(cnt[i] / cnt.sum(), 4)) for i in o]

RESULTS["followers"] = {}
for w in ["ኢትዮጵያ", "አዲስ", "የቡና", "ጠቅላይ", "በአፍሪካ"]:
    RESULTS["followers"][w] = top_followers(w)
    print(f"P(· | {w}):", top_followers(w))

# %% [markdown]
# ## 1.5 Stop-word removal
# 163 curated function words plus `<NUM>` are removed; n-grams are rebuilt.

# %%
print(f"{len(STOPWORDS)} stop-words, e.g.:", sorted(STOPWORDS)[:30])
sw_counts = C.count_words(CLEAN, remove_stopwords=True)
vocab_sw = Vocab.from_counter(sw_counts, 1)
sw = C.encode(CLEAN, vocab_sw, remove_stopwords=True)
lm_sw = NgramLM.build(sw, vocab_sw, order=4, kn=False)
RESULTS["stopword_tokens_removed"] = int(len(full[0]) - len(sw[0]))
print(f"tokens before: {len(full[0]):,}  after: {len(sw[0]):,}  "
      f"({100 * (1 - len(sw[0]) / len(full[0])):.1f}% removed)")
RESULTS["top_ngrams_nostop"] = {}
for n in range(1, 5):
    df = top_table(lm_sw, n)
    RESULTS["top_ngrams_nostop"][n] = df.values.tolist()
    display(Markdown(f"**Top 10 {n}-grams after stop-word removal**"))
    display(df.style.format({"count": "{:,}", "P": "{:.2e}"}))

# %% [markdown]
# ## 1.6 Word clouds
# `<NUM>` stays in the model but is left out of the clouds.

# %%
from wordcloud import WordCloud

def cloud(freqs):
    return WordCloud(font_path=FONT, width=900, height=500, background_color="white",
                     colormap="viridis", max_words=150, prefer_horizontal=1.0,
                     regexp=r"\S+", collocations=False).generate_from_frequencies(freqs)

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
for n in range(1, 5):
    for j, (lm, title) in enumerate([(lm_full, "with stop-words"), (lm_sw, "stop-words removed")]):
        freqs = lm.frequencies(n, 150, skip=(BOS, EOS, "<NUM>"))  # words joined by spaces; no <NUM>
        img = cloud(freqs)
        img.to_file(str(FIG / f"wordcloud_{n}gram_{'sw' if j else 'all'}.png"))
        axes[n - 1, j].imshow(img, interpolation="bilinear")
        axes[n - 1, j].set_title(f"{n}-grams — {title}", fontsize=14)
        axes[n - 1, j].axis("off")
plt.tight_layout(); plt.savefig(FIG / "wordclouds.png", bbox_inches="tight", dpi=80); plt.show()

# %% [markdown]
# With stop-words: function words dominate. Without: names and topics (አዲስ አበባ, ጠቅላይ ሚኒስትር).

# %%
del lm_sw, sw, sw_counts, vocab_sw; gc.collect()

# %% [markdown]
# ## 1.7 Probability of a sentence
# $P(w_1..w_m) \approx \prod_i P(w_i \mid w_{i-n+1}..w_{i-1})$, including `</s>`; MLE vs. Kneser-Ney, as log₁₀.

# %%
test_sents = ["ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።",
              "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
              "ጠቅላይ ሚኒስትሩ ዛሬ ለፓርላማው ሪፖርት አቀረቡ።",
              "ቡና የኢትዮጵያ ዋነኛ የውጭ ምንዛሪ ምንጭ ነው።",
              "ትገኛለች ቀንድ በአፍሪካ ኢትዮጵያ።"]  # scrambled control

rows = []
for s in test_sents:
    words = next(sentences(s))
    for n in range(1, 5):
        lp_mle, _, _ = lm_full.sentence_logprob(words, n, "mle")
        lp_kn, p, _ = lm_full.sentence_logprob(words, n, "kn")
        rows.append({"sentence": s, "n": n,
                     "log10 P (MLE)": lp_mle / np.log(10),
                     "P (MLE)": np.exp(lp_mle),
                     "log10 P (KN)": lp_kn / np.log(10),
                     "perplexity (KN)": np.exp(-lp_kn / len(p))})
sp = pd.DataFrame(rows)
RESULTS["sentence_probs"] = sp.replace([np.inf, -np.inf], None).values.tolist()
display(sp.style.format({"log10 P (MLE)": "{:.2f}", "P (MLE)": "{:.3e}", "log10 P (KN)": "{:.2f}",
                         "perplexity (KN)": "{:,.1f}"}))

words = next(sentences(test_sents[0]))
_, p_mle, tg = lm_full.sentence_logprob(words, 3, "mle")
_, p_kn, _ = lm_full.sentence_logprob(words, 3, "kn")
ctx = [BOS, BOS] + words
breakdown = pd.DataFrame({"P(w | 2 previous) — trigram": [f"P({w} | {ctx[i]} {ctx[i+1]})" for i, w in enumerate(tg)],
                          "MLE": p_mle, "Kneser-Ney": p_kn})
RESULTS["sentence_breakdown"] = breakdown.values.tolist()
display(breakdown.style.format({"MLE": "{:.4f}", "Kneser-Ney": "{:.4f}"}))

# %% [markdown]
# ## 1.8 Random sentence generation
# Sample each next word from the MLE distribution until `</s>`.

# %%
rng = np.random.default_rng(7)
gen = {}
for n in range(1, 5):
    gen[n] = [" ".join(lm_full.generate(n, rng, max_len=30)) for _ in range(4)]
    print(f"\n=== {n}-gram ===")
    for s in gen[n]:
        print(" •", s + " ።")
RESULTS["generated"] = gen

# %% [markdown]
# **As n increases:** unigram = random words → bigram = grammatical pairs → trigram = fluent clauses → 4-gram = copies of training sentences (70% of 4-grams are seen once). Fluency up, novelty down.

# %%
del lm_full, full; gc.collect()
save_results()

# %% [markdown]
# ---
# # Part 2 — Intrinsic evaluation (perplexity)
# Deduplicate sentences → 98% train / 2% test (seed 42) → words seen < 2 times become `<UNK>`.
# $PP = \exp\big(-\tfrac{1}{N}\sum \ln P(w_i \mid h_i)\big)$. Compared: MLE, Laplace, add-k (tuned), interpolated Kneser-Ney.

# %%
DEDUP = DATA / "gpac_dedup.txt"
seen = set()
with open(CLEAN, encoding="utf-8") as f, open(DEDUP, "w", encoding="utf-8") as out:
    for line in f:
        h = hash(line)
        if h not in seen:
            seen.add(h)
            out.write(line)
del seen; gc.collect()
counts_all = C.count_words(DEDUP)
vocab_all = Vocab.from_counter(counts_all, 1)
enc = C.encode(DEDUP, vocab_all)
(train_raw, test_raw) = C.split(*enc, test_frac=0.02, seed=42)
(train_raw, dev_raw) = C.split(*train_raw, test_frac=0.005, seed=1)
del enc; gc.collect()

# training vocabulary: words seen >= 2 times in train
tr_counts = np.bincount(train_raw[0], minlength=len(vocab_all))
keep_words = [vocab_all.words[i] for i in np.flatnonzero(tr_counts >= 2) if i > 2]
vocab = Vocab([BOS, EOS, UNK] + sorted(keep_words, key=lambda w: -tr_counts[vocab_all.index[w]]))
remap = np.full(len(vocab_all), vocab.unk, np.int32)
remap[[vocab_all.index[w] for w in vocab.words[3:]]] = np.arange(3, len(vocab), dtype=np.int32)
remap[0], remap[1] = 0, 1
train = (remap[train_raw[0]], train_raw[1])
test = (remap[test_raw[0]], test_raw[1])
dev = (remap[dev_raw[0]], dev_raw[1])
del train_raw, test_raw, dev_raw, counts_all; gc.collect()

split_info = {"train sentences": len(train[1]), "train tokens": len(train[0]),
              "dev sentences": len(dev[1]), "test sentences": len(test[1]),
              "test tokens": len(test[0]), "vocabulary (min count 2)": len(vocab),
              "test OOV rate (%)": 100 * float((test[0] == vocab.unk).mean())}
RESULTS["split"] = split_info
display(pd.DataFrame({"value": split_info}).style.format("{:,.2f}"))

t = time.time()
lm = NgramLM.build(train, vocab, order=4, kn=True)
print(f"built in {time.time() - t:.0f}s")

# %%
# tune k for add-k on the dev set
dev_l = C.to_lists(dev)
ks = [1.0, 0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00001]
best_k = {}
for n in range(1, 5):
    pps = {k: lm.perplexity(dev_l, n, "addk", k)[0] for k in ks}
    best_k[n] = min(pps, key=pps.get)
    print(f"n={n}: " + "  ".join(f"k={k}: {v:,.0f}" for k, v in pps.items()), "-> best", best_k[n])
RESULTS["best_k"] = best_k

# %%
test_l = C.to_lists(test)
rows = []
for n in range(1, 5):
    pp_mle, zero = lm.perplexity(test_l, n, "mle")
    rows.append({"n": n,
                 "MLE": pp_mle, "MLE: % zero-prob tokens": 100 * zero,
                 "Laplace (add-1)": lm.perplexity(test_l, n, "addk", 1.0)[0],
                 f"add-k (tuned)": lm.perplexity(test_l, n, "addk", best_k[n])[0],
                 "k": best_k[n],
                 "Kneser-Ney": lm.perplexity(test_l, n, "kn")[0]})
pp = pd.DataFrame(rows).set_index("n")
RESULTS["perplexity"] = pp.reset_index().replace([np.inf], None).to_dict("records")
display(pp.style.format("{:,.1f}"))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for col, m in [("Laplace (add-1)", "o-"), ("add-k (tuned)", "s-"), ("Kneser-Ney", "^-")]:
    ax[0].plot(pp.index, pp[col], m, label=col)
ax[0].set(yscale="log", xticks=[1, 2, 3, 4], xlabel="n", ylabel="test perplexity (log scale)",
          title="Perplexity by order and smoothing")
ax[0].legend()
ax[1].bar(pp.index, pp["MLE: % zero-prob tokens"], color="#C44E52")
ax[1].set(xticks=[1, 2, 3, 4], xlabel="n", ylabel="% of test tokens",
          title="MLE: test tokens with zero probability")
plt.tight_layout(); plt.savefig(FIG / "perplexity.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ### Extra: training-set size

# %%
lc = []
for frac in [0.05, 0.15, 0.4]:
    sub = C.subset(train, frac, seed=3)
    m = NgramLM.build(sub, vocab, order=4, kn=True, verbose=False)
    lc.append({"train tokens (M)": len(sub[0]) / 1e6,
               **{f"n={n}": m.perplexity(test_l, n, "kn")[0] for n in range(1, 5)}})
    del m, sub; gc.collect()
lc.append({"train tokens (M)": len(train[0]) / 1e6, **{f"n={n}": pp.loc[n, "Kneser-Ney"] for n in range(1, 5)}})
lc = pd.DataFrame(lc)
RESULTS["learning_curve"] = lc.to_dict("records")
display(lc.style.format("{:,.1f}"))
plt.figure(figsize=(6, 4))
for n in range(1, 5):
    plt.plot(lc["train tokens (M)"], lc[f"n={n}"], "o-", label=f"{n}-gram")
plt.xscale("log"); plt.xlabel("training tokens (millions, log)"); plt.ylabel("KN test perplexity")
plt.title("More data helps higher orders most"); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "learning_curve.png", bbox_inches="tight"); plt.show()
save_results()

# %% [markdown]
# **Findings:** MLE gives zero probabilities (∞ perplexity); add-one gets worse with n; Kneser-Ney improves at every order with diminishing returns; higher orders need more data.

# %% [markdown]
# ---
# # Part 3 — Extrinsic evaluation
# 1. **Next-word prediction**: top-k accuracy and MRR on 5,000 test positions.
# 2. **Keystroke savings**: simulated keyboard with 5 suggestions.
# 3. **Real vs. corrupted sentence**: swap / replace / shuffle (chance = 50%).

# %%
ext = {}
for n in range(1, 5):
    t = time.time()
    r = {"n": n}
    r.update({f"NWP {k}": v for k, v in E.next_word_prediction(lm, test_l, n, 5000).items()
              if k in ("top1", "top3", "top5", "top10", "mrr@10")})
    r["KSS (top-5)"] = E.keystroke_savings(lm, test_l, n, k=5, max_words=1500)["kss"]
    for kind in ("swap", "replace", "shuffle"):
        r[f"discrim. {kind}"] = E.corruption_discrimination(lm, test_l, n, kind, max_sents=3000)["accuracy"]
    ext[n] = r
    print(f"n={n} done in {time.time() - t:.0f}s")
ext = pd.DataFrame(ext.values()).set_index("n")
RESULTS["extrinsic"] = ext.reset_index().to_dict("records")
display((ext * 100).style.format("{:.1f}"))

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for c in ["NWP top1", "NWP top5", "NWP top10"]:
    ax[0].plot(ext.index, 100 * ext[c], "o-", label=c.replace("NWP ", ""))
ax[0].set(title="Next-word prediction accuracy", xlabel="n", ylabel="%", xticks=[1, 2, 3, 4]); ax[0].legend()
ax[1].bar(ext.index, 100 * ext["KSS (top-5)"], color="#8172B2")
ax[1].set(title="Keystroke savings (5 suggestions)", xlabel="n", ylabel="% keystrokes saved", xticks=[1, 2, 3, 4])
for c in ["discrim. swap", "discrim. replace", "discrim. shuffle"]:
    ax[2].plot(ext.index, 100 * ext[c], "o-", label=c.replace("discrim. ", ""))
ax[2].axhline(50, ls="--", c="gray", lw=1)
ax[2].set(title="Real vs. corrupted sentence", xlabel="n", ylabel="accuracy %", xticks=[1, 2, 3, 4]); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG / "extrinsic.png", bbox_inches="tight"); plt.show()

# intrinsic vs extrinsic
plt.figure(figsize=(5.5, 4))
plt.plot(pp["Kneser-Ney"], 100 * ext["NWP top5"], "o-")
for n in range(1, 5):
    plt.annotate(f"n={n}", (pp.loc[n, "Kneser-Ney"], 100 * ext.loc[n, "NWP top5"]),
                 textcoords="offset points", xytext=(6, 4))
plt.xlabel("KN perplexity (lower is better)"); plt.ylabel("top-5 next-word accuracy %")
plt.title("Intrinsic vs extrinsic"); plt.tight_layout()
plt.savefig(FIG / "intrinsic_vs_extrinsic.png", bbox_inches="tight"); plt.show()
save_results()

# %% [markdown]
# **Findings:** same ranking as perplexity (1 ≪ 2 < 3 < 4). Unigram = 50% on swap/shuffle (ignores order). About half the keystrokes are saved.

# %% [markdown]
# ---
# # Part 4 — Next-word suggestion
# Input → last 1–2 words → candidates seen after them → Kneser-Ney probability → top 5 (word completion if the last word is unfinished). App: `app/` and the Hugging Face Space.

# %%
def suggest(text, n=3, k=5, lm=lm):
    """Top-k next-word suggestions for an incomplete Amharic sentence."""
    text = normalize(text)
    toks = tokenize(text)
    prefix = ""
    if text and not text[-1].isspace() and toks:   # last word still being typed
        prefix = toks.pop()
    context = toks[-(n - 1):] if n > 1 else []
    return context, prefix, lm.topk_next(context, n=n, k=k, prefix=prefix)

for q in ["ኢትዮጵያ ውስጥ የቡና ", "ኢትዮጵያ በአፍሪካ ", "አዲስ ", "ጠቅላይ ሚኒስትር ", "የኢትዮጵያ ህዝብ ", "ኢትዮጵያ ውስጥ የቡና ም"]:
    ctx, pre, s = suggest(q)
    print(f"Input: {q!r:28} context={ctx} prefix={pre!r}")
    for w, p in s:
        print(f"      {w:<14} {p:.4f}")
RESULTS["suggestions"] = {q: suggest(q)[2] for q in ["ኢትዮጵያ ውስጥ የቡና ", "ኢትዮጵያ በአፍሪካ ", "አዲስ ", "ጠቅላይ ሚኒስትር "]}

# %%
# Simulated composition: always pick the 1st suggestion (greedy) and the 2nd one
for pick in (0, 1):
    text = "ኢትዮጵያ "
    for _ in range(8):
        s = suggest(text)[2]
        if not s or s[min(pick, len(s) - 1)][0] == EOS:
            break
        text += s[min(pick, len(s) - 1)][0] + " "
    print(f"choose suggestion #{pick + 1} each time →", text.strip() + " ።")

# %% [markdown]
# **App model:** n-grams seen < 3 times are pruned to shrink the model; the cost is measured below.

# %%
APP_MODEL = ROOT / "app" / "model" / "amharic_4gram_kn.npz"
APP_MODEL.parent.mkdir(parents=True, exist_ok=True)
lm.save(APP_MODEL, min_count=3)
print(f"saved {APP_MODEL.name}: {APP_MODEL.stat().st_size / 1e6:.0f} MB")
lm_small = NgramLM.load(APP_MODEL)
print("pruned model:", suggest("ኢትዮጵያ ውስጥ የቡና ", lm=lm_small)[2])
prune_cmp = pd.DataFrame([
    {"model": "full", "n": n, **E.next_word_prediction(lm, test_l, n, 5000)} for n in (2, 3, 4)] + [
    {"model": "pruned (app)", "n": n, **E.next_word_prediction(lm_small, test_l, n, 5000)} for n in (2, 3, 4)]
)[["model", "n", "top1", "top5", "mrr@10", "ms_per_query"]]
RESULTS["pruning"] = prune_cmp.to_dict("records")
display(prune_cmp.style.format({"top1": "{:.3f}", "top5": "{:.3f}", "mrr@10": "{:.3f}", "ms_per_query": "{:.1f}"}))
RESULTS["app_model_mb"] = APP_MODEL.stat().st_size / 1e6
RESULTS["runtime_minutes"] = (time.time() - T0) / 60
save_results()
print(f"Total runtime: {(time.time() - T0) / 60:.1f} min")
