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
# # Reproducibility check — Amharic N-gram Language Models
# NLP Assignment 1 · Neibyu Daniel, Yonas Tadesse, Selamawit Siferh
#
# Re-runs everything from the raw GPAC file (downloaded and SHA-256 checked) and compares the
# results with `outputs/results.json`. Ends with a **PASS / FAIL table**. Fixed seeds throughout.
#
# | Mode | RAM | Time | Checks |
# |---|---|---|---|
# | Full (`SCALE = 1.0`) | ≥ 24 GB | ~20 min | every number, exactly |
# | Quick (`SCALE < 1`, e.g. Colab) | ~8 GB | ~5 min | the trends |
#
# Run: *Runtime → Run all*.

# %% [markdown]
# ## 0. Configuration

# %%
SCALE = None           # None = choose automatically from available RAM; or set 1.0 / 0.25 / 0.1
REPO_URL = "https://github.com/nebadan/amharic-ngram-lm.git"   # used only when run outside the repo
GPAC_DRIVE_ID = "1jEOAORylTBAiNKes_JerD6gFF7cnLFyr"            # link given in the assignment
GPAC_SHA256 = "977706859188f7598ca97a29d7203d86926c11ede591fba31915c74ae9eabcfa"
GPAC_BYTES = 1_063_098_071
TOL = 1e-6             # relative tolerance for full-mode comparisons

# %% [markdown]
# ## 1. Set-up: code, packages, data

# %%
import os, sys, subprocess, json, hashlib, time, gc
from pathlib import Path

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

# find the repository (or clone it, e.g. on Colab)
here = Path.cwd()
ROOT = next((p for p in [here, here.parent] if (p / "amharic_lm" / "ngram.py").exists()), None)
if ROOT is None:
    sh(f"git clone --depth 1 {REPO_URL} amharic-ngram-lm")
    ROOT = here / "amharic-ngram-lm"
sys.path.insert(0, str(ROOT))
print("repository:", ROOT)

try:
    import numpy, pandas, gdown  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q numpy pandas gdown")

import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from amharic_lm.text import normalize, sentences, tokenize, STOPWORDS
from amharic_lm.ngram import NgramLM, Vocab, BOS, EOS
from amharic_lm import corpus as C
from amharic_lm import evaluate as E

REPORTED = json.load(open(ROOT / "outputs" / "results.json", encoding="utf-8"))

def ram_gb():
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
        return kb / 1024 ** 2
    except Exception:
        return 0.0

if SCALE is None:
    SCALE = 1.0 if ram_gb() >= 24 else 0.25
FULL = SCALE >= 1.0
print(f"RAM {ram_gb():.0f} GB → SCALE = {SCALE} ({'full, exact comparison' if FULL else 'quick, qualitative checks'})")

# %%
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
RAW = DATA / "GPAC.txt"
if not RAW.exists() or RAW.stat().st_size != GPAC_BYTES:
    import gdown
    gdown.download(id=GPAC_DRIVE_ID, output=str(RAW), quiet=False)

h = hashlib.sha256()
with open(RAW, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
assert h.hexdigest() == GPAC_SHA256, "GPAC.txt differs from the file we used"
print("GPAC.txt verified:", RAW.stat().st_size, "bytes, sha256 OK")

# %% [markdown]
# ## 2. Preprocessing (Task 1.1)

# %%
T0 = time.time()
CLEAN = DATA / "gpac_clean_repro.txt"      # always rebuilt from the raw file
raw_stats = C.preprocess_file(RAW, CLEAN)
n_sent = sum(1 for _ in open(CLEAN, encoding="utf-8"))
n_tok = sum(len(l.split()) for l in open(CLEAN, encoding="utf-8"))
print(f"sentences {n_sent:,}  tokens {n_tok:,}  ({time.time() - T0:.0f}s)")

# quick mode: a fixed random subset of sentences
WORK = CLEAN
if not FULL:
    rng = np.random.default_rng(123)
    WORK = DATA / f"gpac_clean_{int(SCALE * 100)}pct.txt"
    with open(CLEAN, encoding="utf-8") as f, open(WORK, "w", encoding="utf-8") as out:
        for line in f:
            if rng.random() < SCALE:
                out.write(line)
    print(f"quick mode: using {sum(1 for _ in open(WORK, encoding='utf-8')):,} sentences")

CHECKS = []   # (task, metric, reported, reproduced, ok)
def check(task, metric, reported, reproduced, ok=None):
    if ok is None:
        if isinstance(reported, (int, float)) and isinstance(reproduced, (int, float)):
            ok = abs(reproduced - reported) <= TOL * max(1.0, abs(reported))
        else:
            ok = reported == reproduced
    CHECKS.append((task, metric, reported, reproduced, "PASS" if ok else "FAIL"))

RS = REPORTED["corpus_stats"]
check("1.1", "sentences", RS["Sentences"], n_sent)
check("1.1", "tokens after cleaning", RS["Tokens (after cleaning)"], n_tok)

# %% [markdown]
# ## 3. Task 1 — n-gram model on the (sub)corpus

# %%
word_counts = C.count_words(WORK)
vocab_full = Vocab.from_counter(word_counts, min_count=1)
full = C.encode(WORK, vocab_full)
lm_full = NgramLM.build(full, vocab_full, order=4, kn=True)

counts = [len(lm_full.tables[k].keys) for k in range(1, 5)]
singles = [100 * float((lm_full.tables[k].count == 1).mean()) for k in range(1, 5)]
display(pd.DataFrame({"n": [1, 2, 3, 4], "distinct n-grams": counts, "seen once %": np.round(singles, 1)}))

top = {n: [(" ".join(g), c) for g, c, _ in lm_full.top_ngrams(n, 10, skip=(BOS, EOS))] for n in range(1, 5)}
for n in range(1, 5):
    print(f"top {n}-grams:", [g for g, _ in top[n][:5]])

def cond_prob(prev, w, lm=lm_full):
    V = lm.vocab
    ids, cnt, _ = lm.followers([V.index.get(normalize(prev), V.unk)], 2)
    wid = V.index.get(normalize(w), -1)
    tot = cnt.sum()
    return float(cnt[ids == wid].sum() / tot) if tot else 0.0

p_pm = cond_prob("ጠቅላይ", "ሚኒስትር")
p_horn = cond_prob("ቀንድ", "ትገኛለች")
words = next(sentences("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።"))
sent = {n: (lm_full.sentence_logprob(words, n, "mle")[0] / np.log(10),
            lm_full.sentence_logprob(words, n, "kn")[0] / np.log(10)) for n in range(1, 5)}
display(pd.DataFrame(sent, index=["log10 P (MLE)", "log10 P (KN)"]).T)

rng = np.random.default_rng(7)
gen = {n: [" ".join(lm_full.generate(n, rng, max_len=30)) for _ in range(4)] for n in range(1, 5)}
for n in range(1, 5):
    print(f"{n}-gram sample:", gen[n][0])

if FULL:
    for n in range(1, 5):
        check("1.2", f"distinct {n}-grams", REPORTED["ngram_counts"][n - 1]["unique n-grams"], counts[n - 1])
        check("1.3", f"top-10 {n}-grams", [r[0] for r in REPORTED["top_ngrams"][str(n)]], [g for g, _ in top[n]])
    check("1.4", "P(ሚኒስትር | ጠቅላይ)", [r for r in REPORTED["conditional"] if r[0] == "ጠቅላይ"][0][4], p_pm)
    s = [r for r in REPORTED["sentence_probs"] if r[0].startswith("ኢትዮጵያ በአፍሪካ")]
    for n in range(1, 5):
        check("1.7", f"log10 P(sentence), KN, n={n}", s[n - 1][4], sent[n][1])
    check("1.8", "generated sentences (seed 7)", REPORTED["generated"], {str(k): v for k, v in gen.items()})
else:
    check("1.2", "more distinct n-grams as n grows", "increasing", counts, ok=counts == sorted(counts))
    check("1.2", "singleton share grows with n", "increasing", np.round(singles, 1).tolist(), ok=singles == sorted(singles))
    check("1.3", "top unigrams are function words", "ነው/ላይ/<NUM> in top 5", [g for g, _ in top[1][:5]],
          ok=len({"ነው", "ላይ", "<NUM>"} & {g for g, _ in top[1][:5]}) >= 2)
    check("1.4", "P(ሚኒስትር | ጠቅላይ) is high", "> 0.3", round(p_pm, 3), ok=p_pm > 0.3)
check("1.4", "P(ትገኛለች | ቀንድ) is zero (unseen)", 0.0, p_horn, ok=p_horn == 0.0)
check("1.7", "MLE gives P = 0 for n ≥ 2", "−∞", [round(sent[n][0], 2) for n in (2, 3, 4)],
      ok=all(np.isinf(sent[n][0]) for n in (2, 3, 4)))
check("1.7", "KN gives finite probability", "finite", [round(sent[n][1], 2) for n in range(1, 5)],
      ok=all(np.isfinite(sent[n][1]) for n in range(1, 5)))

del lm_full, full, word_counts
gc.collect()

# %% [markdown]
# ## 4. Task 2 — deduplicate, split, perplexity

# %%
DEDUP = DATA / (WORK.stem + "_dedup.txt")
seen = set()
with open(WORK, encoding="utf-8") as f, open(DEDUP, "w", encoding="utf-8") as out:
    for line in f:
        h = hash(line)
        if h not in seen:
            seen.add(h)
            out.write(line)
del seen
gc.collect()

counts_all = C.count_words(DEDUP)
vocab_all = Vocab.from_counter(counts_all, 1)
enc = C.encode(DEDUP, vocab_all)
train_raw, test_raw = C.split(*enc, test_frac=0.02, seed=42)
train_raw, dev_raw = C.split(*train_raw, test_frac=0.005, seed=1)
del enc, counts_all
gc.collect()

tr_counts = np.bincount(train_raw[0], minlength=len(vocab_all))
keep_words = [vocab_all.words[i] for i in np.flatnonzero(tr_counts >= 2) if i > 2]
vocab = Vocab([BOS, EOS, "<UNK>"] + sorted(keep_words, key=lambda w: -tr_counts[vocab_all.index[w]]))
remap = np.full(len(vocab_all), vocab.unk, np.int32)
remap[[vocab_all.index[w] for w in vocab.words[3:]]] = np.arange(3, len(vocab), dtype=np.int32)
remap[0], remap[1] = 0, 1
train = (remap[train_raw[0]], train_raw[1])
test = (remap[test_raw[0]], test_raw[1])
dev = (remap[dev_raw[0]], dev_raw[1])
del train_raw, test_raw, dev_raw, vocab_all, remap
gc.collect()

lm = NgramLM.build(train, vocab, order=4, kn=True)
dev_l, test_l = C.to_lists(dev), C.to_lists(test)
ks = [1.0, 0.1, 0.01, 0.001, 0.0005, 0.0001, 0.00001]
best_k = {n: min(ks, key=lambda k: lm.perplexity(dev_l, n, "addk", k)[0]) for n in range(1, 5)}

rows = []
for n in range(1, 5):
    _, zero = lm.perplexity(test_l, n, "mle")
    rows.append({"n": n, "MLE zero-prob %": 100 * zero,
                 "Laplace": lm.perplexity(test_l, n, "addk", 1.0)[0],
                 "add-k": lm.perplexity(test_l, n, "addk", best_k[n])[0],
                 "Kneser-Ney": lm.perplexity(test_l, n, "kn")[0]})
pp = pd.DataFrame(rows).set_index("n")
display(pp.round(1))

kn = pp["Kneser-Ney"].tolist()
lap = pp["Laplace"].tolist()
if FULL:
    check("2", "train tokens", REPORTED["split"]["train tokens"], int(len(train[0])))
    check("2", "test tokens", REPORTED["split"]["test tokens"], int(len(test[0])))
    for n in range(1, 5):
        r = REPORTED["perplexity"][n - 1]
        check("2", f"KN perplexity n={n}", r["Kneser-Ney"], kn[n - 1])
        check("2", f"Laplace perplexity n={n}", r["Laplace (add-1)"], lap[n - 1])
        check("2", f"MLE zero-prob % n={n}", r["MLE: % zero-prob tokens"], pp.loc[n, "MLE zero-prob %"])
check("2", "KN perplexity falls with every order", "decreasing", [round(x) for x in kn],
      ok=all(a > b for a, b in zip(kn, kn[1:])))
check("2", "Laplace gets worse as n grows (n ≥ 2)", "increasing", [round(x) for x in lap],
      ok=all(a < b for a, b in zip(lap[1:], lap[2:])))
check("2", "KN beats add-k and Laplace for n ≥ 2", "KN lowest",
      "yes" if all(pp.loc[n, "Kneser-Ney"] < min(pp.loc[n, "add-k"], pp.loc[n, "Laplace"]) for n in (2, 3, 4)) else "no",
      ok=all(pp.loc[n, "Kneser-Ney"] < min(pp.loc[n, "add-k"], pp.loc[n, "Laplace"]) for n in (2, 3, 4)))

# %% [markdown]
# ## 5. Task 3 — extrinsic evaluation

# %%
ext = {}
for n in range(1, 5):
    nwp = E.next_word_prediction(lm, test_l, n, 5000)
    ext[n] = {"top1": nwp["top1"], "top5": nwp["top5"], "mrr@10": nwp["mrr@10"],
              "KSS": E.keystroke_savings(lm, test_l, n, k=5, max_words=1500)["kss"],
              **{k: E.corruption_discrimination(lm, test_l, n, k, max_sents=3000)["accuracy"]
                 for k in ("swap", "replace", "shuffle")}}
ext = pd.DataFrame(ext).T
display((100 * ext).round(1))

if FULL:
    names = {"top1": "NWP top1", "top5": "NWP top5", "KSS": "KSS (top-5)", "swap": "discrim. swap",
             "replace": "discrim. replace", "shuffle": "discrim. shuffle"}
    for n in range(1, 5):
        for col, key in names.items():
            check("3", f"{col} n={n}", REPORTED["extrinsic"][n - 1][key], float(ext.loc[n, col]))
t5 = ext["top5"].tolist()
check("3", "top-5 accuracy rises with n", "increasing", [round(100 * x, 1) for x in t5],
      ok=all(a < b for a, b in zip(t5, t5[1:])))
check("3", "unigram is at chance on swap/shuffle", 0.5, [ext.loc[1, "swap"], ext.loc[1, "shuffle"]],
      ok=ext.loc[1, "swap"] == 0.5 and ext.loc[1, "shuffle"] == 0.5)
check("3", "bigram+ detects shuffled sentences", "> 95%", round(100 * ext.loc[3, "shuffle"], 1), ok=ext.loc[3, "shuffle"] > 0.95)

# %% [markdown]
# ## 6. Task 4 — next-word suggestions

# %%
def suggest(text, n=3, k=5):
    text = normalize(text)
    toks = tokenize(text)
    prefix = toks.pop() if text and not text[-1].isspace() and toks else ""
    return lm.topk_next(toks[-(n - 1):], n=n, k=k, prefix=prefix)

for q in ["ኢትዮጵያ ውስጥ የቡና ", "ኢትዮጵያ በአፍሪካ ", "አዲስ ", "ኢትዮጵያ ውስጥ የቡና ም"]:
    print(f"{q!r:24}", [(w, round(p, 4)) for w, p in suggest(q)])
coffee = [w for w, _ in suggest("ኢትዮጵያ ውስጥ የቡና ")]
if FULL:
    check("4", "suggestions for 'ኢትዮጵያ ውስጥ የቡና'", [w for w, _ in REPORTED["suggestions"]["ኢትዮጵያ ውስጥ የቡና "]], coffee)
check("4", "'አዲስ' → 'አበባ' first", "አበባ", suggest("አዲስ ")[0][0], ok=suggest("አዲስ ")[0][0] == "አበባ")

# %% [markdown]
# ## 7. Result

# %%
res = pd.DataFrame(CHECKS, columns=["task", "check", "reported / expected", "reproduced", "result"])
pd.set_option("display.max_colwidth", 70)
display(res)
n_fail = int((res["result"] == "FAIL").sum())
mode = "full corpus, exact comparison" if FULL else f"{int(SCALE * 100)}% of the corpus, qualitative checks"
display(Markdown(f"### {len(res) - n_fail} / {len(res)} checks passed ({mode}) — total time {(time.time() - T0) / 60:.1f} min"))
