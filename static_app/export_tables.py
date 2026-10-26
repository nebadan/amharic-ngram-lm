"""Export precomputed next-word suggestion tables for the static (browser) app.

For the most frequent one-word and two-word contexts we store the top-K next words
with their interpolated Kneser-Ney probabilities, plus a vocabulary with unigram
probabilities for word completion. The browser app looks up the longest known
context and backs off to shorter ones, like the full model.

Usage: python static_app/export_tables.py [model.npz]
"""
import gzip
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from amharic_lm.ngram import NgramLM  # noqa: E402

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "app" / "model" / "amharic_4gram_kn.npz"
OUT = ROOT / "static_app" / "space" / "tables.bin"
VOCAB_MIN = 10        # words offered for completion / suggestion
CTX_MIN = {2: 10, 3: 20}  # minimum context frequency to precompute
K = 8

LM = None
KEEP = None


def contexts(lm, k, min_count):
    T = lm.tables[k]
    sel = np.flatnonzero(T.ctx_raw >= min_count)
    first = T.by_ctx[T.ctx_start[sel]]
    return T.grams[first, :k - 1], T.ctx_raw[sel]


def score(args):
    ctx, n = args
    cand, p = LM.score_candidates(list(ctx), n, exclude_unk=True, allow_eos=True)
    ok = KEEP[cand]
    cand, p = cand[ok], p[ok]
    top = np.argsort(-p, kind="stable")[:K]
    return [(int(cand[i]), float(p[i])) for i in top]


def main():
    global LM, KEEP
    t0 = time.time()
    LM = NgramLM.load(MODEL)
    V = LM.vocab
    T1 = LM.tables[1]
    counts = np.zeros(len(V), np.int64)
    counts[T1.grams[:, 0]] = T1.count
    KEEP = counts >= VOCAB_MIN
    KEEP[[V.bos, V.unk]] = False
    KEEP[V.eos] = True
    # export vocabulary ordered by frequency; </s> is id 0
    ids = np.flatnonzero(KEEP & (np.arange(len(V)) != V.eos))
    ids = ids[np.argsort(-counts[ids], kind="stable")]
    ids = np.concatenate([[V.eos], ids])
    new_id = np.full(len(V), -1, np.int64)
    new_id[ids] = np.arange(len(ids))
    total = counts[ids[1:]].sum()
    out = {
        "meta": {"K": K, "vocab_min": VOCAB_MIN, "ctx_min": CTX_MIN, "source": MODEL.name},
        "vocab": [V.words[i] for i in ids],
        "uni": [round(float(counts[i] / total), 8) for i in ids],
    }
    out["uni"][0] = 0.0
    with mp.get_context("fork").Pool(8) as pool:
        for k in (2, 3):
            ctxs, _ = contexts(LM, k, CTX_MIN[k])
            jobs = [(tuple(int(x) for x in c), k) for c in ctxs]
            res = pool.map(score, jobs, chunksize=500)
            table = {}
            for c, r in zip(jobs, res):
                key = " ".join(V.words[w] for w in c[0])  # <s> kept literally
                table[key] = [[int(new_id[w]), round(p, 5)] for w, p in r]
            out[f"c{k}"] = table
            print(f"order {k}: {len(table):,} contexts ({time.time() - t0:.0f}s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode()
    OUT.write_bytes(gzip.compress(raw, 9))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB gz, {len(raw) / 1e6:.0f} MB raw) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
