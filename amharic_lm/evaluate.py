"""Extrinsic evaluation tasks for the n-gram models."""
from __future__ import annotations

import time

import numpy as np


def next_word_prediction(lm, sents, n, max_positions=5000, seed=0, ks=(1, 3, 5, 10)):
    """Predict each held-out word from its history.
    Returns top-k accuracy and MRR@10 (unknown targets count as misses)."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i, s in enumerate(sents) for j in range(len(s) + 1)]
    sel = rng.choice(len(pairs), size=min(max_positions, len(pairs)), replace=False)
    hits = {k: 0 for k in ks}
    rr = 0.0
    t0 = time.time()
    for idx in sel:
        i, j = pairs[idx]
        s = sents[i]
        target = s[j] if j < len(s) else lm.vocab.eos
        if target == lm.vocab.unk:
            continue  # an <UNK> can never be a useful suggestion
        cand, p = lm.score_candidates(s[:j], n, extra=[target])
        pt = p[cand == target][0]
        rank = 1 + int((p > pt).sum())
        for k in ks:
            hits[k] += rank <= k
        if rank <= 10:
            rr += 1.0 / rank
    m = len(sel)
    out = {f"top{k}": hits[k] / m for k in ks}
    out["mrr@10"] = rr / m
    out["positions"] = int(m)
    out["ms_per_query"] = 1000 * (time.time() - t0) / m
    return out


def keystroke_savings(lm, sents, n, k=5, max_words=1500, seed=0):
    """Simulate typing with a top-k suggestion list. The user types characters
    until the intended word shows up in the list, then picks it (1 keystroke).
    Spaces are inserted automatically on selection.
    KSS = 1 - keystrokes_with_model / keystrokes_without."""
    rng = np.random.default_rng(seed)
    V = lm.vocab
    order = rng.permutation(len(sents))
    typed = baseline = words = 0
    for i in order:
        s = sents[i]
        for j, w in enumerate(s):
            if w == V.unk:
                continue
            word = V.words[w]
            baseline += len(word) + 1          # characters + space
            cost = None
            for c in range(len(word)):        # c characters typed so far
                sugg = [x for x, _ in lm.topk_next([V.words[t] for t in s[max(0, j - n):j]],
                                                   n=n, k=k, prefix=word[:c], allow_eos=False)]
                if word in sugg:
                    cost = c + 1               # typed c chars + 1 selection
                    break
            typed += cost if cost is not None else len(word) + 1
            words += 1
            if words >= max_words:
                break
        if words >= max_words:
            break
    return {"kss": 1 - typed / baseline, "keystrokes": typed, "baseline": baseline,
            "words": words}


def corruption_discrimination(lm, sents, n, kind="swap", seed=0, max_sents=3000):
    """Real sentence vs. corrupted copy: the model 'wins' when it gives the real
    sentence a higher probability. kind = swap (two adjacent words) or
    shuffle (full random order) or replace (one word -> random frequent word)."""
    rng = np.random.default_rng(seed)
    pool = [s for s in sents if len(s) >= 4][:max_sents]
    corrupt = []
    top = lm._top_unigrams(2000)
    top = top[top > 2]
    for s in pool:
        c = list(s)
        if kind == "swap":
            j = rng.integers(0, len(c) - 1)
            while c[j] == c[j + 1] and len(set(c)) > 1:
                j = rng.integers(0, len(c) - 1)
            c[j], c[j + 1] = c[j + 1], c[j]
        elif kind == "shuffle":
            while c == s and len(set(c)) > 1:
                rng.shuffle(c)
        elif kind == "replace":
            j = rng.integers(0, len(c))
            c[j] = int(rng.choice(top[top != c[j]]))
        corrupt.append(c)
    a = _sent_logprobs(lm, pool, n)
    b = _sent_logprobs(lm, corrupt, n)
    same = np.isclose(a, b)
    acc = float(((a > b) & ~same).mean() + 0.5 * same.mean())
    return {"accuracy": acc, "pairs": len(pool), "ties": float(same.mean())}


def _sent_logprobs(lm, sents, n):
    p, _ = lm.stream_probs(sents, n, "kn")
    lens = np.array([len(s) + 1 for s in sents])
    sid = np.repeat(np.arange(len(sents)), lens)
    return np.bincount(sid, weights=np.log(p), minlength=len(sents))
