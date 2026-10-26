"""Memory-efficient n-gram language model (orders 1..N) built on numpy.

Every n-gram is stored as a 64-bit hash of its word ids, in sorted arrays, so
lookups are binary searches and tens of millions of 4-grams fit in RAM.
Supports MLE, add-k (Laplace) and interpolated Kneser-Ney probabilities.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"
_M = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)


def _mix(x):
    x = x ^ (x >> np.uint64(30))
    x = x * _C1
    x = x ^ (x >> np.uint64(27))
    x = x * _C2
    return x ^ (x >> np.uint64(31))


def hash_cols(cols):
    """Hash a sequence of equal-length id columns (one n-gram per row)."""
    n = len(cols)
    size = len(cols[0]) if n else 1
    h = np.full(size, np.uint64(n + 1), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in cols:
            h = _mix(h * _M + np.asarray(c).astype(np.uint64) + np.uint64(1))
    return h


def _lookup(keys, q):
    """Index of each q in sorted keys, or -1."""
    i = np.searchsorted(keys, q)
    i = np.minimum(i, len(keys) - 1) if len(keys) else i
    ok = (keys[i] == q) if len(keys) else np.zeros(len(q), bool)
    return np.where(ok, i, -1)


class Vocab:
    def __init__(self, words):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.bos, self.eos, self.unk = self.index[BOS], self.index[EOS], self.index[UNK]

    @classmethod
    def from_counter(cls, counter: Counter, min_count=1):
        words = [BOS, EOS, UNK] + sorted(
            (w for w, c in counter.items() if c >= min_count and w not in (BOS, EOS, UNK)),
            key=lambda w: (-counter[w], w))
        return cls(words)

    def __len__(self):
        return len(self.words)

    def encode(self, toks):
        u = self.unk
        return [self.index.get(t, u) for t in toks]


def make_stream(sents, order):
    """Concatenate sentences as <s>*(order-1) w1..wn </s>, as one int32 array.
    `sents` is a list of id lists or a (flat_ids, lengths) pair."""
    if isinstance(sents, tuple):
        flat, lens = np.asarray(sents[0], np.int32), np.asarray(sents[1], np.int64)
    else:
        lens = np.array([len(s) for s in sents], np.int64)
        flat = np.fromiter((w for s in sents for w in s), np.int32, int(lens.sum()))
    pad = order - 1
    span = lens + pad + 1
    out_start = np.concatenate([[0], np.cumsum(span)[:-1]])
    in_start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = np.zeros(int(span.sum()), dtype=np.int32)  # 0 = <s>
    pos = np.repeat(out_start + pad - in_start, lens) + np.arange(len(flat))
    out[pos] = flat
    out[out_start + pad + lens] = 1  # </s>
    return out


def valid_starts(stream, k, bos=0, eos=1):
    """Start positions of k-token windows that stay inside one sentence."""
    L = len(stream) - k + 1
    if L <= 0:
        return np.zeros(0, np.int64)
    isb = np.concatenate([[0], np.cumsum(stream == bos, dtype=np.int32)])
    ise = np.concatenate([[0], np.cumsum(stream == eos, dtype=np.int32)])
    st = np.arange(L)
    n_bos = isb[st + k] - isb[st]
    n_eos_inner = ise[st + k - 1] - ise[st]  # EOS allowed only at the last slot
    return st[(n_bos <= k - 1) & (n_eos_inner == 0)]


@dataclass
class OrderTable:
    k: int
    keys: np.ndarray            # sorted uint64 hashes of k-grams
    grams: np.ndarray           # (M, k) int32 word ids
    count: np.ndarray           # raw counts
    cont: np.ndarray | None = None   # KN continuation counts (raw for <s>-initial)
    # context (first k-1 words) statistics, sorted by ctx key
    ctx_keys: np.ndarray | None = None
    ctx_raw: np.ndarray | None = None
    ctx_cont: np.ndarray | None = None
    ctx_types: np.ndarray | None = None
    ctx_start: np.ndarray | None = None  # offset into by_ctx
    by_ctx: np.ndarray | None = None     # entry indices grouped by context
    D_raw: float = 0.75
    D_cont: float = 0.75


def _discount(counts):
    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    if n1 == 0 or n2 == 0:
        return 0.75
    return n1 / (n1 + 2 * n2)


class NgramLM:
    def __init__(self, vocab: Vocab, order: int):
        self.vocab = vocab
        self.order = order
        self.tables: dict[int, OrderTable] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, sent_ids, vocab: Vocab, order=4, kn=True, verbose=True):
        lm = cls(vocab, order)
        stream = make_stream(sent_ids, order)
        for k in range(1, order + 1):
            st = valid_starts(stream, k)
            h = hash_cols([stream[st + j] for j in range(k)])
            keys, first, cnt = np.unique(h, return_index=True, return_counts=True)
            grams = np.stack([stream[st[first] + j] for j in range(k)], axis=1).astype(np.int32)
            del h, st
            lm.tables[k] = OrderTable(k, keys, grams, cnt.astype(np.int64))
            if verbose:
                print(f"  order {k}: {len(keys):,} unique / {int(cnt.sum()):,} total")
        del stream
        if kn:
            lm._prepare_kn()
        else:
            for k in range(2, order + 1):
                lm._context_index(k)
        return lm

    def _context_index(self, k):
        T = self.tables[k]
        ctx = hash_cols([T.grams[:, j] for j in range(k - 1)])
        perm = np.argsort(ctx, kind="stable")
        cs = ctx[perm]
        ukeys, start, types = np.unique(cs, return_index=True, return_counts=True)
        T.ctx_keys, T.ctx_start, T.ctx_types, T.by_ctx = ukeys, start, types, perm
        T.ctx_raw = np.add.reduceat(T.count[perm], start) if len(start) else start
        if T.cont is not None:
            T.ctx_cont = np.add.reduceat(T.cont[perm], start) if len(start) else start
        return ctx

    def _prepare_kn(self):
        N = self.order
        # continuation counts: number of distinct left extensions
        for k in range(1, N):
            T, U = self.tables[k], self.tables[k + 1]
            suf = hash_cols([U.grams[:, j] for j in range(1, k + 1)])
            sk, sc = np.unique(suf, return_counts=True)
            idx = _lookup(sk, T.keys)
            cont = np.where(idx >= 0, sc[np.maximum(idx, 0)], 0).astype(np.int64)
            starts_bos = T.grams[:, 0] == self.vocab.bos
            T.cont = np.where(starts_bos | (cont == 0), T.count, cont)
            T.D_cont = _discount(T.cont)
        for k in range(1, N + 1):
            T = self.tables[k]
            T.D_raw = _discount(T.count)
            if k >= 2:
                self._context_index(k)
        T1 = self.tables[1]
        V = len(self.vocab) - 1  # exclude <s>
        self._p1 = {}
        for flavor in ("raw", "cont"):
            c = T1.count if flavor == "raw" or T1.cont is None else T1.cont
            D = T1.D_raw if flavor == "raw" or T1.cont is None else T1.D_cont
            S = c.sum()
            p = np.full(len(self.vocab), D * len(c) / S / V)
            p[T1.grams[:, 0]] += np.maximum(c - D, 0) / S
            p[self.vocab.bos] = 0.0
            self._p1[flavor] = p

    # ----------------------------------------------------------- helpers
    def count(self, words):
        """Raw count of an n-gram given as a list of words."""
        k = len(words)
        ids = self.vocab.encode(words)
        T = self.tables[k]
        i = _lookup(T.keys, hash_cols([[x] for x in ids]))[0]
        return int(T.count[i]) if i >= 0 else 0

    def followers(self, context_ids, k=None):
        """(word_ids, counts) of words seen after the given context."""
        k = (len(context_ids) + 1) if k is None else k
        T = self.tables[k]
        h = hash_cols([[x] for x in context_ids[-(k - 1):]])
        ci = _lookup(T.ctx_keys, h)[0]
        if ci < 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int64), None
        s = T.ctx_start[ci]
        kept = getattr(T, "_kept", None)
        e = s + (kept[ci] if kept is not None else T.ctx_types[ci])
        ent = T.by_ctx[s:e]
        return T.grams[ent, -1], T.count[ent], ci

    def top_ngrams(self, k, n=10, skip=()):
        T = self.tables[k]
        order = np.argsort(-T.count, kind="stable")
        total = int(T.count.sum())
        out = []
        skip_ids = {self.vocab.index[s] for s in skip if s in self.vocab.index}
        for i in order:
            g = T.grams[i]
            if skip_ids and any(int(x) in skip_ids for x in g):
                continue
            out.append((tuple(self.vocab.words[x] for x in g), int(T.count[i]), T.count[i] / total))
            if len(out) == n:
                break
        return out

    def frequencies(self, k, n=200, skip=()):
        return {" ".join(g): c for g, c, _ in self.top_ngrams(k, n, skip)}

    # ------------------------------------------------------ probabilities
    def _kn_probs(self, ctx_cols, w, n):
        """KN P(w | context) for arrays. ctx_cols[j] holds the j-th word before w
        (ctx_cols[0] = word right before w)."""
        p = self._p1["raw" if n == 1 else "cont"][w]
        for k in range(2, n + 1):
            highest = k == n
            T = self.tables[k]
            hist = [ctx_cols[j] for j in range(k - 2, -1, -1)]
            hc = hash_cols(hist)
            hk = hash_cols(hist + [w])
            ci = _lookup(T.ctx_keys, hc)
            gi = _lookup(T.keys, hk)
            cvec = T.count if highest else T.cont
            c_hw = np.where(gi >= 0, cvec[np.maximum(gi, 0)], 0)
            c_h = (T.ctx_raw if highest else T.ctx_cont)[np.maximum(ci, 0)]
            ty = T.ctx_types[np.maximum(ci, 0)]
            D = T.D_raw if highest else T.D_cont
            pk = (np.maximum(c_hw - D, 0) + D * ty * p) / np.maximum(c_h, 1)
            p = np.where(ci >= 0, pk, p)
        return p

    def _addk_probs(self, ctx_cols, w, n, a):
        V = len(self.vocab) - 1
        T = self.tables[n]
        if n == 1:
            gi = _lookup(T.keys, hash_cols([w]))
            c = np.where(gi >= 0, T.count[np.maximum(gi, 0)], 0)
            return (c + a) / (T.count.sum() + a * V)
        hist = [ctx_cols[j] for j in range(n - 2, -1, -1)]
        ci = _lookup(T.ctx_keys, hash_cols(hist))
        gi = _lookup(T.keys, hash_cols(hist + [w]))
        c_hw = np.where(gi >= 0, T.count[np.maximum(gi, 0)], 0)
        c_h = np.where(ci >= 0, T.ctx_raw[np.maximum(ci, 0)], 0)
        return (c_hw + a) / (c_h + a * V)

    def _mle_probs(self, ctx_cols, w, n):
        T = self.tables[n]
        if n == 1:
            gi = _lookup(T.keys, hash_cols([w]))
            c = np.where(gi >= 0, T.count[np.maximum(gi, 0)], 0)
            return c / T.count.sum()
        hist = [ctx_cols[j] for j in range(n - 2, -1, -1)]
        ci = _lookup(T.ctx_keys, hash_cols(hist))
        gi = _lookup(T.keys, hash_cols(hist + [w]))
        c_hw = np.where(gi >= 0, T.count[np.maximum(gi, 0)], 0)
        c_h = np.where(ci >= 0, T.ctx_raw[np.maximum(ci, 0)], 0)
        return np.where(c_h > 0, c_hw / np.maximum(c_h, 1), 0.0)

    def probs(self, ctx_cols, w, n, method="kn", k=1.0):
        if method == "kn":
            return self._kn_probs(ctx_cols, w, n)
        if method == "addk":
            return self._addk_probs(ctx_cols, w, n, k)
        if method == "mle":
            return self._mle_probs(ctx_cols, w, n)
        raise ValueError(method)

    def stream_probs(self, sent_ids, n, method="kn", k=1.0):
        """Probability of every predicted token (words + </s>) in sentences."""
        stream = make_stream(sent_ids, self.order)
        pos = np.flatnonzero(stream != self.vocab.bos)
        ctx = [stream[pos - j - 1] for j in range(max(n - 1, 0))]
        return self.probs(ctx, stream[pos], n, method, k), stream[pos]

    def perplexity(self, sent_ids, n, method="kn", k=1.0):
        p, _ = self.stream_probs(sent_ids, n, method, k)
        zero = float((p <= 0).mean())
        if zero > 0:
            return float("inf"), zero
        return float(np.exp(-np.log(p).mean())), 0.0

    def sentence_logprob(self, words, n, method="kn", k=1.0):
        ids = self.vocab.encode(words)
        p, tgt = self.stream_probs([ids], n, method, k)
        with np.errstate(divide="ignore"):
            lp = np.log(p)
        return float(lp.sum()), p, [self.vocab.words[t] for t in tgt]

    # ------------------------------------------------------ prediction
    def topk_next(self, context_words, n=None, k=5, prefix="", exclude_unk=True,
                  allow_eos=True, cand_pool=300):
        """Top-k next words with KN probabilities given preceding words.
        `prefix` restricts to words starting with a partially typed string."""
        n = n or self.order
        cand, p = self.score_candidates(self.vocab.encode(context_words), n, prefix,
                                        exclude_unk, allow_eos, cand_pool)
        top = np.argsort(-p, kind="stable")[:k]
        return [(self.vocab.words[cand[i]], float(p[i])) for i in top]

    def score_candidates(self, context_ids, n, prefix="", exclude_unk=True,
                         allow_eos=True, cand_pool=300, extra=None):
        """Candidate next-word ids and their KN probabilities. Exact for the top
        of the ranking: any word outside the candidate set scores proportional
        to its unigram probability, so the top unigrams cover it."""
        V = self.vocab
        ctx = [V.bos] * (n - 1) + list(context_ids)
        ctx = ctx[-(n - 1):] if n > 1 else []
        if prefix:
            cand = self._prefix_ids(prefix)
        else:
            parts = [self._top_unigrams(cand_pool)]
            for kk in range(2, n + 1):
                f, _, _ = self.followers(ctx, kk) if len(ctx) >= kk - 1 else (np.zeros(0, np.int32), 0, 0)
                parts.append(f)
            if extra is not None:
                parts.append(np.asarray(extra, np.int32))
            cand = np.unique(np.concatenate(parts).astype(np.int32))
        drop = [V.bos] + ([V.unk] if exclude_unk else []) + ([] if allow_eos else [V.eos])
        cand = cand[~np.isin(cand, drop)].astype(np.int32)
        if len(cand) == 0:
            return cand, np.zeros(0)
        ctx_cols = [np.full(len(cand), ctx[-1 - j], np.int32) for j in range(n - 1)]
        return cand, self._kn_probs(ctx_cols, cand, n)

    def _top_unigrams(self, m):
        if not hasattr(self, "_topuni") or len(self._topuni) < m:
            self._topuni = np.argsort(-self._p1["cont" if self.order > 1 else "raw"])[:m]
        return self._topuni[:m]

    def _prefix_ids(self, prefix):
        if not hasattr(self, "_sorted_words"):
            order = np.argsort(np.array(self.vocab.words, dtype=object).astype(str))
            self._sorted_ids = order
            self._sorted_words = np.array(self.vocab.words, dtype=object)[order].astype(str)
        lo = np.searchsorted(self._sorted_words, prefix, "left")
        hi = np.searchsorted(self._sorted_words, prefix + "\U0010FFFF", "left")
        return self._sorted_ids[lo:hi]

    # ------------------------------------------------------ generation
    def generate(self, n, rng, max_len=30, temperature=1.0):
        """Sample a sentence from the MLE n-gram distribution (Shannon method)."""
        V = self.vocab
        ctx = [V.bos] * (n - 1)
        out = []
        T1 = self.tables[1]
        while len(out) < max_len:
            if n == 1:
                ids, cnt = T1.grams[:, 0], T1.count
            else:
                ids, cnt, ci = self.followers(ctx[-(n - 1):], n)
                if ci is None:
                    break
            p = cnt.astype(float) ** (1.0 / temperature)
            w = int(rng.choice(ids, p=p / p.sum()))
            if w == V.eos:
                break
            out.append(V.words[w])
            ctx.append(w)
        return out

    # ------------------------------------------------------ persistence
    def save(self, path, min_count=None):
        """Save to .npz. min_count prunes n-grams (order>=2) seen fewer times;
        context statistics are kept exact so probabilities stay normalized."""
        arrs = {}
        for k, T in self.tables.items():
            keep = np.ones(len(T.keys), bool)
            if min_count and k >= 2:
                keep = T.count >= min_count
                if T.cont is not None and k < self.order:
                    keep |= T.cont >= min_count
            for name in ("keys", "grams", "count", "cont"):
                a = getattr(T, name)
                if a is not None:
                    a = a[keep]
                    arrs[f"{k}_{name}"] = a.astype(np.int32) if min_count and a.dtype == np.int64 else a
            ckeep = None
            if min_count and k >= 2:
                # drop statistics of contexts with no surviving follower; such
                # contexts then back off to the lower order at query time
                kept_ctx = hash_cols([T.grams[keep, j] for j in range(k - 1)])
                ckeep = np.isin(T.ctx_keys, kept_ctx)
            for name in ("ctx_keys", "ctx_raw", "ctx_cont", "ctx_types"):
                a = getattr(T, name)
                if a is not None:
                    a = a if ckeep is None else a[ckeep]
                    arrs[f"{k}_{name}"] = a.astype(np.int32) if min_count and a.dtype == np.int64 else a
            arrs[f"{k}_D"] = np.array([T.D_raw, T.D_cont])
        for f, p in getattr(self, "_p1", {}).items():
            arrs[f"p1_{f}"] = p.astype(np.float32 if min_count else np.float64)
        arrs["meta"] = np.frombuffer(json.dumps({"order": self.order, "words": self.vocab.words},
                                                ensure_ascii=False).encode(), dtype=np.uint8)
        np.savez_compressed(path, **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        meta = json.loads(bytes(z["meta"]).decode())
        lm = cls(Vocab(meta["words"]), meta["order"])
        for k in range(1, lm.order + 1):
            g = lambda n: z[f"{k}_{n}"] if f"{k}_{n}" in z else None
            T = OrderTable(k, g("keys"), g("grams"), g("count"), g("cont"))
            T.ctx_keys, T.ctx_raw, T.ctx_cont = g("ctx_keys"), g("ctx_raw"), g("ctx_cont")
            T.D_raw, T.D_cont = (float(x) for x in z[f"{k}_D"])
            if k >= 2:
                # rebuild follower index from (possibly pruned) entries; keep the
                # saved exact context totals / type counts for normalization
                ctx = hash_cols([T.grams[:, j] for j in range(k - 1)])
                perm = np.argsort(ctx, kind="stable")
                cs = ctx[perm]
                lo = np.searchsorted(cs, T.ctx_keys, "left")
                hi = np.searchsorted(cs, T.ctx_keys, "right")
                T.by_ctx, T.ctx_start = perm, lo
                T._kept = hi - lo
                T.ctx_types = z[f"{k}_ctx_types"]
            lm.tables[k] = T
        lm._p1 = {f: z[f"p1_{f}"].astype(np.float64) for f in ("raw", "cont") if f"p1_{f}" in z}
        return lm
