"""Corpus loading: clean the raw GPAC file once, then encode to compact id arrays."""
from __future__ import annotations

import os
from array import array
from collections import Counter

import numpy as np

from .text import sentences, STOPWORDS


def preprocess_file(raw_path, out_path, chunk_chars=4_000_000):
    """Normalize + sentence-split + tokenize the raw corpus.
    Reads fixed-size chunks (the GPAC file is a single 1 GB line) and cuts each
    chunk after its last sentence end so no sentence is split in two.
    Writes one tokenized sentence per line. Returns raw-text statistics."""
    stats = Counter()
    with open(raw_path, encoding="utf-8", errors="ignore") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        carry = ""
        while True:
            block = f.read(chunk_chars)
            if not block:
                break
            stats["raw_chars"] += len(block)
            stats["raw_ws_tokens"] += len(block.split())
            text = carry + block
            cut = max(text.rfind("።"), text.rfind("፡፡"), text.rfind("?"), text.rfind("\n"))
            if cut < 0:
                cut = text.rfind(" ")
            carry = text[cut + 1:]
            _flush(text[:cut + 1], out, stats)
        if carry:
            _flush(carry, out, stats)
    return dict(stats)


def _flush(text, out, stats):
    for s in sentences(text):
        out.write(" ".join(s))
        out.write("\n")
        stats["sentences"] += 1
        stats["tokens"] += len(s)


def iter_clean(path, remove_stopwords=False):
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if remove_stopwords:
                toks = [t for t in toks if t not in STOPWORDS]
            if toks:
                yield toks


def count_words(path, remove_stopwords=False):
    c = Counter()
    for toks in iter_clean(path, remove_stopwords):
        c.update(toks)
    return c


def encode(path, vocab, remove_stopwords=False):
    """Return (flat int32 ids, int32 sentence lengths)."""
    idx, u = vocab.index, vocab.unk
    ids, lens = array("i"), array("i")
    for toks in iter_clean(path, remove_stopwords):
        ids.extend([idx.get(t, u) for t in toks])
        lens.append(len(toks))
    return np.frombuffer(ids, dtype=np.int32).copy(), np.frombuffer(lens, dtype=np.int32).copy()


def split(flat, lens, test_frac=0.02, seed=42):
    """Random sentence-level train/test split of an encoded corpus."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(lens)) < test_frac
    offs = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)])
    sent_of_tok = np.repeat(np.arange(len(lens)), lens)
    tok_test = is_test[sent_of_tok]
    return (flat[~tok_test], lens[~is_test]), (flat[tok_test], lens[is_test])


def subset(corpus, frac, seed=0):
    flat, lens = corpus
    rng = np.random.default_rng(seed)
    keep = rng.random(len(lens)) < frac
    sent_of_tok = np.repeat(np.arange(len(lens)), lens)
    return flat[keep[sent_of_tok]], lens[keep]


def to_lists(corpus):
    flat, lens = corpus
    offs = np.concatenate([[0], np.cumsum(lens, dtype=np.int64)])
    return [flat[offs[i]:offs[i + 1]].tolist() for i in range(len(lens))]
