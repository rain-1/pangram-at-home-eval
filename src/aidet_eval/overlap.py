"""Leakage/independence grouping and near-duplicate detection.

Match normalisation (lowercase, alphanumeric tokens) is used only for fingerprints; it never
changes the text that is evaluated.

* exact links: identical match-normalised text fingerprint;
* near-duplicates: MinHash (64 permutations, word 5-gram shingles, xxhash64) with LSH banding
  (16 bands x 4 rows); candidate pairs are verified by estimated Jaccard >= 0.8;
* native keys from adapters (topic/source/prompt/book/article/essay/session IDs).
Union-find over all of these yields leakage groups.
"""

from __future__ import annotations

import collections
import re

import numpy as np
import xxhash

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_PERM = 64
BANDS, ROWS = 16, 4
SHINGLE = 5
JACCARD_THRESHOLD = 0.8
_MERSENNE = np.uint64((1 << 61) - 1)
_rng = np.random.RandomState(42)
_A = _rng.randint(1, (1 << 61) - 1, size=NUM_PERM, dtype=np.uint64)
_B = _rng.randint(0, (1 << 61) - 1, size=NUM_PERM, dtype=np.uint64)

POLICY = {"normalisation": "lowercase; tokens = [a-z0-9]+ runs", "shingle_words": SHINGLE, "num_perm": NUM_PERM,
          "lsh_bands": BANDS, "lsh_rows": ROWS, "jaccard_threshold": JACCARD_THRESHOLD, "hash": "xxhash64 seed 0",
          "min_tokens_for_near_dup": 20, "min_tokens_for_exact_fingerprint_link": 8}


def match_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def fingerprint(text: str) -> str:
    return xxhash.xxh128_hexdigest(" ".join(match_tokens(text)).encode())


def minhash(tokens: list[str]) -> np.ndarray | None:
    if len(tokens) < POLICY["min_tokens_for_near_dup"]:
        return None
    sh = np.fromiter((xxhash.xxh64_intdigest(" ".join(tokens[i:i + SHINGLE]).encode()) for i in range(len(tokens) - SHINGLE + 1)),
                     dtype=np.uint64)
    sh = sh & _MERSENNE  # 61-bit domain
    # (a*x + b) mod p with uint64 wraparound is an acceptable universal-ish family for MinHash here
    vals = (np.outer(sh, _A) + _B) & _MERSENNE
    return vals.min(axis=0)


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.p.setdefault(x, x)
        while p != self.p.setdefault(p, p):
            self.p[p] = self.p[self.p[p]]
            p = self.p[p]
        root = p
        while x != root:
            nxt = self.p[x]
            self.p[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def near_duplicate_pairs(ids: list[str], texts: list[str], verification: str = "minhash_estimate") -> tuple[list[tuple[str, str, float]], dict]:
    """verification='minhash_estimate' (benchmark-v1 default, reproducible) or 'exact_jaccard'
    (recommended for later versions: candidate pairs re-scored on exact 5-gram shingle sets,
    removing MinHash estimator noise near the 0.8 threshold)."""
    sigs: dict[str, np.ndarray] = {}
    for i, t in zip(ids, texts):
        m = minhash(match_tokens(t))
        if m is not None:
            sigs[i] = m
    buckets: dict[tuple, list[str]] = collections.defaultdict(list)
    for i, s in sigs.items():
        for b in range(BANDS):
            buckets[(b, s[b * ROWS:(b + 1) * ROWS].tobytes())].append(i)
    cand = set()
    big = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > 500:  # degenerate bucket (boilerplate); verify all pairs against first member only
            big += 1
            for m in members[1:]:
                cand.add((members[0], m) if members[0] < m else (m, members[0]))
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                cand.add((a, b) if a < b else (b, a))
    pairs = []
    text_of = dict(zip(ids, texts)) if verification == "exact_jaccard" else {}
    shingles: dict[str, set] = {}

    def sh(i):
        if i not in shingles:
            t = match_tokens(text_of[i])
            shingles[i] = {" ".join(t[k:k + SHINGLE]) for k in range(len(t) - SHINGLE + 1)}
        return shingles[i]

    for a, b in cand:
        if verification == "exact_jaccard":
            j = len(sh(a) & sh(b)) / len(sh(a) | sh(b))
        else:
            j = float(np.mean(sigs[a] == sigs[b]))
        if j >= JACCARD_THRESHOLD:
            pairs.append((a, b, j))
    stats = {"n_signed": len(sigs), "n_unsigned_short": len(ids) - len(sigs), "lsh_candidate_pairs": len(cand),
             "verified_pairs": len(pairs), "degenerate_buckets_over_500": big, "verification": verification}
    return pairs, stats
