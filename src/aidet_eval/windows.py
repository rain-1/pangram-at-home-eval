"""Frozen canonical window manifest for protocol shared_windows_v1.

Windows are built on canonical text only (never using gold labels), are shared by every model,
and are sized so that after each frozen reference tokenizer's preprocessing + special tokens,
every window fits all budgets. No hidden truncation is permitted at inference time.

Algorithm (deterministic):
  words = maximal \\S+ runs. Start at word s with n = 192. While the window does not fit every
  budget, n = max(1, floor(0.8 n)). A single word that still does not fit is split into
  consecutive code-point chunks (binary search for the longest fitting prefix; forward progress
  guaranteed). Next start = s + max(1, n - round(n * 64/192)). Stop once the last word is covered.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

from aidet_eval import paths
from aidet_eval.models.editlens_preprocess import PREPROCESS_VERSION, UPSTREAM
from aidet_eval.util import fingerprint, log_event, sha256_text, utcnow, write_json

PROTOCOL_ID = "shared_windows_v1"
TARGET_WORDS = 192
OVERLAP_WORDS = 64
WORD_RE = re.compile(r"\S+")


class Budget:
    """Token counting for every frozen reference tokenizer (tokenizer + preprocessing only)."""

    def __init__(self, model_specs: list[dict]):
        from aidet_eval.models.base import load_detector

        self.dets = [load_detector(s) for s in model_specs]
        for d in self.dets:
            d.load_tokenizer()
        self.budgets = {d.model_id: d.max_length for d in self.dets}

    def counts(self, texts: list[str]) -> dict[str, list[int]]:
        return {d.model_id: d.count_tokens(texts) for d in self.dets}

    def fits(self, texts: list[str]) -> list[bool]:
        c = self.counts(texts)
        return [all(c[m][i] <= self.budgets[m] for m in c) for i in range(len(texts))]


def _split_word(text: str, a: int, b: int, budget: Budget) -> list[tuple[int, int]]:
    out, s = [], a
    while s < b:
        lo, hi = s + 1, b
        if not budget.fits([text[s:lo]])[0]:
            raise RuntimeError("a single code point exceeds the token budget")
        while lo < hi:  # longest fitting prefix
            mid = (lo + hi + 1) // 2
            if budget.fits([text[s:mid]])[0]:
                lo = mid
            else:
                hi = mid - 1
        out.append((s, lo))
        s = lo
    return out


def make_windows(text: str, budget: Budget) -> list[dict]:
    words = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    W = len(words)
    wins: list[dict] = []
    s = 0
    while s < W:
        n = min(TARGET_WORDS, W - s)
        while True:
            a, b = words[s][0], words[s + n - 1][1]
            if budget.fits([text[a:b]])[0]:
                wins.append({"start": a, "end": b, "n_words": n, "kind": "words", "first_word": s})
                break
            if n == 1:
                for ca, cb in _split_word(text, a, b, budget):
                    wins.append({"start": ca, "end": cb, "n_words": 1, "kind": "codepoint_split", "first_word": s})
                break
            n = max(1, int(n * 0.8))
        if s + n >= W:
            break
        ov = round(n * OVERLAP_WORDS / TARGET_WORDS)
        s = s + max(1, n - ov)
    return wins


def protocol_definitions(model_specs: list[dict], window_hash: str | None) -> list[dict]:
    pre = {"version": PREPROCESS_VERSION, "upstream": UPSTREAM}
    tok = {s["id"]: {"tokenizer": s["tokenizer"], "revision": s["base_revision"], "max_length": s["max_length"]} for s in model_specs}
    shared = {
        "protocol_id": PROTOCOL_ID, "version": "1.0.0", "primary": True, "implementation_kind": "harness_wrapper",
        "window_construction": {"unit": "whitespace word (\\S+)", "target_words": TARGET_WORDS, "nominal_overlap_words": OVERLAP_WORDS,
                                "shrink_factor": 0.8, "overlong_word_fallback": "explicit code-point split",
                                "gold_independent": True},
        "special_token_budgeting": "each window, after reference preprocessing, tokenised with special tokens by every frozen reference tokenizer must be <= that model's max_length",
        "budgets": tok, "preprocessing": pre, "preprocessing_applied_to": "each window",
        "truncation": "prohibited (over-budget input -> error)", "document_aggregation": "max(window_score)",
        "aggregation_note": "harness wrapper; not upstream Pangram document behaviour; predeclared before scoring",
        "localisation_mode": "unsupported (window scores are contextual diagnostics only)",
        "window_manifest_sha256": window_hash,
    }
    native = {
        "protocol_id": "native_truncate_v1", "version": "1.0.0", "primary": False, "implementation_kind": "upstream_native_reference",
        "preprocessing": pre, "preprocessing_applied_to": "entire document",
        "truncation": "tokenizer truncation=True, max_length per model (right side), as upstream inference.py",
        "budgets": tok, "document_aggregation": "single input", "coverage_accounting": "model-visible canonical intervals via exact preprocessing map + tokenizer offsets",
        "localisation_mode": "unsupported",
    }
    for p in (shared, native):
        p["fingerprint"] = fingerprint({k: v for k, v in p.items() if k != "fingerprint"})
    return [shared, native]


def build_window_manifest(bench: str) -> Path:
    d = Path(bench)
    if (d / "windows" / "FROZEN").exists():
        print(f"window manifest already frozen in {d/'windows'}")
        return d / "windows"
    specs = yaml.safe_load(open(paths.CONFIGS / "models.yaml"))["models"]
    budget = Budget(specs)
    ex = pd.read_parquet(d / "examples.parquet", columns=["example_id", "text", "text_sha256"])
    rows = []
    for i, r in enumerate(ex.itertuples(index=False)):
        for k, w in enumerate(make_windows(r.text, budget)):
            rows.append({"window_id": f"{r.example_id}#w{k:03d}", "example_id": r.example_id, "window_index": k, **w,
                         "window_text_sha256": sha256_text(r.text[w["start"]:w["end"]]),
                         "example_text_sha256": r.text_sha256})
        if i % 2000 == 0:
            log_event("windows_progress", done=i, total=len(ex))
    wdf = pd.DataFrame(rows)
    texts = {r.example_id: r.text for r in ex.itertuples(index=False)}
    counts = budget.counts([texts[e][a:b] for e, a, b in zip(wdf["example_id"], wdf["start"], wdf["end"])])
    for m, c in counts.items():
        wdf[f"tokens__{m}"] = c
        assert (wdf[f"tokens__{m}"] <= budget.budgets[m]).all()
    (d / "windows").mkdir(exist_ok=True)
    wdf.to_parquet(d / "windows" / "shared_windows_v1.parquet", index=False)
    whash = sha256_text(json.dumps(wdf[["window_id", "start", "end", "window_text_sha256"]].values.tolist()))
    no_win = sorted(set(ex["example_id"]) - set(wdf["example_id"]))
    protos = protocol_definitions(specs, whash)
    write_json(d / "windows" / "protocols.json", {"created_at": utcnow(), "protocols": protos,
                                                   "n_windows": len(wdf), "n_examples": len(ex),
                                                   "examples_without_windows": no_win,
                                                   "windows_per_example": wdf.groupby("example_id").size().describe().to_dict()})
    (d / "windows" / "FROZEN").write_text(json.dumps({"window_manifest_sha256": whash, "frozen_at": utcnow()}))
    log_event("windows_done", n_windows=len(wdf), n_examples=len(ex), sha=whash[:16])
    return d / "windows"
