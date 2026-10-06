"""Synthetic fixtures. Everything here is data_origin=synthetic_fixture with 'fixture-' IDs and
must never be mistaken for baseline results. No corpora downloads or GPU are required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aidet_eval import paths
from aidet_eval.adapters import records_to_frame
from aidet_eval.adapters.common import Rec
from aidet_eval.util import sha256_text, write_json

TRICKY = "Emoji 👩‍👩‍👧 and combining é then <script>alert(1)</script> & “quotes”."


def _rec(i, text, authorship, role, panels, domain="news", spans=None, group=None, mech=None):
    r = Rec(dataset_key="fixture", dataset_id="fixture/synthetic", revision="fixture", original_row_id=str(i), text=text,
            original_split="fixture", domain=domain, subdomain="fixture", source_group_id=f"fixture-src-{group or i}",
            leakage_keys=[f"fixture-src-{group or i}"], independence_key=f"fixture-ind-{group or i}")
    r.authorship = authorship
    r.ai_involvement = {"human": "no", "ai": "yes", "mixed": "yes"}[authorship]
    r.mix_mechanism = mech
    r.ai_char_spans = spans if spans is not None else ([] if authorship == "human" else [[0, len(text)]])
    r.annotation_completeness = "complete"
    r.provenance_tier = "independently_audited" if authorship == "human" else None
    row = r.to_row()
    row["example_id"] = f"fixture-{i:03d}"
    row["role"] = role
    row["panel_ids"] = json.dumps(panels)
    return row


@pytest.fixture()
def fixture_run(tmp_path, monkeypatch):
    """A tiny frozen benchmark + run with synthetic scores, exported through the real pipeline."""
    for name in ("ARTIFACTS", "RUNS", "INVENTORY"):
        pass
    monkeypatch.setattr(paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "artifacts" / "runs")
    monkeypatch.setattr(paths, "INVENTORY", tmp_path / "inventory")
    rng = np.random.default_rng(0)
    rows = []
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    i = 0
    for role, n in (("calibration", 40), ("evaluation", 30)):
        for _ in range(n):
            t = " ".join(rng.choice(words, 30)) + (" " + TRICKY if i % 7 == 0 else "")
            rows.append(_rec(i, t, "human", role, ["calibration_human"] if role == "calibration" else ["audited_human_core"]))
            i += 1
    for _ in range(20):
        t = " ".join(rng.choice(words, 40))
        rows.append(_rec(i, t, "ai", "evaluation", ["fully_ai_core"]))
        i += 1
    for _ in range(10):
        t = " ".join(rng.choice(words, 50))
        rows.append(_rec(i, t, "mixed", "evaluation", ["controlled_replacement"], spans=[[10, 60]], mech="sentence_replacement"))
        i += 1
    df = pd.DataFrame(rows)
    from aidet_eval.adapters.common import JSON_COLS, dumps

    for c in JSON_COLS:
        if c in df:
            df[c] = df[c].map(lambda v: None if v is None else (v if isinstance(v, str) else dumps(v)))
    df["leakage_group_id"] = "lg_" + df["example_id"]
    df["independence_group_id"] = "ig_" + df["example_id"]
    df["length_bin"] = "lt50w"
    for c in ("ai_fraction_bin", "block_count_bin", "block_length_bin", "block_position"):
        df[c] = None
    df["exposure__fixture_model"] = "no_detected_overlap"
    bench = tmp_path / "benchmark-fixture"
    (bench / "windows").mkdir(parents=True)
    df.to_parquet(bench / "examples.parquet", index=False)
    cal_ids = sorted(df.loc[df.role == "calibration", "example_id"])
    manifest = {"benchmark_id": "fixture-benchmark", "benchmark_version": "0.0.0", "profile": "smoke", "purpose": "development_regression",
                "release_role": "synthetic fixture", "seed": 42, "benchmark_hash": sha256_text("fixture"),
                "fingerprints": {"selection_sha256": "x", "text_gold_fingerprint": "y", "calibration_membership_sha256": sha256_text("\n".join(cal_ids))},
                "cells": [], "counts": {"examples": len(df), "evaluation": int((df.role == "evaluation").sum()), "calibration": len(cal_ids),
                                        "evaluation_leakage_groups": 0, "evaluation_independence_groups": 0},
                "domain_mappings": {}, "overlap": {}, "excerpt_policy": "-", "offset_convention": "code points", "length_function": "-",
                "length_bins": [], "ai_fraction_bins": [], "reserve_sources": {}, "selection_policy": {}, "plan_sha256": "-",
                "training_exposure": {}}
    write_json(bench / "manifest.json", manifest)
    win = pd.DataFrame([{"window_id": f"{e}#w000", "example_id": e, "window_index": 0, "start": 0, "end": len(t), "n_words": len(t.split()),
                         "kind": "words", "first_word": 0, "window_text_sha256": sha256_text(t), "example_text_sha256": sha256_text(t)}
                        for e, t in zip(df.example_id, df.text)])
    win.to_parquet(bench / "windows" / "shared_windows_v1.parquet", index=False)
    protos = [{"protocol_id": "shared_windows_v1", "version": "1.0.0", "implementation_kind": "harness_wrapper", "fingerprint": "fp-shared",
               "localisation_mode": "unsupported", "document_aggregation": "max(window_score)", "truncation": "prohibited", "window_manifest_sha256": "w"}]
    write_json(bench / "windows" / "protocols.json", {"protocols": protos, "n_windows": len(win), "n_examples": len(df), "windows_per_example": {}})
    # synthetic run
    rd = paths.RUNS / "fixture-run"
    (rd / "predictions" / "fixture_model").mkdir(parents=True)
    write_json(rd / "run.json", {"run_id": "fixture-run", "created_at": "2026-01-01T00:00:00+00:00", "benchmark_hash": manifest["benchmark_hash"],
                                 "benchmark_id": "fixture-benchmark", "profile": "smoke", "requested_models": ["fixture_model"],
                                 "requested_protocols": ["shared_windows_v1"], "data_origin": "synthetic_fixture", "git": {}, "platform": {}})
    write_json(rd / "models" / "fixture_model.json", {"model_id": "fixture_model", "status": "loaded", "model_fingerprint": "fixture",
                                                      "describe": {"display_name": "Synthetic fixture model", "score_semantics": "synthetic_fixture_score",
                                                                   "precision": {"weights": "none"}}, "runtime": {"device": "none"}})
    with open(rd / "predictions" / "fixture_model" / "shared_windows_v1.jsonl", "w") as f:
        for e, a in zip(df.example_id, df.authorship):
            if e == "fixture-095":
                continue  # one deliberately missing prediction -> failure accounting
            base = {"human": 0.2, "ai": 0.8, "mixed": 0.6}[a]
            s = float(np.clip(base + rng.normal(0, 0.15), 0, 1))
            probs = [1 - s, 0.0, 0.0, s]
            sc = float(np.dot(probs, [0, 1, 2, 3]) / 3)
            f.write(json.dumps({"input_id": f"{e}#w000", "example_id": e, "status": "ok", "score": sc,
                                "raw_outputs": {"logits": [0, 0, 0, 0], "bucket_probabilities": probs, "predicted_bucket": 0, "n_buckets": 4},
                                "coverage": {"mapping_status": "exact", "deleted_by_preprocessing": [], "truncated": False},
                                "n_tokens": 10, "input_fingerprint": "x", "cache_key": "k", "error": None, "seconds_per_item": 0.001,
                                "at": "2026-01-01T00:00:01+00:00"}) + "\n")
    write_json(paths.INVENTORY / "inventory.json", {"sources": []})
    from aidet_eval.evaluate import evaluate_run

    evaluate_run(str(bench), "fixture-run")
    return {"bench": bench, "run_dir": rd, "examples": df}
