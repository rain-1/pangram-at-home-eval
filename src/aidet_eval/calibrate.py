"""Freeze operating points from human calibration documents only.

For each model x protocol x target alpha: n = eligible, successfully scored human calibration docs,
k = floor(alpha * n); sort scores descending; threshold = score at zero-based index k; decision
rule `score > threshold`. This admits at most k calibration false positives including ties.
No eligible calibration data -> blocked operating point (never a default 0.5).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aidet_eval import paths
from aidet_eval.docpreds import doc_predictions
from aidet_eval.metrics import wilson
from aidet_eval.predict import default_run_id
from aidet_eval.util import log_event, read_json, sha256_text, utcnow, write_json

ALGORITHM = "descending_order_statistic_v1: threshold = sorted_desc[floor(alpha*n)], flag iff score > threshold"
OPERATING_POINTS = [("human_fpr_1pct", 0.01, True), ("human_fpr_5pct", 0.05, False)]


def threshold_for(scores: list[float], alpha: float) -> dict:
    n = len(scores)
    if n == 0:
        return {"status": "blocked", "reason": "no_successfully_scored_calibration_documents", "threshold": None}
    s = sorted(scores, reverse=True)
    k = math.floor(alpha * n)
    if k >= n:
        return {"status": "blocked", "reason": "alpha*n >= n", "threshold": None}
    thr = s[k]
    fp = sum(1 for x in scores if x > thr)
    ties = sum(1 for x in scores if x == thr)
    lo, hi = wilson(fp, n)
    return {"status": "ok", "threshold": thr, "k": k, "n": n, "false_positives": fp, "actual_fpr": fp / n,
            "tied_scores_at_threshold": ties, "max_allowed_false_positives": k,
            "actual_fpr_ci": {"method": "wilson_score", "level": 0.95, "lower": lo, "upper": hi,
                              "assumptions": "calibration documents treated as independent"}}


def calibrate_run(bench: str, run_id: str | None = None) -> dict:
    bench_dir = Path(bench)
    run_id = run_id or default_run_id(bench_dir)
    rd = paths.RUNS / run_id
    run = read_json(rd / "run.json")
    ex = pd.read_parquet(bench_dir / "examples.parquet", columns=["example_id", "role", "authorship", "domain", "eligibility"])
    cal = ex[(ex["role"] == "calibration") & (ex["authorship"] == "human")]
    cal_ids = sorted(cal["example_id"])
    cal_hash = sha256_text("\n".join(cal_ids))
    man = read_json(bench_dir / "manifest.json")
    assert cal_hash == sha256_text("\n".join(sorted(cal_ids))), "calibration ids unstable"
    records = []
    for mid in run["requested_models"]:
        for pid in run["requested_protocols"]:
            dp = doc_predictions(rd, bench_dir, mid, pid, cal_ids)
            ok = dp[dp["status"] == "ok"]
            failed = dp[dp["status"] != "ok"]
            scores = [float(x) for x in ok["score"]]
            for op_id, alpha, primary in OPERATING_POINTS:
                t = threshold_for(scores, alpha)
                complete = len(failed) == 0
                status = t["status"] if t["status"] != "ok" else ("ok" if complete else "provisional")
                rec = {
                    "calibration_id": f"{mid}|{pid}|{op_id}", "model_id": mid, "protocol_id": pid, "operating_point_id": op_id,
                    "primary": primary, "task": "human_false_positive_control", "target_fpr": alpha, "comparator": ">",
                    "threshold": t.get("threshold"), "status": status,
                    "reasons": ([t["reason"]] if t.get("reason") else []) + ([] if complete else [f"{len(failed)} calibration documents failed/abstained: provisional"]),
                    "calibration_example_ids_sha256": sha256_text("\n".join(cal_ids)),
                    "benchmark_calibration_membership_sha256": man["fingerprints"]["calibration_membership_sha256"],
                    "n_selected": len(cal_ids), "n_eligible": int((cal["eligibility"] == "eligible").sum()), "n_scored": len(scores),
                    "n_failed": int(len(failed)), "failed_example_ids": sorted(failed["example_id"])[:200],
                    "false_positives": t.get("false_positives"), "actual_fpr": t.get("actual_fpr"),
                    "actual_fpr_ci": t.get("actual_fpr_ci"), "k": t.get("k"),
                    "tied_scores_at_threshold": t.get("tied_scores_at_threshold"),
                    "calibration_distribution": cal.groupby("domain").size().to_dict(),
                    "weights": "empirical (each calibration document weight 1)",
                    "selection_algorithm": ALGORITHM, "algorithm_version": "1",
                    "note": "target FPR is a calibration-set property, not a guaranteed test FPR",
                }
                records.append(rec)
    out = {"created_at": utcnow(), "run_id": run_id, "calibration_membership_sha256": man["fingerprints"]["calibration_membership_sha256"],
           "operating_points": records}
    write_json(rd / "calibration.json", out)
    for r in records:
        log_event("calibrated", id=r["calibration_id"], status=r["status"], threshold=r["threshold"], actual_fpr=r["actual_fpr"], n=r["n_scored"])
    return out
