"""Calibration sensitivity analysis (exploratory; never replaces official operating points).

For each model x protocol x target alpha:
  * per calibration dataset d: the same order-statistic threshold computed on d alone, with a
    percentile bootstrap interval (documents resampled, 2000 replicates, seed 42);
  * pooled (official rule), leave-one-dataset-out pooled, equal-weight pooled (smallest threshold
    whose *mean* per-dataset calibration FPR <= alpha), and max-over-datasets (every dataset's
    calibration FPR <= alpha);
  * consequences of each variant on the evaluation panels (audited/student/instruction human FPR,
    fully-AI and replacement recall by domain), computed from saved predictions.

Statistical notes recorded in the output:
  - With n = 500 the 1% threshold is the 6th-highest score (k = 5): a high-variance tail
    order statistic, hence the intervals.
  - Dropping a dataset whose threshold is an outlier changes the population the FPR target
    refers to; that dataset's FPR must still be reported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from aidet_eval.calibrate import OPERATING_POINTS, threshold_for
from aidet_eval.docpreds import doc_predictions
from aidet_eval.util import log_event, read_json, utcnow, write_json

REPS, SEED = 2000, 42


def _thr(scores: np.ndarray, alpha: float) -> float | None:
    n = len(scores)
    if n == 0:
        return None
    k = math.floor(alpha * n)
    return float(np.sort(scores)[::-1][k]) if k < n else None


def _boot(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(scores)
    k = math.floor(alpha * n)
    samp = np.sort(scores[rng.integers(0, n, (REPS, n))], axis=1)[:, ::-1][:, k]
    return float(np.percentile(samp, 2.5)), float(np.percentile(samp, 97.5))


def _equal_weight(by: dict[str, np.ndarray], alpha: float) -> float:
    cands = np.unique(np.concatenate(list(by.values())))
    for t in cands:  # ascending: first t meeting the constraint is the smallest
        if np.mean([(s > t).mean() for s in by.values()]) <= alpha:
            return float(t)
    return float(cands[-1])


def run_sensitivity(bench: str, run_dir: str) -> Path:
    bench_dir, rd = Path(bench), Path(run_dir)
    run = read_json(rd / "run.json")
    ex = pd.read_parquet(bench_dir / "examples.parquet", columns=["example_id", "role", "authorship", "dataset_key", "domain", "panel_ids"])
    ex["panels"] = ex["panel_ids"].map(json.loads)
    cal = ex[ex["role"] == "calibration"]
    ev = ex[ex["role"] == "evaluation"]
    out = {"created_at": utcnow(), "run_id": run["run_id"], "status": "exploratory_sensitivity_analysis",
           "note": "Not official operating points. Thresholds use the official order-statistic rule (flag iff score > t) on subsets of the SAME calibration documents.",
           "bootstrap": {"replicates": REPS, "seed": SEED, "unit": "calibration document"},
           "statistical_caveats": [
               "At n=500 per dataset, the 1% threshold is the 6th-highest score (k=5); the 5% threshold is the 26th (k=25). Tail order statistics are noisy: compare intervals, not points.",
               "Excluding a dataset with an outlying threshold redefines the population the FPR target covers; its FPR must still be reported.",
               "Equal-weight pooling targets the mean per-dataset FPR; max-over-datasets targets every dataset individually (most conservative)."],
           "results": []}
    for mid in run["requested_models"]:
        for pid in run["requested_protocols"]:
            dp = doc_predictions(rd, bench_dir, mid, pid, ex["example_id"].tolist()).set_index("example_id")
            sc = ex.set_index("example_id").join(dp[["status", "score"]])
            sc = sc[sc["status"] == "ok"]
            csc = sc.loc[sc.index.isin(cal["example_id"])]
            by = {d: g["score"].to_numpy(float) for d, g in csc.groupby("dataset_key")}
            allc = csc["score"].to_numpy(float)
            for op_id, alpha, _ in OPERATING_POINTS:
                per = []
                for d, s in sorted(by.items()):
                    lo, hi = _boot(s, alpha)
                    per.append({"dataset": d, "domain": cal[cal.dataset_key == d]["domain"].iloc[0], "n": int(len(s)),
                                "k": math.floor(alpha * len(s)), "threshold": _thr(s, alpha), "ci_lower": lo, "ci_upper": hi,
                                "pooled_threshold_fpr_on_this_dataset": None})
                pooled = _thr(allc, alpha)
                for p in per:
                    p["pooled_threshold_fpr_on_this_dataset"] = float((by[p["dataset"]] > pooled).mean())
                thr_vals = [p["threshold"] for p in per]
                variants = {"pooled_official_rule": pooled,
                            "equal_weight_mean_fpr": _equal_weight(by, alpha),
                            "max_over_datasets": max(thr_vals),
                            "median_of_datasets": float(np.median(thr_vals))}
                for d in by:
                    rest = np.concatenate([s for k2, s in by.items() if k2 != d])
                    variants[f"leave_out:{d}"] = _thr(rest, alpha)
                cons = {}
                for name, t in variants.items():
                    cons[name] = _consequences(sc, ev, t)
                assert abs(pooled - threshold_for(list(allc), alpha)["threshold"]) == 0
                out["results"].append({"model_id": mid, "protocol_id": pid, "operating_point_id": op_id, "target_fpr": alpha,
                                       "per_dataset": per, "summary": {"min": min(thr_vals), "median": float(np.median(thr_vals)),
                                                                       "max": max(thr_vals),
                                                                       "outlier_rule": "dataset whose 95% bootstrap interval overlaps no other dataset's interval (high = above all, low = below all)",
                                                                       "outliers": [f"{p['dataset']} (high)" for p in per if all(p["ci_lower"] > q["ci_upper"] for q in per if q is not p)] + [f"{p['dataset']} (low)" for p in per if all(p["ci_upper"] < q["ci_lower"] for q in per if q is not p)]},
                                       "variants": variants, "consequences": cons})
            log_event("calibration_sensitivity", model=mid, protocol=pid)
    path = rd / "calibration_sensitivity.json"
    write_json(path, out)
    return path


def _consequences(sc: pd.DataFrame, ev: pd.DataFrame, t: float) -> dict:
    e = sc.loc[sc.index.isin(ev["example_id"])]
    pan = ev.set_index("example_id")["panels"]
    def rate(mask):
        m = e[mask]
        return {"value": float((m["score"] > t).mean()) if len(m) else None, "numerator": int((m["score"] > t).sum()), "denominator": int(len(m))}
    inp = e.index.map(pan)
    has = lambda p: np.array([p in x for x in inp])  # noqa: E731
    out = {"threshold": t}
    for dom in ["research", "news", "reviews", "expository", "creative"]:
        out[f"audited_human_fpr:{dom}"] = rate(has("audited_human_core") & (e["domain"] == dom).to_numpy())
        out[f"fully_ai_recall:{dom}"] = rate(has("fully_ai_core") & (e["domain"] == dom).to_numpy())
        out[f"replacement_recall:{dom}"] = rate(has("controlled_replacement") & (e["domain"] == dom).to_numpy())
    out["student_human_fpr"] = rate(has("student_human"))
    out["instruction_human_fpr"] = rate(has("modern_human_instructions"))
    out["fully_ai_recall:all_core"] = rate(has("fully_ai_core"))
    out["replacement_recall:all_core"] = rate(has("controlled_replacement"))
    out["aitdna_mixed_recall"] = rate(has("aitdna_real_cowriting") & (e["authorship"] == "mixed").to_numpy())
    out["block_replacement_recall"] = rate(has("heterogeneous_block_replacement"))
    return out
