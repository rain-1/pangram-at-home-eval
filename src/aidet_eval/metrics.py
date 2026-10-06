"""Metric values, intervals and failure bounds following RESULTS_CONTRACT.md section 6.

Every metric is a dict with status/value/unit/percent/numerator/denominator/... Undefined values
are null with a machine-readable reason; never NaN/inf, never zero-for-missing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

Z95 = 1.959963984540054
BOOT_REPS = 2000
BOOT_SEED = 42


def metric(status: str, value: float | None = None, *, unit: str = "proportion", numerator: int | None = None,
           denominator: int | None = None, denominator_population: str | None = None, n_examples: int = 0,
           n_independence_groups: int = 0, ci: dict | None = None, failure_bounds: dict | None = None,
           reason: str | None = None, extra: dict | None = None) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        raise ValueError("non-finite metric value")
    m = {"status": status, "value": value, "unit": unit,
         "percent": (100.0 * value) if (value is not None and unit == "proportion") else None,
         "numerator": numerator, "denominator": denominator, "denominator_population": denominator_population,
         "n_examples": int(n_examples), "n_independence_groups": int(n_independence_groups), "ci": ci,
         "failure_bounds": failure_bounds, "reason": reason}
    if extra:
        m.update(extra)
    return m


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        raise ValueError("n=0")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def ci_dict(method: str, lo: float, hi: float, *, clustering: str | None, n_groups: int, assumptions: str,
            status: str = "ok", seed: int | None = None, reps: int | None = None, degenerate: bool = False,
            undefined_replicates: int = 0) -> dict:
    return {"method": method, "level": 0.95, "lower": lo, "upper": hi, "unit": "proportion",
            "lower_percent": 100 * lo, "upper_percent": 100 * hi, "clustering_variable": clustering,
            "n_groups": int(n_groups), "seed": seed, "replicates": reps, "assumptions": assumptions, "status": status,
            "degenerate_all_zero": bool(degenerate), "undefined_replicates": int(undefined_replicates)}


def group_sums(flags: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group (numerator, denominator) arrays."""
    uniq, inv = np.unique(groups, return_inverse=True)
    num = np.bincount(inv, weights=flags.astype(float), minlength=len(uniq))
    den = np.bincount(inv, minlength=len(uniq)).astype(float)
    return num, den


def cluster_bootstrap_rate(num_g: np.ndarray, den_g: np.ndarray, reps: int = BOOT_REPS, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    G = len(num_g)
    idx = rng.integers(0, G, size=(reps, G))
    n = num_g[idx].sum(1)
    d = den_g[idx].sum(1)
    ok = d > 0
    vals = n[ok] / d[ok]
    if len(vals) == 0:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int((~ok).sum())


def rate(flags: np.ndarray, groups: np.ndarray, *, denominator_population: str, n_eligible: int,
         n_unscored: int, higher_is_error: bool, reason_if_empty: str = "no_scored_examples_in_cell") -> dict:
    """Proportion with Wilson (independent) or cluster-bootstrap (clustered) CI and failure bounds."""
    n = int(len(flags))
    if n == 0:
        st = "not_applicable" if n_eligible == 0 else "undefined"
        return metric(st, None, denominator_population=denominator_population, n_examples=0,
                      reason=reason_if_empty if n_eligible == 0 else "all_eligible_predictions_failed",
                      failure_bounds=_bounds(0, 0, n_eligible, n_unscored, higher_is_error))
    k = int(flags.sum())
    num_g, den_g = group_sums(flags, groups)
    G = len(num_g)
    v = k / n
    if G == n:
        lo, hi = wilson(k, n)
        ci = ci_dict("wilson_score", lo, hi, clustering=None, n_groups=G,
                     assumptions="each example treated as an independent observation (one per independence group)")
    else:
        b = cluster_bootstrap_rate(num_g, den_g)
        if b is None:
            ci = None
        else:
            lo, hi, und = b
            ci = ci_dict("cluster_percentile_bootstrap", lo, hi, clustering="independence_group_id", n_groups=G,
                         assumptions="independence groups resampled with replacement", seed=BOOT_SEED, reps=BOOT_REPS,
                         degenerate=(lo == hi == 0.0) or (lo == hi == 1.0), undefined_replicates=und)
    extra = {}
    if k == 0 and higher_is_error:
        # separately labelled group-level any-error bound (not a document-level bound)
        extra["group_level_any_error_upper_95"] = {
            "value": 1 - 0.05 ** (1 / G), "n_groups": G, "method": "zero-event one-sided binomial over independence groups",
            "note": "bound on the share of groups with any error; not an exact document-level bound for unequal clusters"}
    return metric("ok", v, numerator=k, denominator=n, denominator_population=denominator_population, n_examples=n,
                  n_independence_groups=G, ci=ci, failure_bounds=_bounds(k, n, n_eligible, n_unscored, higher_is_error),
                  extra=extra or None)


def _bounds(k: int, n: int, n_eligible: int, n_unscored: int, higher_is_error: bool) -> dict | None:
    if n_unscored == 0 or n_eligible == 0:
        return None
    lo = k / n_eligible
    hi = (k + n_unscored) / n_eligible
    return {"lower": lo, "upper": hi, "n_eligible": n_eligible, "n_unscored": n_unscored,
            "note": "bounds over all eligible examples treating unscored ones as all-negative/all-positive decisions; not observed rates"}


# ------------------------------------------------------------------- ranking metrics

def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = labels == 1, labels == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, bootstrap: bool = True) -> dict:
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    support = {"n_positive": n1, "n_negative": n0, "prevalence": (n1 / (n1 + n0)) if (n1 + n0) else None}
    if n1 == 0 or n0 == 0:
        reason = "one_class_slice"
        return {"auroc": metric("undefined", None, unit="score", reason=reason, extra={"class_support": support}),
                "average_precision": metric("undefined", None, unit="score", reason=reason, extra={"class_support": support})}
    a = auroc(scores, labels)
    ap = average_precision(scores, labels)
    G = len(np.unique(groups))
    ci_a = None
    if bootstrap:
        lo_hi, und, reps = _auroc_cluster_bootstrap(scores, labels, groups)
        if lo_hi is not None:
            ci_a = ci_dict("cluster_percentile_bootstrap", lo_hi[0], lo_hi[1], clustering="independence_group_id", n_groups=G,
                           assumptions="independence groups resampled with replacement (group-count weights)",
                           seed=BOOT_SEED, reps=reps, undefined_replicates=und)
            ci_a["unit"] = "score"
            ci_a.pop("lower_percent"), ci_a.pop("upper_percent")
    note = "AP depends on the sampled class balance (prevalence shown); not a deployment precision"
    return {"auroc": metric("ok", a, unit="score", n_examples=n1 + n0, n_independence_groups=G, ci=ci_a, extra={"class_support": support}),
            "average_precision": metric("ok", ap, unit="score", n_examples=n1 + n0, n_independence_groups=G,
                                        extra={"class_support": support, "note": note})}


def _auroc_cluster_bootstrap(scores, labels, groups, reps: int = BOOT_REPS, seed: int = BOOT_SEED):
    """Vectorised group bootstrap of AUROC via weighted Mann-Whitney U (ties count 1/2).

    A replicate draws G groups with replacement; each example's weight is its group's draw count,
    which is exactly equivalent to concatenating the resampled groups.
    """
    order = np.argsort(scores, kind="mergesort")
    s, y = scores[order], labels[order].astype(float)
    ug, inv = np.unique(groups[order], return_inverse=True)
    G = len(ug)
    rng = np.random.default_rng(seed)
    counts = np.zeros((reps, G))
    for r in range(reps):  # bincount per replicate (cheap)
        counts[r] = np.bincount(rng.integers(0, G, G), minlength=G)
    W = counts[:, inv]  # reps x n example weights
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]  # tie blocks in sorted order
    pos = np.add.reduceat(W * y, starts, axis=1)
    neg = np.add.reduceat(W * (1 - y), starts, axis=1)
    cum_neg_before = np.cumsum(neg, axis=1) - neg
    U = (pos * (cum_neg_before + 0.5 * neg)).sum(1)
    denom = pos.sum(1) * neg.sum(1)
    ok = denom > 0
    if not ok.any():
        return None, int(reps), reps
    vals = U[ok] / denom[ok]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))), int((~ok).sum()), reps


def paired_difference(a_flags: np.ndarray, b_flags: np.ndarray, groups: np.ndarray) -> dict:
    """Difference in rates (a - b) on a common population, in percentage points, paired group bootstrap."""
    n = len(a_flags)
    if n == 0:
        return {"status": "undefined", "reason": "no_common_examples", "difference_pp": None, "ci_pp": None}
    na, _ = group_sums(a_flags, groups)
    nb, den = group_sums(b_flags, groups)
    diff = (a_flags.mean() - b_flags.mean()) * 100
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(den), size=(BOOT_REPS, len(den)))
    d = den[idx].sum(1)
    vals = (na[idx].sum(1) - nb[idx].sum(1)) / d * 100
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"status": "ok", "difference_pp": float(diff), "ci_pp": {"method": "paired_cluster_percentile_bootstrap", "level": 0.95,
            "lower": lo, "upper": hi, "clustering_variable": "independence_group_id", "n_groups": int(len(den)),
            "seed": BOOT_SEED, "replicates": BOOT_REPS, "degenerate": lo == hi},
            "n_common": int(n)}
