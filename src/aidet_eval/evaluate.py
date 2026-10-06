"""Aggregate evaluation: predeclared slice cells, ranking metrics, coverage, comparisons.

Produces the full result document (RESULTS_CONTRACT.md) in memory; export.py writes it and derives
the summary deterministically. Decisions are recomputed from raw scores and the recorded exact
thresholds with the recorded strict comparator; no rounded value feeds any calculation.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from aidet_eval import metrics as M
from aidet_eval import paths
from aidet_eval.build import jl
from aidet_eval.docpreds import doc_predictions
from aidet_eval.predict import default_run_id
from aidet_eval.util import fingerprint, log_event, nan_to_none, read_json, sha256_text, utcnow

HUMAN_PANELS_AUDITED = ["audited_human_core", "student_human", "modern_human_instructions"]
TASKS = [
    {"task_id": "human_vs_fully_ai", "positive": "authorship == ai", "negative": "authorship == human", "excluded": "mixed, unknown",
     "class_order": ["human", "ai"], "description": "Detect fully AI-generated documents while rarely flagging human writing"},
    {"task_id": "human_vs_any_ai_involvement", "positive": "ai_involvement == yes (ai or mixed; mechanisms visible as slices)",
     "negative": "ai_involvement == no (human)", "excluded": "unknown", "class_order": ["human", "ai_involvement"],
     "description": "Detect any documented AI involvement (fully AI, insertions, editing, co-writing)"},
]
WINDOW_BINS = [("1w", 1, 1), ("2w", 2, 2), ("3_4w", 3, 4), ("5_9w", 5, 9), ("ge10w", 10, 10**6)]


def window_bin(n: int) -> str | None:
    for name, lo, hi in WINDOW_BINS:
        if lo <= n <= hi:
            return name
    return None


def _prep_examples(bench_dir: Path) -> pd.DataFrame:
    ex = pd.read_parquet(bench_dir / "examples.parquet")
    ex["panels"] = ex["panel_ids"].map(json.loads)
    ex["generation_task"] = ex["extensions"].map(lambda e: (jl(e) or {}).get("generation_task"))
    lab = ex["original_labels"].map(lambda s: jl(s) or {})
    ex["prompt_name"] = lab.map(lambda d: d.get("prompt_name"))
    ex["essay_score"] = lab.map(lambda d: d.get("holistic_essay_score"))
    ex["grade_level"] = lab.map(lambda d: d.get("grade_level") or None)
    ex["text_type"] = lab.map(lambda d: d.get("text_type"))
    ex["beemo_field"] = lab.map(lambda d: d.get("field"))
    ex["aitdna_setting"] = ex["extensions"].map(lambda e: (jl(e) or {}).get("setting"))
    win = pd.read_parquet(bench_dir / "windows" / "shared_windows_v1.parquet", columns=["example_id"])
    wc = win.groupby("example_id").size()
    ex["n_windows"] = ex["example_id"].map(wc).fillna(0).astype(int)
    ex["window_count_bin"] = ex["n_windows"].map(window_bin)
    return ex


# ------------------------------------------------------------------ cell machinery

class CellBuilder:
    def __init__(self, ex: pd.DataFrame, preds: pd.DataFrame, model_id: str, protocol_id: str, ops: list[dict],
                 exposure_col: str | None):
        self.ex = ex
        self.model_id, self.protocol_id = model_id, protocol_id
        df = ex.merge(preds[["example_id", "status", "score"]], on="example_id", how="left")
        df["status"] = df["status"].fillna("missing")
        self.df = df
        self.ops = ops
        self.exposure_col = exposure_col
        self.cells: list[dict] = []

    def add(self, task_id: str | None, panel_id: str, population: str, mask: np.ndarray, slice_: dict, view: str = "all",
            ranking: tuple[np.ndarray, np.ndarray] | None = None):
        d = self.df[mask]
        if view == "excluding_known_training_overlap" and self.exposure_col:
            d = d[d[self.exposure_col] != "known_overlap"]
        ids_hash = sha256_text("\n".join(sorted(d["example_id"])))
        scored = d["status"] == "ok"
        support = {
            "selected": int(len(d)), "eligible": int((d["eligibility"].isin(["eligible"]) | d["dataset_key"].eq("damasha")).sum()),
            "scored": int(scored.sum()), "error": int(d["status"].isin(["error", "missing"]).sum()),
            "abstained": int((d["status"] == "abstain").sum()), "skipped": 0, "blocked": 0,
            "human": int((d["authorship"] == "human").sum()), "ai": int((d["authorship"] == "ai").sum()),
            "mixed": int((d["authorship"] == "mixed").sum()), "unknown": int((d["authorship"] == "unknown").sum()),
            "n_source_groups": int(d["source_group_id"].nunique()), "n_leakage_groups": int(d["leakage_group_id"].nunique()),
            "n_independence_groups": int(d["independence_group_id"].nunique()),
        }
        cov = M.metric("ok" if len(d) else "not_applicable", (support["scored"] / len(d)) if len(d) else None,
                       numerator=support["scored"], denominator=len(d), denominator_population="selected_examples",
                       n_examples=len(d), n_independence_groups=support["n_independence_groups"],
                       reason=None if len(d) else "empty_cell")
        base = {"model_id": self.model_id, "protocol_id": self.protocol_id, "task_id": task_id, "panel_id": panel_id,
                "split": "evaluation", "slice": {**slice_, "view": view}, "population_definition": population,
                "example_ids_hash": ids_hash, "support": support, "warnings": []}
        if len(d) == 0:
            base["warnings"].append("requested_cell_is_empty")
        # operating-point cells
        for op in self.ops:
            thr, st = op["threshold"], op["status"]
            cell = dict(base)
            cell["operating_point_id"] = op["operating_point_id"]
            cell["cell_id"] = fingerprint([self.model_id, self.protocol_id, op["operating_point_id"], task_id, panel_id, slice_, view])[:24]
            mets = {"prediction_coverage": cov}
            if thr is None:
                for name in ("human_fpr", "fully_ai_recall", "mixed_recall"):
                    mets[name] = M.metric("not_computed", None, reason=f"operating_point_{st}")
                cell["metrics"], cell["confusion_matrix"] = mets, None
                self.cells.append(cell)
                continue
            for name, cls, err in (("human_fpr", "human", True), ("fully_ai_recall", "ai", False), ("mixed_recall", "mixed", False)):
                sub = d[d["authorship"] == cls]
                if len(sub) == 0:
                    mets[name] = M.metric("not_applicable", None, reason=f"no_{cls}_examples_in_cell")
                    continue
                ok = sub[sub["status"] == "ok"]
                flags = (ok["score"].to_numpy(dtype=float) > thr)
                m = M.rate(flags, ok["independence_group_id"].to_numpy(), denominator_population=f"scored_{cls}_examples",
                           n_eligible=len(sub), n_unscored=int(len(sub) - len(ok)), higher_is_error=err)
                if st == "provisional" and m["status"] == "ok":
                    m["status"] = "provisional"
                    m["reason"] = "calibration_incomplete"
                mets[name] = m
            if st == "provisional":
                cell["warnings"] = cell["warnings"] + ["provisional_operating_point"]
            cell["metrics"] = mets
            cell["confusion_matrix"] = self._confusion(d, thr, task_id)
            self.cells.append(cell)
        # threshold-free cell (ranking metrics)
        if ranking is not None or task_id is not None:
            cell = dict(base)
            cell["operating_point_id"] = None
            cell["cell_id"] = fingerprint([self.model_id, self.protocol_id, None, task_id, panel_id, slice_, view])[:24]
            mets = {"prediction_coverage": cov}
            if task_id is not None:
                lab = self._labels(d, task_id)
                ok = d[(d["status"] == "ok") & lab.notna()]
                y = lab[ok.index].to_numpy(dtype=int)
                mets.update(M.ranking_metrics(ok["score"].to_numpy(dtype=float), y, ok["independence_group_id"].to_numpy()))
                cell["warnings"] = cell["warnings"] + (["ranking_curve_is_descriptive_not_calibrated"])
            cell["metrics"] = mets
            cell["confusion_matrix"] = None
            self.cells.append(cell)

    @staticmethod
    def _labels(d: pd.DataFrame, task_id: str | None) -> pd.Series:
        if task_id == "human_vs_fully_ai":
            return d["authorship"].map({"human": 0, "ai": 1})
        return d["ai_involvement"].map({"no": 0, "yes": 1})

    def _confusion(self, d: pd.DataFrame, thr: float, task_id: str | None) -> dict | None:
        if task_id is None:
            return None
        lab = self._labels(d, task_id)
        ok = d[(d["status"] == "ok") & lab.notna()]
        y = lab[ok.index].to_numpy(dtype=int)
        p = (ok["score"].to_numpy(dtype=float) > thr).astype(int)
        cm = {"class_order": ["negative_human", "positive"], "positive_definition": next(t["positive"] for t in TASKS if t["task_id"] == task_id),
              "negative_definition": "authorship == human", "excluded": "unscored, unknown" + (", mixed" if task_id == "human_vs_fully_ai" else ""),
              "tn": int(((y == 0) & (p == 0)).sum()), "fp": int(((y == 0) & (p == 1)).sum()),
              "fn": int(((y == 1) & (p == 0)).sum()), "tp": int(((y == 1) & (p == 1)).sum())}
        if task_id == "human_vs_any_ai_involvement":
            mech = ok["mix_mechanism"].fillna(ok["authorship"])
            cm["positives_by_mechanism"] = {str(k): {"tp": int(((y == 1) & (p == 1) & (mech == k).to_numpy()).sum()),
                                                     "fn": int(((y == 1) & (p == 0) & (mech == k).to_numpy()).sum())}
                                            for k in sorted(mech[y == 1].unique())}
        return cm


def build_cells(cb: CellBuilder) -> None:
    d = cb.df
    inp = d["panels"]

    def P(p):  # membership mask
        return inp.map(lambda ps: p in ps).to_numpy()

    def add_by(panel, col, task=None, extra_mask=None, values=None, slice_extra=None):
        base = P(panel) if panel != "_all_evaluation" else np.ones(len(d), bool)
        if extra_mask is not None:
            base = base & extra_mask
        vals = values if values is not None else sorted(v for v in d.loc[base, col].dropna().unique())
        for v in vals:
            cb.add(task, panel, f"{panel} where {col} == {v!r}", base & (d[col] == v).to_numpy(), {**(slice_extra or {}), col: v})

    panels = sorted({p for ps in inp for p in ps})
    task_for = {"fully_ai_core": "human_vs_fully_ai", "fully_ai_noncore": "human_vs_fully_ai"}
    for p in panels:
        cb.add(None, p, f"all evaluation examples in panel {p}", P(p), {})
    # domain-specific
    for p in ["audited_human_core", "fully_ai_core", "fully_ai_noncore", "controlled_replacement", "controlled_replacement_noncore",
              "benchmark_source_matched_human_references"]:
        add_by(p, "domain")
    for p in ["audited_human_core", "fully_ai_core", "benchmark_source_matched_human_references"]:
        add_by(p, "dataset_key")
        for dom in sorted(d.loc[P(p), "domain"].dropna().unique()):
            add_by(p, "dataset_key", extra_mask=(d["domain"] == dom).to_numpy(), slice_extra={"domain": dom})
    # lengths & provenance & window counts (human false positives)
    for p in HUMAN_PANELS_AUDITED + ["benchmark_source_matched_human_references", "aitdna_real_cowriting", "fully_ai_core",
                                     "controlled_replacement"]:
        add_by(p, "length_bin")
    for dom in ["research", "news", "reviews", "expository", "creative"]:
        add_by("audited_human_core", "length_bin", extra_mask=(d["domain"] == dom).to_numpy(), slice_extra={"domain": dom})
        add_by("audited_human_core", "subdomain", extra_mask=(d["domain"] == dom).to_numpy(), slice_extra={"domain": dom})
    hum_eval = (d["authorship"] == "human").to_numpy()
    add_by("_all_evaluation", "human_provenance_category", extra_mask=hum_eval, slice_extra={"authorship": "human"})
    add_by("_all_evaluation", "provenance_tier", extra_mask=hum_eval, slice_extra={"authorship": "human"})
    if cb.protocol_id == "shared_windows_v1":
        for p in HUMAN_PANELS_AUDITED + ["fully_ai_core", "controlled_replacement"]:
            add_by(p, "window_count_bin")
    # fully AI
    for p in ["fully_ai_core", "fully_ai_noncore"]:
        add_by(p, "generator")
        add_by(p, "generation_task")
    # mixtures
    for p in ["controlled_replacement", "controlled_replacement_noncore", "damasha_tagged_mixed_unknown_mechanism"]:
        for col in ("ai_fraction_bin", "block_count_bin", "block_length_bin", "block_position"):
            add_by(p, col)
    for dom in ["research", "news", "reviews", "expository"]:
        add_by("controlled_replacement", "ai_fraction_bin", extra_mask=(d["domain"] == dom).to_numpy(), slice_extra={"domain": dom})
    # v2 panels: block replacement (generate-heterogeneous) + matched controls; LAMP human edits of AI drafts
    for col in ("domain", "subdomain", "generator", "ai_fraction_bin", "block_count_bin", "block_length_bin", "block_position", "length_bin"):
        add_by("heterogeneous_block_replacement", col)
    for col in ("domain", "subdomain", "length_bin"):
        add_by("heterogeneous_matched_human_controls", col)
    for col in ("authorship", "subdomain", "generator"):
        add_by("lamp_human_edited_ai", col)
    # AITDNA
    for col in ("authorship", "subdomain", "generator", "ai_fraction_bin", "aitdna_setting"):
        add_by("aitdna_real_cowriting", col)
    # Beemo
    for col in ("beemo_field", "subdomain", "generator"):
        add_by("beemo_matched_editing", col)
    # student / instruction / editlens
    for col in ("essay_score", "grade_level", "subdomain", "prompt_name"):
        add_by("student_human", col)
    add_by("modern_human_instructions", "subdomain")
    for col in ("text_type", "subdomain"):
        add_by("editlens_in_family_reference", col)
    # independent-of-known-training-overlap view for panel x domain
    for p in ["audited_human_core", "fully_ai_core", "controlled_replacement", "benchmark_source_matched_human_references"]:
        base = P(p)
        for dom in sorted(d.loc[base, "domain"].dropna().unique()):
            cb.add(None, p, f"{p} where domain == {dom!r}, excluding known training overlap", base & (d["domain"] == dom).to_numpy(),
                   {"domain": dom}, view="excluding_known_training_overlap")

    # --- task-level ranking + confusion cells
    audited = np.zeros(len(d), bool)
    for p in HUMAN_PANELS_AUDITED:
        audited |= P(p)
    core_h = P("audited_human_core")
    fai = P("fully_ai_core") | P("fully_ai_noncore")
    rep = P("controlled_replacement") | P("controlled_replacement_noncore")
    refs = P("benchmark_source_matched_human_references")
    dom = d["domain"].to_numpy()
    for dm in ["research", "news", "reviews", "expository", "creative", "narrative_summary"]:
        m = dom == dm
        cb.add("human_vs_fully_ai", "task:audited_humans_vs_fully_ai", f"audited human controls vs fully-AI documents, domain {dm} (different sources)",
               (core_h | fai) & m, {"domain": dm, "human_reference": "independently_audited"})
        cb.add("human_vs_fully_ai", "task:matched_refs_vs_fully_ai", f"benchmark source-matched human references vs fully-AI, domain {dm}",
               (refs | fai) & m, {"domain": dm, "human_reference": "benchmark_provided"})
        cb.add("human_vs_any_ai_involvement", "task:audited_humans_vs_ai_or_replacement", f"audited humans vs fully-AI or sentence replacement, domain {dm}",
               (core_h | fai | rep) & m, {"domain": dm, "human_reference": "independently_audited"})
        cb.add("human_vs_any_ai_involvement", "task:matched_refs_vs_replacement", f"source-matched human references vs sentence replacement, domain {dm}",
               (refs | rep) & m, {"domain": dm, "human_reference": "benchmark_provided"})
    cb.add("human_vs_fully_ai", "task:all_audited_humans_vs_all_fully_ai", "all audited human controls vs all fully-AI", audited | fai, {})
    cb.add("human_vs_any_ai_involvement", "task:aitdna_human_vs_cowritten", "AITDNA human-only vs co-written", P("aitdna_real_cowriting"), {})
    bm = P("beemo_matched_editing")
    bf = d["beemo_field"].to_numpy()
    cb.add("human_vs_fully_ai", "task:beemo_human_vs_model_output", "Beemo human_output vs model_output", bm & np.isin(bf, ["human_output", "model_output"]), {})
    cb.add("human_vs_any_ai_involvement", "task:beemo_human_vs_human_edited_ai", "Beemo human_output vs expert-edited AI", bm & np.isin(bf, ["human_output", "human_edits"]), {})
    het = P("heterogeneous_block_replacement") | P("heterogeneous_matched_human_controls")
    for dm in ["creative", "research", "expository"]:
        cb.add("human_vs_any_ai_involvement", "task:heterogeneous_controls_vs_block_replacement",
               f"matched human controls vs block-replacement mixtures, domain {dm}", het & (dom == dm), {"domain": dm})
    el = P("editlens_in_family_reference")
    tt = d["text_type"].to_numpy()
    cb.add("human_vs_fully_ai", "task:editlens_human_vs_generated", "EditLens (in-family) human vs AI-generated", el & np.isin(tt, ["human_written", "ai_generated"]), {})
    cb.add("human_vs_any_ai_involvement", "task:editlens_human_vs_ai_edited", "EditLens (in-family) human vs AI-edited", el & np.isin(tt, ["human_written", "ai_edited"]), {})


# ------------------------------------------------------------------ predictions export

def prediction_records(dp: pd.DataFrame, ex_role: dict, model_id: str, protocol_id: str, ops: list[dict], score_semantics: str) -> list[dict]:
    out = []
    for r in dp.to_dict("records"):
        status = {"ok": "ok", "abstain": "skipped", "error": "error", "missing": "error"}[r["status"]]
        dec = []
        for op in ops:
            flag = None
            if r["status"] == "ok" and op["threshold"] is not None:
                flag = bool(r["score"] > op["threshold"])
            dec.append({"operating_point_id": op["operating_point_id"], "calibration_id": op["calibration_id"], "flag_ai": flag})
        windows, raw, cov, timing, cache, inp_fp, warn = None, None, {}, {}, None, None, []
        if protocol_id == "shared_windows_v1":
            windows = []
            secs = 0.0
            for w in r["windows"] or []:
                rec = w["record"] or {}
                c = rec.get("coverage") or {}
                windows.append({"window_id": w["window_id"], "canonical_start": w["start"], "canonical_end": w["end"], "n_words": w["n_words"],
                                "construction": w["kind"], "status": rec.get("status", "missing"), "score": rec.get("score"),
                                "raw_outputs": rec.get("raw_outputs"), "n_tokens_model_visible": rec.get("n_tokens"),
                                "model_input_transform": "editlens clean_text@7d615e3 applied to canonical[start:end]",
                                "model_input_sha256": rec.get("input_fingerprint"),
                                "coverage": {"mapping_status": c.get("mapping_status"), "mapping_reason": c.get("mapping_reason"),
                                             "deleted_by_preprocessing": c.get("deleted_by_preprocessing"),
                                             "preprocessing_steps_fired": c.get("preprocessing_steps_fired"),
                                             "truncated": c.get("truncated")},
                                "error": rec.get("error")})
                secs += rec.get("seconds_per_item") or 0.0
            n_tr = sum(1 for w in windows if w["coverage"]["truncated"])
            deleted = [iv for w in windows for iv in (w["coverage"]["deleted_by_preprocessing"] or [])]
            cov = {"canonical_window_coverage": "complete (windows span every non-whitespace code point by construction)",
                   "n_windows": len(windows), "n_windows_truncated": n_tr,
                   "preprocessing_deleted_canonical": deleted, "mapping_exact_all_windows": all(w["coverage"]["mapping_status"] == "exact" for w in windows)}
            if r.get("partial_max_lower_bound") is not None:
                warn.append("aggregate_incomplete_lower_bound_available")
            timing = {"model_seconds_sum": secs}
            inp_fp = sha256_text("|".join(w["model_input_sha256"] or "" for w in windows))
            raw = None
        else:
            rec = r["native"] or {}
            c = rec.get("coverage") or {}
            raw = rec.get("raw_outputs")
            cov = {"truncated": c.get("truncated"), "n_tokens_full": c.get("n_tokens_full"), "n_tokens_used": c.get("n_tokens_used"),
                   "model_visible_canonical": c.get("model_visible_canonical"), "first_unread_canonical": c.get("first_unread_canonical"),
                   "preprocessing_deleted_canonical": c.get("deleted_by_preprocessing"), "mapping_status": c.get("mapping_status"),
                   "mapping_reason": c.get("mapping_reason"), "preprocessing_steps_fired": c.get("preprocessing_steps_fired")}
            timing = {"model_seconds_sum": rec.get("seconds_per_item")}
            cache, inp_fp = rec.get("cache_key"), rec.get("input_fingerprint")
        out.append({
            "prediction_id": f"{r['example_id']}|{model_id}|{protocol_id}", "example_id": r["example_id"], "model_id": model_id,
            "protocol_id": protocol_id, "role": ex_role[r["example_id"]], "status": status,
            "status_detail": r["status"] if r["status"] != "ok" else None,
            "score": r["score"] if r["status"] == "ok" else None, "score_semantics": score_semantics,
            "partial_max_lower_bound": r.get("partial_max_lower_bound"),
            "raw_outputs": raw, "windows": windows, "native_span_predictions": None, "coverage": cov,
            "operating_point_decisions": dec, "timing": timing, "cache_key": cache, "input_fingerprint": inp_fp,
            "error_id": (f"err:{model_id}:{protocol_id}:{r['status']}" if r["status"] in ("error", "missing") else None),
            "warnings": warn})
    return out


# ------------------------------------------------------------------ main

def load_external_thresholds(path: str | None, run: dict) -> list[dict]:
    """Thresholds calibrated outside this benchmark (e.g. on your own held-out calibration set).

    YAML: {source: "<where/how calibrated>", thresholds: {<model_id>: {<protocol_id>: {<name>: <float>}}}}
    They become operating points with IDs 'external:<name>' (never an official calibration ID), use the
    same strict `score > threshold` rule, and carry no calibration statistics from this benchmark.
    """
    if not path:
        return []
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    out = []
    for mid, protos in (cfg.get("thresholds") or {}).items():
        for pid, named in protos.items():
            for name, thr in named.items():
                if not isinstance(thr, (int, float)) or not np.isfinite(thr):
                    raise SystemExit(f"external threshold {mid}/{pid}/{name} is not a finite number")
                out.append({"calibration_id": f"{mid}|{pid}|external:{name}", "model_id": mid, "protocol_id": pid,
                            "operating_point_id": f"external:{name}", "primary": False, "task": "external_threshold",
                            "target_fpr": None, "comparator": ">", "threshold": float(thr), "status": "ok",
                            "reasons": [f"externally supplied: {cfg.get('source', 'unspecified source')}"],
                            "calibration_example_ids_sha256": "external", "n_selected": 0, "n_eligible": 0, "n_scored": 0,
                            "n_failed": 0, "false_positives": None, "actual_fpr": None, "tied_scores_at_threshold": None,
                            "selection_algorithm": "external (not calibrated on this benchmark)", "external_source_file": str(path)})
    return out


def evaluate_run(bench: str, run_id: str | None = None, external_thresholds: str | None = None) -> Path:
    from aidet_eval.results.export import export_full

    bench_dir = Path(bench)
    run_id = run_id or default_run_id(bench_dir)
    rd = paths.RUNS / run_id
    run = read_json(rd / "run.json")
    if not (rd / "calibration.json").exists():
        from aidet_eval.calibrate import calibrate_run

        calibrate_run(bench, run_id)
    cal = read_json(rd / "calibration.json")
    cal["operating_points"] = cal["operating_points"] + load_external_thresholds(external_thresholds, run)
    ex = _prep_examples(bench_dir)
    ex_role = dict(zip(ex["example_id"], ex["role"]))
    all_ids = ex["example_id"].tolist()
    eval_ex = ex[ex["role"] == "evaluation"].copy()
    all_cells, all_preds, comparisons, errors = [], [], [], []
    flags_store: dict[tuple, pd.DataFrame] = {}
    timing_rows = []
    for mid in run["requested_models"]:
        mstat = read_json(rd / "models" / f"{mid}.json") if (rd / "models" / f"{mid}.json").exists() else {"status": "not_run"}
        sem = (mstat.get("describe") or {}).get("score_semantics", "unknown")
        for pid in run["requested_protocols"]:
            ops = [o for o in cal["operating_points"] if o["model_id"] == mid and o["protocol_id"] == pid]
            dp = doc_predictions(rd, bench_dir, mid, pid, all_ids)
            all_preds += prediction_records(dp, ex_role, mid, pid, ops, sem)
            st = dp["status"].value_counts().to_dict()
            for s_, n in st.items():
                if s_ in ("error", "missing"):
                    errors.append({"error_id": f"err:{mid}:{pid}:{s_}", "model_id": mid, "protocol_id": pid, "kind": s_, "count": int(n),
                                   "example_ids": sorted(dp.loc[dp["status"] == s_, "example_id"])[:500],
                                   "message": "inference failed or prediction missing for at least one required input"})
            cb = CellBuilder(eval_ex, dp, mid, pid, ops, f"exposure__{mid}" if f"exposure__{mid}" in eval_ex else None)
            build_cells(cb)
            all_cells += cb.cells
            flags_store[(mid, pid)] = cb.df[["example_id", "status", "score"]].set_index("example_id")
            # timing per panel
            sh = pd.read_json(rd / "predictions" / mid / f"{pid}.jsonl", lines=True) if (rd / "predictions" / mid / f"{pid}.jsonl").exists() else pd.DataFrame()
            if len(sh):
                sh = sh.drop_duplicates("input_id", keep="last")
                secs = sh.groupby("example_id")["seconds_per_item"].sum()
                for p in sorted({q for ps in ex["panels"] for q in ps}):
                    ids = ex.loc[ex["panels"].map(lambda ps: p in ps), "example_id"]
                    s = secs.reindex(ids).dropna()
                    timing_rows.append({"model_id": mid, "protocol_id": pid, "panel_id": p, "n_documents": int(len(s)),
                                        "model_seconds_total": float(s.sum()),
                                        "documents_per_model_second": float(len(s) / s.sum()) if s.sum() > 0 else None,
                                        "note": "GPU forward time attributed per item (batch wall time / batch size); excludes load/tokenisation"})
            log_event("evaluate_cells", model=mid, protocol=pid, n_cells=len(cb.cells))
    # paired comparisons between models (same protocol & operating point), on common scored IDs
    models = run["requested_models"]
    for pid in run["requested_protocols"]:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                if (a, pid) not in flags_store or (b, pid) not in flags_store:
                    continue
                comparisons += _compare(eval_ex, flags_store, cal, a, b, pid)
    timing_rows = nan_to_none(timing_rows)
    return export_full(bench_dir, rd, run, cal, ex, all_preds, all_cells, comparisons, errors, timing_rows, TASKS)


def _compare(ex, flags_store, cal, a, b, pid) -> list[dict]:
    out = []
    A, B = flags_store[(a, pid)], flags_store[(b, pid)]
    for op_id in sorted({o["operating_point_id"] for o in cal["operating_points"]}):
        ta = next((o for o in cal["operating_points"] if o["model_id"] == a and o["protocol_id"] == pid and o["operating_point_id"] == op_id), None)
        tb = next((o for o in cal["operating_points"] if o["model_id"] == b and o["protocol_id"] == pid and o["operating_point_id"] == op_id), None)
        if not ta or not tb or ta["threshold"] is None or tb["threshold"] is None:
            continue
        specs = []
        for p, metric_name, cls in (("audited_human_core", "human_fpr", "human"), ("student_human", "human_fpr", "human"),
                                    ("modern_human_instructions", "human_fpr", "human"),
                                    ("benchmark_source_matched_human_references", "human_fpr", "human"),
                                    ("fully_ai_core", "fully_ai_recall", "ai"), ("fully_ai_noncore", "fully_ai_recall", "ai"),
                                    ("controlled_replacement", "mixed_recall", "mixed"), ("controlled_replacement_noncore", "mixed_recall", "mixed"),
                                    ("aitdna_real_cowriting", "mixed_recall", "mixed"), ("aitdna_real_cowriting", "human_fpr", "human"),
                                    ("beemo_matched_editing", "mixed_recall", "mixed"), ("beemo_matched_editing", "fully_ai_recall", "ai"),
                                    ("beemo_matched_editing", "human_fpr", "human"), ("damasha_tagged_mixed_unknown_mechanism", "mixed_recall", "mixed"),
                                    ("heterogeneous_block_replacement", "mixed_recall", "mixed"), ("heterogeneous_matched_human_controls", "human_fpr", "human"),
                                    ("lamp_human_edited_ai", "fully_ai_recall", "ai"), ("lamp_human_edited_ai", "mixed_recall", "mixed")):
            if not (ex["panels"].map(lambda ps, p=p: p in ps)).any():
                continue
            base = ex[ex["panels"].map(lambda ps: p in ps) & (ex["authorship"] == cls)]
            specs.append((p, metric_name, None, base))
            if p in ("audited_human_core", "fully_ai_core", "controlled_replacement", "heterogeneous_block_replacement"):
                for dm in sorted(base["domain"].dropna().unique()):
                    specs.append((p, metric_name, dm, base[base["domain"] == dm]))
        for p, metric_name, dm, base in specs:
            ids = base["example_id"]
            sa, sb = A.reindex(ids), B.reindex(ids)
            common = ids[(sa["status"] == "ok").to_numpy() & (sb["status"] == "ok").to_numpy()]
            fa = (A.loc[common, "score"].to_numpy(dtype=float) > ta["threshold"])
            fb = (B.loc[common, "score"].to_numpy(dtype=float) > tb["threshold"])
            g = base.set_index("example_id").loc[common, "independence_group_id"].to_numpy()
            diff = M.paired_difference(fa, fb, g)
            out.append({"comparison_id": fingerprint([a, b, pid, op_id, p, metric_name, dm])[:24],
                        "left": {"run_id": None, "model_id": a}, "right": {"run_id": None, "model_id": b},
                        "protocol_id": pid, "operating_point_id": op_id, "calibration_policy": "per-model thresholds from identical calibration IDs",
                        "task_or_panel": p, "slice": {"domain": dm} if dm else {}, "metric": metric_name,
                        "n_eligible": int(len(ids)), "n_common_scored": int(len(common)),
                        "common_example_ids_sha256": sha256_text("\n".join(sorted(common))),
                        "left_estimate": float(fa.mean()) if len(common) else None, "right_estimate": float(fb.mean()) if len(common) else None,
                        "difference_pp": diff.get("difference_pp"), "uncertainty": diff.get("ci_pp"), "status": diff["status"],
                        "relative_change": (float(fa.mean() / fb.mean() - 1) if len(common) and fb.mean() > 0 else None),
                        "notes": "difference in percentage points (left - right); paired on common scored examples"})
    return out
