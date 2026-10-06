"""Write results.full.json (+ .gz), derive results.summary.json deterministically, schemas, CSV, Markdown.

The summary is a pure function of the full document bytes (`summarise`), so `export --run` can
regenerate it from results.full.json alone and verify reproducibility.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata as md
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aidet_eval import paths
from aidet_eval.build import jl
from aidet_eval.calibrate import threshold_for
from aidet_eval.results.schema import SCHEMA_VERSION, FullResult, SummaryResult, write_schemas
from aidet_eval.util import file_hashes, log_event, nan_to_none, read_json, sha256_text, utcnow

PKGS = ["torch", "transformers", "peft", "bitsandbytes", "tokenizers", "accelerate", "datasets", "huggingface-hub", "pyarrow",
        "pandas", "numpy", "scipy", "scikit-learn", "pydantic", "emoji", "xxhash"]


# =========================================================================== assemble

def _example_record(r: dict) -> dict:
    ai = jl(r["ai_char_spans"])
    caveats = []
    if r.get("artefact_flags"):
        caveats += jl(r["artefact_flags"]) or []
    hp_ev = jl(r["human_provenance_evidence"])
    if hp_ev and hp_ev.get("limitations"):
        caveats.append(hp_ev["limitations"])
    exposure = {k.split("__", 1)[1]: r[k] for k in r if k.startswith("exposure__")}
    return {
        "example_id": r["example_id"],
        "source": {"dataset_id": r["dataset_id"], "dataset_key": r["dataset_key"], "revision": r["revision"], "config": r["config"],
                   "original_split": r["original_split"], "original_row_id": r["original_row_id"],
                   "parent_ids": jl(r["parent_ids"]) or [], "source_urls": jl(r["source_urls"]) or []},
        "source_group_id": r["source_group_id"], "leakage_group_id": r["leakage_group_id"],
        "independence_group_id": r["independence_group_id"], "role": r["role"], "panel_ids": jl(r["panel_ids"]),
        "domain": r["domain"], "subdomain": r["subdomain"], "language": r["language"],
        "original_labels": jl(r["original_labels"]) or {},
        "truth": {"authorship": r["authorship"], "authorship_basis": r["authorship_basis"], "ai_involvement": r["ai_involvement"],
                  "mix_mechanism": r["mix_mechanism"], "generator": r["generator"],
                  "generation_parameters": jl(r["generation_parameters"]), "ai_char_spans": ai,
                  "human_char_spans": jl(r["human_char_spans"]), "unknown_char_spans": jl(r["unknown_char_spans"]),
                  "annotation_completeness": r["annotation_completeness"], "annotation_provenance": r["annotation_provenance"],
                  "ai_fraction_char": r["ai_fraction_char"], "span_count": None if ai is None else len(ai),
                  "span_lengths": None if ai is None else [b - a for a, b in ai]},
        "human_provenance": {"category": r["human_provenance_category"], "tier": r["provenance_tier"], "evidence": hp_ev,
                             "verified_dates": jl(r["verified_dates"]), "eligibility": r["eligibility"], "caveats": caveats},
        "text": r["text"], "text_sha256": r["text_sha256"], "word_count": int(r["word_count"]), "char_count": int(r["char_count"]),
        "length_bin": r["length_bin"], "parent_excerpt": jl(r["parent_excerpt"]), "canonicalisation": jl(r["canonicalisation"]) or {},
        "rights": jl(r["rights"]) or {}, "training_exposure": exposure, "artefact_flags": jl(r["artefact_flags"]) or [],
        "extensions": {**(jl(r["extensions"]) or {}), "eligibility_reasons": jl(r["eligibility_reasons"]) or [],
                       "original_text_sha256": r["original_text_sha256"],
                       "slices": {k: r.get(k) for k in ("ai_fraction_bin", "block_count_bin", "block_length_bin", "block_position")}},
    }


def _sources() -> list[dict]:
    inv = read_json(paths.INVENTORY / "inventory.json")
    out = []
    for e in inv["sources"]:
        st = e["acquisition_status"] or "not_started"
        out.append({"source_id": e["source_id"], "name": e["name"], "authoritative_location": str(e["resolved_location"]),
                    "catalogue_upstream": e["upstream"], "pinned_revision": e["resolved_revision"],
                    "repository_migration": e.get("repository_migration"),
                    "acquisition_status": st, "download_completeness": e["download_completeness"],
                    "acquisition_manifest": e["acquisition_manifest"], "n_files": e["n_files"],
                    "reported_rows": e["reported_counts"] or None, "observed_rows": e["measured_counts"].get("rows_total"),
                    "observed_rows_by_file": e["measured_counts"].get("rows_by_file"),
                    "eligibility_status": e["decision_eligibility"], "provenance_assessment": e["decision_provenance"],
                    "rights_restrictions": e["decision_rights"], "continuation_command": e["continuation_command"],
                    "inventory_detail": "data/inventory/inventory.json", "warnings": e.get("notes") or []})
    return out


def _packages() -> dict:
    out = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _models(rd: Path, run: dict, bench_manifest: dict) -> list[dict]:
    out = []
    for mid in run["requested_models"]:
        p = rd / "models" / f"{mid}.json"
        m = read_json(p) if p.exists() else {"status": "not_run"}
        d = m.get("describe") or {}
        out.append({"model_id": mid, "display_name": d.get("display_name", mid), "status": m.get("status"),
                    "error": m.get("error"), "checkpoint": d.get("checkpoint"), "checkpoint_revision": d.get("checkpoint_revision"),
                    "base_model": d.get("base_model"), "base_revision": d.get("base_revision"), "tokenizer": d.get("tokenizer"),
                    "tokenizer_revision": d.get("tokenizer_revision"), "file_sha256": d.get("file_sha256"),
                    "adapter": d.get("adapter"), "adapter_version": d.get("adapter_version"), "loader": d.get("loader"),
                    "head": d.get("head"), "precision": d.get("precision"), "n_buckets": d.get("n_buckets"),
                    "device": (m.get("runtime") or {}).get("device"), "runtime": m.get("runtime"),
                    "reference_implementation": d.get("reference_implementation"), "preprocessing": d.get("preprocessing"),
                    "score_semantics": d.get("score_semantics", "unknown"), "score_formula": d.get("score_formula"),
                    "score_range": d.get("score_range"), "score_orientation": d.get("score_orientation"),
                    "not_a_calibrated_probability": d.get("not_a_calibrated_probability"),
                    "capabilities": {"native_spans": False, "class_probabilities": "ordinal_bucket_distribution",
                                     "document_or_window_scorer": True},
                    "training_exposure_audit": bench_manifest.get("training_exposure", {}).get("pangram_editlens"),
                    "libraries": d.get("libraries"), "licence": d.get("licence"),
                    "model_spec_fingerprint": m.get("model_fingerprint")})
    return out


def export_full(bench_dir: Path, rd: Path, run: dict, cal: dict, ex: pd.DataFrame, preds: list[dict], cells: list[dict],
                comparisons: list[dict], errors: list[dict], timing: list[dict], tasks: list[dict]) -> Path:
    man = read_json(bench_dir / "manifest.json")
    prot = read_json(bench_dir / "windows" / "protocols.json")
    examples = [_example_record(r) for r in ex.to_dict("records")]
    parity = read_json(paths.ARTIFACTS / "parity" / "parity_report.json") if (paths.ARTIFACTS / "parity" / "parity_report.json").exists() else None
    tests = read_json(paths.ARTIFACTS / "tests" / "pytest_summary.json") if (paths.ARTIFACTS / "tests" / "pytest_summary.json").exists() else None
    # execution completeness
    exp_counts = {}
    for mid in run["requested_models"]:
        for pid in run["requested_protocols"]:
            ps = [p for p in preds if p["model_id"] == mid and p["protocol_id"] == pid]
            c = pd.Series([p["status"] for p in ps]).value_counts().to_dict() if ps else {}
            exp_counts[f"{mid}|{pid}"] = {"expected": len(ex), "records": len(ps), **{k: int(v) for k, v in c.items()},
                                          "inference_complete": len(ps) == len(ex) and c.get("error", 0) == 0}
    shard_times = []
    for f in (rd / "predictions").rglob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                pass
            try:
                shard_times.append(json.loads(line)["at"])
            except Exception:  # noqa: BLE001
                pass
    panel_defs = []
    for pid in sorted({p for e in examples for p in e["panel_ids"]}):
        mem = [e for e in examples if pid in e["panel_ids"]]
        panel_defs.append({"panel_id": pid, "n_members": len(mem), "role": mem[0]["role"] if mem else None,
                           "target_labels": sorted({e["truth"]["authorship"] for e in mem}),
                           "controls": ("independently_audited" if pid in ("audited_human_core", "student_human") else
                                        "curator_attested" if pid == "modern_human_instructions" else
                                        "benchmark_provided" if pid == "benchmark_source_matched_human_references" else
                                        "logged_session" if pid == "aitdna_real_cowriting" else "panel-native"),
                           "membership": "explicit example_ids in examples[].panel_ids",
                           "denominators": "per metric: scored examples of the target class in the cell"})
    shortfalls = [c for c in man["cells"] if c.get("missing", 0) > 0]
    full = {
        "schema_version": SCHEMA_VERSION, "artifact_kind": "aidet_eval.results.full", "data_origin": run.get("data_origin", "measured"),
        "run_id": run["run_id"], "status": "partial",
        "benchmark": {
            "benchmark_id": man["benchmark_id"], "benchmark_version": man["benchmark_version"], "profile": man["profile"],
            "purpose": man["purpose"], "release_role": man["release_role"], "seed": man["seed"],
            "manifest_sha256": file_hashes(bench_dir / "manifest.json")["sha256"], "benchmark_hash": man["benchmark_hash"],
            "fingerprints": {**man["fingerprints"], "window_manifest_sha256": prot["protocols"][0]["window_manifest_sha256"]},
            "calibration_membership_sha256": man["fingerprints"]["calibration_membership_sha256"],
            "panel_definitions": panel_defs, "task_definitions": tasks, "selection_targets_and_realised": man["cells"],
            "realised_counts": man["counts"], "shortfalls": shortfalls, "domain_mappings": man["domain_mappings"],
            "provenance_eligibility_rules": {"trusted_human_view": "provenance_tier == independently_audited (pre-LLM releases, PERSUADE) or curator_attested (No Robots), shown separately",
                                             "benchmark_provided_view": "RAID/LLMTrace human rows linked to selected AI items; never promoted automatically"},
            "grouping_policy": {"leakage": "union-find over native keys, match-fingerprints (>=8 tokens) and MinHash near-duplicates",
                                "independence": "writer/author/session/source keys merged with leakage groups", "overlap": man["overlap"]},
            "excerpt_policy": man["excerpt_policy"], "offset_convention": man["offset_convention"], "length_function": man["length_function"],
            "length_bins": man["length_bins"], "ai_fraction_bins": man["ai_fraction_bins"], "reserve_sources": man["reserve_sources"],
            "selection_policy": man["selection_policy"], "benchmark_dir": str(bench_dir)},
        "run": {"run_id": run["run_id"], "created_at": run["created_at"], "started_at": run["created_at"],
                "finished_at": max(shard_times) if shard_times else None, "exported_at": None,
                "commands": run.get("invocations"), "resolved_configuration": {"models_yaml_sha256": file_hashes(paths.CONFIGS / "models.yaml")["sha256"],
                                                                                "sources_yaml_sha256": file_hashes(paths.CONFIGS / "sources.yaml")["sha256"],
                                                                                "plan_sha256": man["plan_sha256"]},
                "code": run.get("git"), "dependency_lock_sha256": run.get("dependency_lock_sha256"), "packages": _packages(),
                "platform": run.get("platform"), "hardware": [m for m in (read_json(p) for p in (rd / "models").glob("*.json"))][0].get("runtime") if list((rd / "models").glob("*.json")) else None,
                "reproducibility": {"benchmark_seed": man["seed"], "bootstrap_seed": 42, "bootstrap_replicates": 2000,
                                    "gpu_numerics": "bf16 autocast (+NF4 for Llama) on CUDA; scores reproducible up to kernel nondeterminism"},
                "requested": {"models": run["requested_models"], "protocols": run["requested_protocols"],
                              "panels": sorted({p for e in examples for p in e["panel_ids"]})},
                "execution_counts": exp_counts, "debug_limit": run.get("debug_limit"),
                "inference_completed": {k: v["inference_complete"] for k, v in exp_counts.items()},
                "export_completed": True,
                "loader_parity": ({"scope": parity["scope"], "models": [{k: m[k] for k in m if k != "per_text"} for m in parity["models"]]} if parity else None),
                "tests": tests},
        "sources": _sources(),
        "models": _models(rd, run, man),
        "protocols": prot["protocols"],
        "calibrations": cal["operating_points"],
        "examples": examples,
        "predictions": preds,
        "aggregates": cells,
        "comparisons": comparisons,
        "errors": errors,
        "validation": {"structural_ok": False, "semantic_checks": [], "ok": False},
        "warnings": [],
        "extensions": {"timing_by_panel": timing, "window_manifest_summary": {k: prot[k] for k in ("n_windows", "n_examples", "windows_per_example")}},
    }
    full["warnings"] += _standing_warnings(full)
    full = nan_to_none(full)
    checks = semantic_validate(full)
    full["validation"] = {"structural_ok": True, "semantic_checks": checks, "ok": all(c["ok"] for c in checks)}
    all_inf = all(v["inference_complete"] for v in exp_counts.values())
    cal_ok = all(c["status"] == "ok" for c in cal["operating_points"])
    profile_ok = man["profile"] == "full" and not run.get("debug_limit")
    full["status"] = "complete" if (all_inf and cal_ok and full["validation"]["ok"] and profile_ok) else "partial"
    if full["status"] != "complete":
        full["warnings"].append({"code": "run_partial", "message": f"inference_complete={all_inf} calibration_ok={cal_ok} validation_ok={full['validation']['ok']} full_profile={profile_ok}"})
    try:
        FullResult.model_validate(full)
    except Exception as e:  # noqa: BLE001
        full["validation"]["structural_ok"] = False
        full["validation"]["ok"] = False
        full["status"] = "partial"
        full["warnings"].append({"code": "structural_validation_failed", "message": str(e)[:2000]})
    return write_outputs(rd, full)


def _standing_warnings(full: dict) -> list[dict]:
    w = [{"code": "score_semantics", "message": "Open Pangram scores are ordinal AI-intervention extent signals (expected bucket / (K-1)); not calibrated probabilities of AI presence and not AI-character fractions. No Brier/ECE is computed."},
         {"code": "native_localisation_unavailable", "message": "Both baselines are document/window scorers; native span metrics are unavailable. Window scores are contextual diagnostics only."},
         {"code": "shared_windows_aggregation", "message": "shared_windows_v1 uses max(window score), a harness wrapper predeclared before scoring; longer documents have more windows, so FPR/recall are also reported by window count."},
         {"code": "no_single_overall_score", "message": "No official overall quality number is produced; compare domain-specific recall against domain-specific human FPR at frozen operating points."},
         {"code": "creative_ai_cells_missing", "message": "No evidence-supported creative-writing AI or mixed source exists in the catalogue releases; creative fully-AI and replacement cells are explicitly empty (plot summaries are reported separately as narrative_summary)."},
         {"code": "in_family_panel", "message": "editlens_in_family_reference comes from the EditLens test split (same pipeline as the baselines' training data); not independent evidence."},
         {"code": "xsum_training_overlap", "message": "233 XSum documents occur in EditLens training data; XSum is marked possible_overlap and those documents known_overlap. An 'excluding_known_training_overlap' view is provided."},
         {"code": "tokenised_corpora", "message": "Scientific Papers and WikiText human texts are pre-tokenised by their original releases (flagged artefacts); research/expository human FPR partly reflects this format."},
         {"code": "raid_training_release", "message": "RAID rows come from the labelled training release, not unseen evaluation for RAID-trained detectors."},
         {"code": "fill_gaps_single_generator", "message": "All controlled sentence-replacement mixtures (LLMTrace fill_gaps) were generated by gemini-2.5-flash; generator balance is not possible for this panel."},
         {"code": "development_release", "message": "benchmark-v1 is a development/regression benchmark that will be inspected repeatedly; it is not a confirmation release."}]
    return w


# =========================================================================== validate

def semantic_validate(full: dict) -> list[dict]:
    checks = []

    def chk(name, ok, detail=None):
        checks.append({"invariant": name, "ok": bool(ok), "detail": detail})

    exs = full["examples"]
    ids = [e["example_id"] for e in exs]
    chk("unique_example_ids", len(ids) == len(set(ids)))
    by_id = {e["example_id"]: e for e in exs}
    bad_hash = [e["example_id"] for e in exs if e["text"] is not None and sha256_text(e["text"]) != e["text_sha256"]]
    chk("text_hashes_match", not bad_hash, bad_hash[:5])
    bad_iv = []
    for e in exs:
        n = e["char_count"]
        if e["text"] is not None and len(e["text"]) != n:
            bad_iv.append(e["example_id"])
        for a, b in (e["truth"]["ai_char_spans"] or []):
            if not (0 <= a < b <= n):
                bad_iv.append(e["example_id"])
    chk("intervals_valid_codepoints", not bad_iv, bad_iv[:5])
    unk_as_human = [e["example_id"] for e in exs if e["truth"]["annotation_completeness"] != "complete" and e["truth"]["human_char_spans"]
                    and e["source"]["dataset_key"] not in ("aitdna",)]
    chk("unknown_gold_not_treated_as_human", not unk_as_human, unk_as_human[:5])
    g_role = {}
    leak = []
    for e in exs:
        r = g_role.setdefault(e["leakage_group_id"], e["role"])
        if r != e["role"]:
            leak.append(e["leakage_group_id"])
    chk("calibration_evaluation_leakage_disjoint", not leak, leak[:5])
    keys = [(p["example_id"], p["model_id"], p["protocol_id"]) for p in full["predictions"]]
    chk("unique_prediction_keys", len(keys) == len(set(keys)))
    chk("predictions_resolve_to_examples", all(p["example_id"] in by_id for p in full["predictions"]))
    expected = len(exs) * len(full["models"]) * len(full["protocols"])
    chk("every_expected_prediction_represented", len(keys) == expected, {"expected": expected, "records": len(keys)})
    # numeric invariants
    bad_num = []
    for p in full["predictions"]:
        outs = [w["raw_outputs"] for w in (p["windows"] or []) if w["raw_outputs"]] + ([p["raw_outputs"]] if p["raw_outputs"] else [])
        scores = [w["score"] for w in (p["windows"] or []) if w["raw_outputs"]] + ([p["score"]] if p["raw_outputs"] else [])
        for ro, sc in zip(outs, scores):
            pr = np.array(ro["bucket_probabilities"])
            k = len(pr)
            if abs(pr.sum() - 1) > 1e-6 or not np.isfinite(pr).all() or sc is None or abs(float(pr @ np.arange(k) / (k - 1)) - sc) > 1e-9:
                bad_num.append(p["prediction_id"])
                break
        if p["status"] == "ok" and p["protocol_id"] == "shared_windows_v1":
            ws = [w["score"] for w in p["windows"] if w["status"] == "ok"]
            if not ws or abs(max(ws) - p["score"]) > 0 or any(w["status"] in ("error", "missing") for w in p["windows"]):
                bad_num.append(p["prediction_id"])
            if any(w["coverage"].get("truncated") for w in p["windows"]):
                bad_num.append(p["prediction_id"] + ":truncated_window")
    chk("scores_finite_probabilities_normalised_formula_and_max_aggregation", not bad_num, bad_num[:5])
    # thresholds and decisions reproducible
    cal_ids = {e["example_id"] for e in exs if e["role"] == "calibration"}
    bad_thr = []
    for c in full["calibrations"]:
        if c["task"] == "external_threshold":
            continue  # not derived from this benchmark's calibration documents
        sc = [p["score"] for p in full["predictions"] if p["model_id"] == c["model_id"] and p["protocol_id"] == c["protocol_id"]
              and p["example_id"] in cal_ids and p["status"] == "ok"]
        t = threshold_for(sc, c["target_fpr"])
        if t.get("threshold") != c["threshold"]:
            bad_thr.append(c["calibration_id"])
    chk("thresholds_reproducible_from_calibration_predictions_only", not bad_thr, bad_thr)
    thr = {c["calibration_id"]: c["threshold"] for c in full["calibrations"]}
    bad_dec = [p["prediction_id"] for p in full["predictions"] for d in p["operating_point_decisions"]
               if (d["flag_ai"] is not None) != (p["status"] == "ok" and thr[d["calibration_id"]] is not None)
               or (d["flag_ai"] is not None and d["flag_ai"] != (p["score"] > thr[d["calibration_id"]]))]
    chk("decisions_reproducible_strict_comparator", not bad_dec, bad_dec[:5])
    # metric reconciliation on panel-level cells
    pred_ix = {(p["example_id"], p["model_id"], p["protocol_id"]): p for p in full["predictions"]}
    bad_rec = []
    for c in full["aggregates"]:
        if c["slice"].get("view") != "all" or set(c["slice"]) != {"view"} or c["operating_point_id"] is None or c["task_id"]:
            continue
        cid = f"{c['model_id']}|{c['protocol_id']}|{c['operating_point_id']}"
        members = [e for e in exs if c["panel_id"] in e["panel_ids"]]
        hum = [pred_ix[(e["example_id"], c["model_id"], c["protocol_id"])] for e in members if e["truth"]["authorship"] == "human"]
        ok = [p for p in hum if p["status"] == "ok"]
        m = c["metrics"]["human_fpr"]
        if thr[cid] is None:
            if m["value"] is not None or m["status"] != "not_computed":
                bad_rec.append(c["cell_id"] + ":blocked_threshold_but_value")
            continue
        fp = sum(1 for p in ok if p["score"] > thr[cid])
        if hum and (m["numerator"] != fp or m["denominator"] != len(ok)):
            bad_rec.append(c["cell_id"])
        if m["value"] is not None and m["percent"] is not None and abs(m["percent"] - 100 * m["value"]) > 1e-9:
            bad_rec.append(c["cell_id"] + ":percent")
    chk("metric_populations_reconcile_with_predictions", not bad_rec, bad_rec[:5])
    chk("native_span_metrics_unavailable_for_baselines", all(p["native_span_predictions"] is None for p in full["predictions"]))
    nan_found = _has_nonfinite(full)
    chk("finite_numbers_only", not nan_found)
    return checks


def _has_nonfinite(o) -> bool:
    if isinstance(o, float):
        return not math.isfinite(o)
    if isinstance(o, dict):
        return any(_has_nonfinite(v) for v in o.values())
    if isinstance(o, list):
        return any(_has_nonfinite(v) for v in o)
    return False


# =========================================================================== summary

def summarise(full: dict, full_sha: str) -> dict:
    """Deterministic reduction of the full document (no texts, no per-window arrays, no new timestamps)."""
    b = full["benchmark"]
    cov = []
    for k, v in full["run"]["execution_counts"].items():
        mid, pid = k.split("|")
        cov.append({"model_id": mid, "protocol_id": pid, **v})
    return {
        "schema_version": SCHEMA_VERSION, "artifact_kind": "aidet_eval.results.summary", "data_origin": full["data_origin"],
        "run_id": full["run_id"], "status": full["status"], "full_result_sha256": full_sha,
        "benchmark": {k: b[k] for k in ("benchmark_id", "benchmark_version", "profile", "purpose", "release_role", "seed",
                                        "manifest_sha256", "benchmark_hash", "fingerprints", "calibration_membership_sha256",
                                        "realised_counts", "shortfalls", "task_definitions", "domain_mappings", "length_bins",
                                        "ai_fraction_bins", "offset_convention")}
                     | {"panel_definitions": b["panel_definitions"]},
        "run": {k: full["run"][k] for k in ("run_id", "created_at", "finished_at", "code", "dependency_lock_sha256", "packages",
                                            "hardware", "reproducibility", "requested", "execution_counts", "inference_completed",
                                            "loader_parity", "tests", "debug_limit")},
        "models": [{k: m[k] for k in ("model_id", "display_name", "status", "checkpoint", "checkpoint_revision", "base_model",
                                      "base_revision", "precision", "n_buckets", "device", "score_semantics", "score_formula",
                                      "score_range", "score_orientation", "not_a_calibrated_probability", "capabilities",
                                      "training_exposure_audit", "model_spec_fingerprint", "loader", "head")} for m in full["models"]],
        "protocols": [{k: p[k] for k in p if k not in ("budgets",)} | {"budgets": p.get("budgets")} for p in full["protocols"]],
        "operating_points": [{k: c[k] for k in c if k != "failed_example_ids"} for c in full["calibrations"]],
        "source_status": [{k: s[k] for k in ("source_id", "name", "acquisition_status", "download_completeness", "pinned_revision",
                                             "reported_rows", "observed_rows", "eligibility_status", "rights_restrictions",
                                             "continuation_command")} for s in full["sources"]],
        "cells": full["aggregates"],
        "comparisons": full["comparisons"],
        "coverage": cov,
        "warnings": full["warnings"],
        "extensions": {"timing_by_panel": full["extensions"].get("timing_by_panel"), "validation": full["validation"],
                       "errors": [{k: e[k] for k in e if k != "example_ids"} | {"n_example_ids": len(e.get("example_ids", []))} for e in full["errors"]]},
    }


def _dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=False).encode("utf-8")


def write_outputs(rd: Path, full: dict) -> Path:
    full["run"]["exported_at"] = full["run"].get("exported_at") or utcnow()
    fb = _dump(full)
    (rd / "results.full.json").write_bytes(fb)
    full_sha = hashlib.sha256(fb).hexdigest()
    with gzip.GzipFile(rd / "results.full.json.gz", "wb", mtime=0) as g:
        g.write(fb)
    summ = summarise(full, full_sha)
    SummaryResult.model_validate(summ)
    (rd / "results.summary.json").write_bytes(json.dumps(summ, ensure_ascii=False, allow_nan=False, indent=1).encode())
    write_schemas(rd)
    write_csv(rd, summ)
    write_markdown(rd, summ)
    try:
        from aidet_eval.viewer.server import index_path

        index_path(rd)  # pre-build the viewer's SQLite index so opening a run never parses the full JSON on demand
    except Exception as e:  # noqa: BLE001
        log_event("viewer_index_failed", error=str(e)[:300])
    log_event("export_done", run=str(rd), status=full["status"], full_mb=round(len(fb) / 1e6, 1), full_sha=full_sha[:16],
              validation_ok=full["validation"]["ok"])
    return rd


def export_run(run: str, redact_texts: bool = False) -> None:
    """Re-derive the summary from an existing results.full.json and verify it is reproducible."""
    rd = Path(run)
    fb = (rd / "results.full.json").read_bytes()
    full = json.loads(fb)
    sha = hashlib.sha256(fb).hexdigest()
    summ = summarise(full, sha)
    new = json.dumps(summ, ensure_ascii=False, allow_nan=False, indent=1).encode()
    old = (rd / "results.summary.json").read_bytes() if (rd / "results.summary.json").exists() else b""
    print("summary reproducible from full file" if new == old else "summary regenerated (differs from previous export)")
    (rd / "results.summary.json").write_bytes(new)
    write_csv(rd, summ)
    write_markdown(rd, summ)
    if redact_texts:
        red = dict(full)
        red["examples"] = [{**e, "text": None} for e in full["examples"]]
        red["data_origin"] = full["data_origin"]
        red["extensions"] = {**full["extensions"], "redaction": {"texts_removed": True,
                             "unavailable_capabilities": ["full-text inspection", "span highlighting", "window text display"],
                             "warning": "Contains source-derived metadata of copyrighted/student/non-commercial datasets; check licences before sharing."}}
        (rd / "results.full.redacted.json").write_bytes(_dump(red))
        print(f"wrote {rd / 'results.full.redacted.json'}")


# =========================================================================== CSV / Markdown

def write_csv(rd: Path, summ: dict) -> None:
    cols = ["cell_id", "model_id", "protocol_id", "operating_point_id", "task_id", "panel_id", "slice", "metric", "status", "value",
            "percent", "numerator", "denominator", "ci_lower", "ci_upper", "ci_method", "n_examples", "n_independence_groups", "reason",
            "selected", "scored", "error"]
    with open(rd / "results.summary.cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in summ["cells"]:
            for name, m in c["metrics"].items():
                ci = m.get("ci") or {}
                w.writerow([c["cell_id"], c["model_id"], c["protocol_id"], c["operating_point_id"], c["task_id"], c["panel_id"],
                            json.dumps(c["slice"], sort_keys=True), name, m["status"], m["value"], m["percent"], m["numerator"],
                            m["denominator"], ci.get("lower"), ci.get("upper"), ci.get("method"), m["n_examples"],
                            m["n_independence_groups"], m["reason"], c["support"]["selected"], c["support"]["scored"], c["support"]["error"]])


def _fmt(m: dict | None) -> str:
    if not m or m["value"] is None:
        return f"n/a ({m['reason']})" if m else "n/a"
    def p(x):  # adaptive precision so small nonzero rates are never displayed as 0.0%
        return f"{x:.2f}" if 0 < x < 1 else f"{x:.1f}"

    s = f"{p(m['percent'])}% ({m['numerator']}/{m['denominator']})" if m.get("percent") is not None else f"{m['value']:.3f}"
    ci = m.get("ci")
    if ci and ci.get("lower_percent") is not None:
        s += f" [{p(ci['lower_percent'])}, {p(ci['upper_percent'])}]"
        if ci.get("degenerate_all_zero"):
            s += " (degenerate bootstrap CI)"
    if m["status"] == "provisional":
        s += " PROVISIONAL"
    return s


def write_markdown(rd: Path, summ: dict) -> None:
    L = [f"# Detector evaluation report: {summ['run_id']}", "",
         f"Status: **{summ['status']}** · benchmark `{summ['benchmark']['benchmark_id']}` ({summ['benchmark']['profile']}, {summ['benchmark']['purpose']}) · "
         f"benchmark hash `{summ['benchmark']['benchmark_hash'][:16]}` · full-result sha256 `{summ['full_result_sha256'][:16]}`", "",
         "Generated from `results.summary.json` only. Rates are percentages of scored examples with counts and 95% intervals "
         "(Wilson for independent observations, independence-group cluster bootstrap otherwise). Scores are ordinal extent signals, not probabilities.", ""]
    L.append("## Operating points (frozen on human calibration documents)")
    L.append("| model | protocol | op | threshold (`score >`) | calibration FPR | n | status |")
    L.append("|---|---|---|---|---|---|---|")
    for o in summ["operating_points"]:
        fpr = "n/a" if o["actual_fpr"] is None else f"{100 * o['actual_fpr']:.2f}% ({o['false_positives']}/{o['n_scored']})"
        L.append(f"| {o['model_id']} | {o['protocol_id']} | {o['operating_point_id']} | {o['threshold']!r} | {fpr} | {o['n_scored']} | {o['status']} |")
    cells = summ["cells"]
    for pid in ("shared_windows_v1", "native_truncate_v1"):
        for op in sorted({o["operating_point_id"] for o in summ["operating_points"]}):
            L += ["", f"## {pid} · {op}", "", "| model | domain | audited human FPR | fully-AI recall | sentence-replacement recall | block-replacement recall | matched-ref human FPR |", "|---|---|---|---|---|---|---|"]
            for mid in sorted({c["model_id"] for c in cells}):
                def get(panel, dom, metric):
                    for c in cells:
                        if (c["model_id"] == mid and c["protocol_id"] == pid and c["operating_point_id"] == op and c["panel_id"] == panel
                                and c["task_id"] is None and c["slice"] == {"domain": dom, "view": "all"}):
                            return c["metrics"].get(metric)
                    return None
                for dom in ("research", "news", "reviews", "expository", "creative", "narrative_summary"):
                    L.append(f"| {mid} | {dom} | {_fmt(get('audited_human_core', dom, 'human_fpr'))} | "
                             f"{_fmt(get('fully_ai_core', dom, 'fully_ai_recall') or get('fully_ai_noncore', dom, 'fully_ai_recall'))} | "
                             f"{_fmt(get('controlled_replacement', dom, 'mixed_recall') or get('controlled_replacement_noncore', dom, 'mixed_recall'))} | "
                             f"{_fmt(get('heterogeneous_block_replacement', dom, 'mixed_recall'))} | "
                             f"{_fmt(get('benchmark_source_matched_human_references', dom, 'human_fpr'))} |")
            L += ["", "| model | panel | human FPR | fully-AI recall | mixed recall | coverage |", "|---|---|---|---|---|---|"]
            for mid in sorted({c["model_id"] for c in cells}):
                for c in cells:
                    if c["model_id"] == mid and c["protocol_id"] == pid and c["operating_point_id"] == op and c["slice"] == {"view": "all"} and c["task_id"] is None:
                        m = c["metrics"]
                        L.append(f"| {mid} | {c['panel_id']} | {_fmt(m.get('human_fpr'))} | {_fmt(m.get('fully_ai_recall'))} | {_fmt(m.get('mixed_recall'))} | {_fmt(m.get('prediction_coverage'))} |")
    L += ["", "## Warnings", ""] + [f"- **{w['code']}**: {w['message']}" for w in summ["warnings"]]
    L += ["", "## Source acquisition", "", "| source | status | observed rows | eligibility |", "|---|---|---|---|"]
    for s in summ["source_status"]:
        L.append(f"| {s['source_id']} | {s['acquisition_status']} | {s['observed_rows']} | {s['eligibility_status']} |")
    (rd / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
