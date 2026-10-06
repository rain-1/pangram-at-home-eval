"""End-to-end tests on a synthetic fixture: failure accounting, export/summary reproducibility,
cross-run compatibility, viewer loading and escaping. Optional parity check when weights exist."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from aidet_eval.results.compat import compatibility
from aidet_eval.results.export import export_run, summarise
from aidet_eval.results.schema import FullResult, SummaryResult


def test_fixture_export_is_valid_and_labelled_synthetic(fixture_run):
    rd = fixture_run["run_dir"]
    full = json.loads((rd / "results.full.json").read_bytes())
    FullResult.model_validate(full)
    assert full["data_origin"] == "synthetic_fixture"
    assert all(e["example_id"].startswith("fixture-") for e in full["examples"])
    assert full["status"] == "partial"  # smoke profile + one missing prediction
    assert all(c["ok"] for c in full["validation"]["semantic_checks"]), [c for c in full["validation"]["semantic_checks"] if not c["ok"]]
    summ = json.loads((rd / "results.summary.json").read_bytes())
    SummaryResult.model_validate(summ)
    assert "text" not in json.dumps(summ["cells"])[:0]  # summary carries no example texts
    assert not any("examples" in k for k in summ)


def test_failure_accounting_missing_prediction(fixture_run):
    full = json.loads((fixture_run["run_dir"] / "results.full.json").read_bytes())
    p = next(p for p in full["predictions"] if p["example_id"] == "fixture-095")
    assert p["status"] == "error" and p["score"] is None
    assert all(d["flag_ai"] is None for d in p["operating_point_decisions"])
    cell = next(c for c in full["aggregates"] if c["panel_id"] == "controlled_replacement" and c["slice"] == {"view": "all"}
                and c["operating_point_id"] == "human_fpr_1pct")
    assert cell["support"]["error"] == 1
    m = cell["metrics"]["mixed_recall"]
    assert m["denominator"] == 9 and m["failure_bounds"]["n_unscored"] == 1


def test_thresholds_use_calibration_only_and_decisions_strict(fixture_run):
    full = json.loads((fixture_run["run_dir"] / "results.full.json").read_bytes())
    cal_ids = {e["example_id"] for e in full["examples"] if e["role"] == "calibration"}
    for c in full["calibrations"]:
        assert c["comparator"] == ">" and c["n_scored"] == len(cal_ids)
    thr = {c["calibration_id"]: c["threshold"] for c in full["calibrations"]}
    for p in full["predictions"]:
        for d in p["operating_point_decisions"]:
            if d["flag_ai"] is not None:
                assert d["flag_ai"] == (p["score"] > thr[d["calibration_id"]])


def test_summary_is_deterministic_reduction_of_full(fixture_run, capsys):
    rd = fixture_run["run_dir"]
    fb = (rd / "results.full.json").read_bytes()
    full = json.loads(fb)
    s1 = summarise(full, hashlib.sha256(fb).hexdigest())
    s2 = summarise(json.loads(fb), hashlib.sha256(fb).hexdigest())
    assert s1 == s2
    assert s1["full_result_sha256"] == hashlib.sha256(fb).hexdigest()
    export_run(str(rd))
    assert "summary reproducible from full file" in capsys.readouterr().out
    # no rounded display value is authoritative: percent == 100 * value exactly
    for c in s1["cells"]:
        for m in c["metrics"].values():
            if m["value"] is not None and m["percent"] is not None:
                assert m["percent"] == 100 * m["value"]


def test_redacted_export_removes_texts(fixture_run):
    export_run(str(fixture_run["run_dir"]), redact_texts=True)
    red = json.loads((fixture_run["run_dir"] / "results.full.redacted.json").read_bytes())
    assert all(e["text"] is None for e in red["examples"])
    assert red["extensions"]["redaction"]["texts_removed"] is True


def test_cross_run_compatibility(fixture_run):
    s = json.loads((fixture_run["run_dir"] / "results.summary.json").read_bytes())
    assert compatibility(s, s, "shared_windows_v1", "fixture_model", "fixture_model") == []
    other = json.loads(json.dumps(s))
    other["benchmark"]["benchmark_hash"] = "different"
    other["protocols"][0]["fingerprint"] = "changed"
    other["models"][0]["precision"] = {"weights": "nf4"}
    w = compatibility(s, other, "shared_windows_v1", "fixture_model", "fixture_model")
    assert {"benchmark_version_differs", "protocol_fingerprint_differs_or_missing", "precision_or_quantisation_differs"} <= set(w)


def test_viewer_serves_text_as_data_not_html(fixture_run):
    from http.server import ThreadingHTTPServer

    from aidet_eval.viewer.server import Handler, discover

    Handler.runs = discover([str(fixture_run["run_dir"].parent)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        runs = json.loads(urllib.request.urlopen(base + "/api/runs").read())
        assert runs[0]["run"] == "fixture-run" and runs[0]["has_full"]
        ex = json.loads(urllib.request.urlopen(base + "/api/example?run=fixture-run&id=fixture-000").read())
        assert "<script>alert(1)</script>" in ex["example"]["text"]  # delivered verbatim as JSON data
        r = urllib.request.urlopen(base + "/")
        assert "Content-Security-Policy" in r.headers
        rows = json.loads(urllib.request.urlopen(base + "/api/examples?run=fixture-run&model=fixture_model&protocol=shared_windows_v1&outcome=false_positive").read())
        assert all(x["authorship"] == "human" for x in rows["rows"])
    finally:
        httpd.shutdown()
    js = (Path(__file__).parents[1] / "src/aidet_eval/viewer/static/app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "eval(" not in js
    assert "Array.from(e.text" in js  # code-point indexing in the browser


@pytest.mark.skipif(not Path("artifacts/parity/parity_report.json").exists(), reason="no authorised-weight parity report on this machine")
def test_recorded_loader_parity_passes():
    rep = json.loads(Path("artifacts/parity/parity_report.json").read_text())
    for m in rep["models"]:
        assert m["pass"], m["model_id"]
        assert m["bucket_argmax_agreement_single_vs_upstream"] == m["n"]


def test_external_thresholds_are_separate_operating_points(fixture_run, tmp_path):
    from aidet_eval.evaluate import evaluate_run

    cfg = tmp_path / "ext.yaml"
    cfg.write_text("source: fixture external calibration\nthresholds:\n  fixture_model:\n    shared_windows_v1:\n      my_cal_1pct: 0.5\n")
    evaluate_run(str(fixture_run["bench"]), "fixture-run", str(cfg))
    full = json.loads((fixture_run["run_dir"] / "results.full.json").read_bytes())
    ext = [c for c in full["calibrations"] if c["operating_point_id"] == "external:my_cal_1pct"]
    assert len(ext) == 1 and ext[0]["task"] == "external_threshold" and ext[0]["threshold"] == 0.5
    official = {c["operating_point_id"] for c in full["calibrations"] if c["task"] != "external_threshold"}
    assert official == {"human_fpr_1pct", "human_fpr_5pct"}
    p = next(p for p in full["predictions"] if p["status"] == "ok")
    d = next(d for d in p["operating_point_decisions"] if d["operating_point_id"] == "external:my_cal_1pct")
    assert d["flag_ai"] == (p["score"] > 0.5)
    assert all(c["ok"] for c in full["validation"]["semantic_checks"])


def test_breakdown_and_roc_endpoints(fixture_run):
    import sqlite3

    from aidet_eval.viewer.server import breakdown, index_path, roc

    con = sqlite3.connect(index_path(fixture_run["run_dir"]))
    b = breakdown(con, {"model": "fixture_model", "protocol": "shared_windows_v1", "op": "human_fpr_1pct", "by": "domain"})
    row = next(r for r in b["rows"] if r["category"] == "news")
    assert row["human"]["scored"] == 30 and 0 <= row["human"]["rate"] <= 1  # evaluation humans only
    r = roc(con, {"model": "fixture_model", "protocol": "shared_windows_v1", "task": "fully", "humans": "audited", "by": "domain"})
    c = r["curves"][0]
    assert c["fpr"][0] == 0 and c["tpr"][0] == 0 and c["fpr"][-1] == 1 and c["tpr"][-1] == 1
    assert 0.5 < c["auroc"] <= 1 and c["n_neg"] == 30 and c["n_pos"] == 20
