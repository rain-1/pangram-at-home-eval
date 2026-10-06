"""Command-line entry point: python -m aidet_eval <command>."""

from __future__ import annotations

import argparse
import json
import sys


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def cmd_doctor(a) -> int:
    from aidet_eval.doctor import doctor

    return doctor()


def cmd_download(a) -> int:
    from aidet_eval.acquire import acquire

    keys = _csv(a.sources)
    res = acquire(keys, resume=a.resume)
    for m in res:
        print(f"{m['key']:28s} {m['status']:9s} {m.get('n_files_ok')}/{m.get('n_files_expected')} {m.get('error') or ''}")
    return 0


def cmd_status(a) -> int:
    from aidet_eval.acquire import acquisition_status

    for r in acquisition_status():
        print(f"{r['key']:28s} {r['status']:12s} {r.get('n_files_ok')}/{r.get('n_files_expected')}  {(r.get('error') or '')[:100]}")
    return 0


def cmd_inventory(a) -> int:
    from aidet_eval.inventory import run_inventory

    run_inventory(_csv(a.sources))
    return 0


def cmd_normalise(a) -> int:
    from aidet_eval.adapters import normalise

    normalise(_csv(a.sources))
    return 0


def cmd_build(a) -> int:
    from aidet_eval.build import build

    build(a.config, a.profile, a.out)
    return 0


def cmd_validate(a) -> int:
    from aidet_eval.build import validate_benchmark

    ok = validate_benchmark(a.benchmark)
    return 0 if ok else 1


def cmd_windows(a) -> int:
    from aidet_eval.windows import build_window_manifest

    build_window_manifest(a.benchmark)
    return 0


def cmd_predict(a) -> int:
    from aidet_eval.predict import predict

    predict(a.benchmark, _csv(a.models), _csv(a.protocols), resume=a.resume, run_id=a.run_id,
            limit=a.limit, roles=_csv(a.roles))
    return 0


def cmd_parity(a) -> int:
    from aidet_eval.parity import run_parity

    return run_parity(_csv(a.models))


def cmd_calibrate(a) -> int:
    from aidet_eval.calibrate import calibrate_run

    calibrate_run(a.benchmark, a.run_id)
    return 0


def cmd_evaluate(a) -> int:
    from aidet_eval.evaluate import evaluate_run

    evaluate_run(a.benchmark, a.run_id, a.external_thresholds)
    return 0


def cmd_cal_sens(a) -> int:
    from aidet_eval.calib_sensitivity import run_sensitivity

    print(run_sensitivity(a.benchmark, a.run))
    return 0


def cmd_export(a) -> int:
    from aidet_eval.results.export import export_run

    export_run(a.run, redact_texts=a.redact)
    return 0


def cmd_view(a) -> int:
    from aidet_eval.viewer.server import serve

    serve(a.runs, a.host, a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aidet_eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check environment, hardware, access").set_defaults(fn=cmd_doctor)

    d = sub.add_parser("download", help="acquire catalogue sources (resumable)")
    d.add_argument("--catalogue", default=None, help="catalogue path (informational; registry is configs/sources.yaml)")
    d.add_argument("--all", action="store_true")
    d.add_argument("--sources", default=None, help="comma-separated registry keys")
    d.add_argument("--resume", action="store_true", default=True)
    d.set_defaults(fn=cmd_download)

    sub.add_parser("status", help="acquisition status table").set_defaults(fn=cmd_status)

    i = sub.add_parser("inventory", help="measure per-source inventories")
    i.add_argument("--sources", default=None)
    i.set_defaults(fn=cmd_inventory)

    n = sub.add_parser("normalise", help="derive canonical parquet per source")
    n.add_argument("--sources", default=None)
    n.set_defaults(fn=cmd_normalise)

    b = sub.add_parser("build", help="select and freeze a benchmark version")
    b.add_argument("--config", default="spec/benchmark_plan.yaml")
    b.add_argument("--profile", default="full", choices=["full", "smoke"])
    b.add_argument("--out", default=None)
    b.set_defaults(fn=cmd_build)

    v = sub.add_parser("validate", help="validate a frozen benchmark")
    v.add_argument("--benchmark", required=True)
    v.set_defaults(fn=cmd_validate)

    w = sub.add_parser("windows", help="build the frozen shared_windows_v1 manifest")
    w.add_argument("--benchmark", required=True)
    w.set_defaults(fn=cmd_windows)

    pr = sub.add_parser("predict", help="run detectors (resumable shards)")
    pr.add_argument("--benchmark", required=True)
    pr.add_argument("--models", default="open_pangram_roberta,open_pangram_llama")
    pr.add_argument("--protocols", default="shared_windows_v1,native_truncate_v1")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--resume", action="store_true", default=True)
    pr.add_argument("--limit", type=int, default=None, help="debug: limit examples (marks run partial)")
    pr.add_argument("--roles", default=None, help="restrict to calibration,evaluation")
    pr.set_defaults(fn=cmd_predict)

    pa = sub.add_parser("parity", help="adapter vs pinned upstream implementation parity")
    pa.add_argument("--models", default="open_pangram_roberta,open_pangram_llama")
    pa.set_defaults(fn=cmd_parity)

    c = sub.add_parser("calibrate", help="freeze operating points from human calibration docs")
    c.add_argument("--benchmark", required=True)
    c.add_argument("--run-id", default=None)
    c.set_defaults(fn=cmd_calibrate)

    e = sub.add_parser("evaluate", help="compute aggregates and export results")
    e.add_argument("--benchmark", required=True)
    e.add_argument("--run-id", default=None)
    e.add_argument("--external-thresholds", default=None, help="YAML of thresholds calibrated elsewhere -> operating points 'external:<name>'")
    e.set_defaults(fn=cmd_evaluate)

    cs = sub.add_parser("calibration-sensitivity", help="exploratory per-dataset thresholds (never replaces official ones)")
    cs.add_argument("--benchmark", required=True)
    cs.add_argument("--run", required=True)
    cs.set_defaults(fn=cmd_cal_sens)

    x = sub.add_parser("export", help="(re)export results.full/summary JSON for a run")
    x.add_argument("--run", required=True)
    x.add_argument("--redact", action="store_true", help="write a text-redacted local export")
    x.set_defaults(fn=cmd_export)

    vw = sub.add_parser("view", help="local results viewer")
    vw.add_argument("--runs", default="artifacts/runs")
    vw.add_argument("--host", default="127.0.0.1")
    vw.add_argument("--port", type=int, default=8765)
    vw.set_defaults(fn=cmd_view)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
