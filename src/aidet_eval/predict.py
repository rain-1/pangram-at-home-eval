"""Resumable prediction runner.

Shards: artifacts/runs/<run_id>/predictions/<model_id>/<protocol_id>.jsonl (append-only).
Each record is an input-level output (a window or a native document input) with a cache key over
(model fingerprint, protocol fingerprint, input text hash, truncation flag). A resumed run reuses
the same frozen benchmark/window manifest; a changed benchmark hash under an existing run ID is refused.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from aidet_eval import paths
from aidet_eval.models.base import DetectorInput, load_detector
from aidet_eval.util import fingerprint, log_event, read_json, sha256_text, utcnow, write_json

CHUNK = 1024


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=paths.ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status), "status_sha256": sha256_text(status or "")}


def default_run_id(bench_dir: Path) -> str:
    m = read_json(bench_dir / "manifest.json")
    return f"{m['benchmark_id']}__{m['benchmark_hash'][:10]}"


def init_run(bench_dir: Path, run_id: str | None, models: list[str], protocols: list[str]) -> Path:
    man = read_json(bench_dir / "manifest.json")
    run_id = run_id or default_run_id(bench_dir)
    rd = paths.RUNS / run_id
    rd.mkdir(parents=True, exist_ok=True)
    rj = rd / "run.json"
    if rj.exists():
        prev = read_json(rj)
        if prev["benchmark_hash"] != man["benchmark_hash"]:
            raise SystemExit(f"run {run_id} belongs to benchmark {prev['benchmark_hash'][:10]}; refusing to mix benchmarks")
        prev["requested_models"] = sorted(set(prev["requested_models"]) | set(models))
        prev["requested_protocols"] = sorted(set(prev["requested_protocols"]) | set(protocols))
        prev.setdefault("invocations", []).append({"at": utcnow(), "models": models, "protocols": protocols})
        write_json(rj, prev)
        return rd
    lock = paths.ROOT / "uv.lock"
    write_json(rj, {
        "run_id": run_id, "created_at": utcnow(), "benchmark_dir": str(bench_dir), "benchmark_id": man["benchmark_id"],
        "benchmark_hash": man["benchmark_hash"], "profile": man["profile"], "requested_models": models,
        "requested_protocols": protocols, "git": git_state(),
        "dependency_lock_sha256": sha256_text(lock.read_text()) if lock.exists() else None,
        "platform": {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine()},
        "invocations": [{"at": utcnow(), "models": models, "protocols": protocols}],
    })
    return rd


def _load_done(shard: Path) -> dict[str, str]:
    done = {}
    if shard.exists():
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from an interrupted write; it will be recomputed
                if r.get("status") in ("ok", "abstain", "error"):
                    done[r["input_id"]] = r["cache_key"]
    return done


def _reuse_index(rd: Path, mid: str, pid: str) -> dict[str, dict]:
    """Records from OTHER runs keyed by content key (model fp, protocol params w/o window-set hash, input sha, truncate)."""
    idx: dict[str, dict] = {}
    for other in paths.RUNS.glob("*/run.json"):
        od = other.parent
        if od == rd:
            continue
        shard = od / "predictions" / mid / f"{pid}.jsonl"
        bench_dir = Path(read_json(other).get("benchmark_dir", ""))
        protos = bench_dir / "windows" / "protocols.json"
        if not shard.exists() or not protos.exists():
            continue
        pdef = next(p for p in read_json(protos)["protocols"] if p["protocol_id"] == pid)
        backfill = None
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "ok":
                    continue
                ck = r.get("content_key")
                if not ck:  # older shards: derive from that run's own frozen benchmark
                    backfill = backfill if backfill is not None else _backfill_inputs(bench_dir)
                    if r["input_id"] not in backfill or not r.get("model_fingerprint"):
                        continue
                    tsha, off = backfill[r["input_id"]]
                    ck = content_key(r["model_fingerprint"], pdef, tsha, pid == "native_truncate_v1", off)
                idx[ck] = {**r, "content_key": ck, "reused_from_run": od.name}
    return idx


def content_key(mfp: str, pdef: dict, text_sha: str, truncate: bool, offset: int) -> str:
    """Identical model, protocol parameters (excluding the window-set hash), input text, truncation and
    canonical offset (coverage offsets are document-relative) => identical computation."""
    params = {k: v for k, v in pdef.items() if k not in ("window_manifest_sha256", "fingerprint")}
    return sha256_text(f"{mfp}|{fingerprint(params)}|{text_sha}|{truncate}|{offset}")


def _backfill_inputs(bench_dir: Path) -> dict[str, tuple[str, int]]:
    """input_id -> (input text sha256, canonical offset) reconstructed from a frozen benchmark."""
    ex = pd.read_parquet(bench_dir / "examples.parquet", columns=["example_id", "text"])
    texts = dict(zip(ex["example_id"], ex["text"]))
    out = {f"{e}#native": (sha256_text(t), 0) for e, t in texts.items()}
    win = pd.read_parquet(bench_dir / "windows" / "shared_windows_v1.parquet", columns=["window_id", "example_id", "start", "end"])
    for w in win.itertuples(index=False):
        out[w.window_id] = (sha256_text(texts[w.example_id][w.start:w.end]), int(w.start))
    return out


def predict(bench: str, models: list[str] | None, protocols: list[str] | None, resume: bool = True,
            run_id: str | None = None, limit: int | None = None, roles: list[str] | None = None, reuse: bool = True) -> Path:
    os.chdir(paths.ROOT)
    bench_dir = Path(bench)
    specs = {s["id"]: s for s in yaml.safe_load(open(paths.CONFIGS / "models.yaml"))["models"]}
    models = models or list(specs)
    protocols = protocols or ["shared_windows_v1", "native_truncate_v1"]
    if not (bench_dir / "windows" / "FROZEN").exists():
        raise SystemExit("window manifest not frozen; run `python -m aidet_eval windows --benchmark ...` first")
    pdefs = {p["protocol_id"]: p for p in read_json(bench_dir / "windows" / "protocols.json")["protocols"]}
    rd = init_run(bench_dir, run_id, models, protocols)
    ex = pd.read_parquet(bench_dir / "examples.parquet", columns=["example_id", "text", "role"])
    if roles:
        ex = ex[ex["role"].isin(roles)]
    if limit:
        ex = ex.sort_values("example_id").head(limit)
        rj = read_json(rd / "run.json")
        rj["debug_limit"] = limit
        write_json(rd / "run.json", rj)
    texts = dict(zip(ex["example_id"], ex["text"]))
    win = pd.read_parquet(bench_dir / "windows" / "shared_windows_v1.parquet")
    win = win[win["example_id"].isin(texts)]

    for mid in models:
        if mid not in specs:
            raise SystemExit(f"unknown model {mid}")
        det = load_detector(specs[mid])
        model_status = {"model_id": mid, "at": utcnow()}
        try:
            det.load()
            model_status.update(status="loaded", describe=det.describe(), runtime=det.runtime_info())
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            model_status.update(status="blocked" if ("401" in msg or "403" in msg or "gated" in msg.lower()) else "failed", error=msg)
            write_json(rd / "models" / f"{mid}.json", model_status)
            log_event("model_load_failed", model=mid, error=msg[:300])
            continue
        mfp = fingerprint(det.describe())
        model_status["model_fingerprint"] = mfp
        write_json(rd / "models" / f"{mid}.json", model_status)
        for pid in protocols:
            pfp = pdefs[pid]["fingerprint"]
            truncate = pid == "native_truncate_v1"
            if pid == "shared_windows_v1":
                inputs = [DetectorInput(w.window_id, texts[w.example_id][w.start:w.end], int(w.start))
                          for w in win.itertuples(index=False)]
            else:
                inputs = [DetectorInput(f"{e}#native", t, 0) for e, t in texts.items()]
            keys = {i.input_id: sha256_text(f"{mfp}|{pfp}|{sha256_text(i.text)}|{truncate}") for i in inputs}
            ckeys = {i.input_id: content_key(mfp, pdefs[pid], sha256_text(i.text), truncate, i.canonical_offset) for i in inputs}
            shard = rd / "predictions" / mid / f"{pid}.jsonl"
            shard.parent.mkdir(parents=True, exist_ok=True)
            done = _load_done(shard) if resume else {}
            todo = [i for i in inputs if done.get(i.input_id) != keys[i.input_id]]
            if reuse and todo:
                ridx = _reuse_index(rd, mid, pid)
                reused = [i for i in todo if ckeys[i.input_id] in ridx]
                if reused:
                    with open(shard, "a", encoding="utf-8") as f:
                        for i in reused:
                            r = dict(ridx[ckeys[i.input_id]])
                            r.update(input_id=i.input_id, example_id=i.input_id.split("#")[0], cache_key=keys[i.input_id],
                                     protocol_fingerprint=pfp, at=utcnow())
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    rs = {i.input_id for i in reused}
                    todo = [i for i in todo if i.input_id not in rs]
                    log_event("predict_reused", model=mid, protocol=pid, n=len(reused))
            log_event("predict_start", model=mid, protocol=pid, n_inputs=len(inputs), todo=len(todo))
            todo.sort(key=lambda i: len(i.text))
            t_start = time.time()
            for c in range(0, len(todo), CHUNK):
                chunk = todo[c:c + CHUNK]
                t0 = time.time()
                outs = det.predict(chunk, truncate=truncate)
                wall = time.time() - t0
                with open(shard, "a", encoding="utf-8") as f:
                    for o in outs:
                        f.write(json.dumps({
                            "input_id": o.input_id, "example_id": o.input_id.split("#")[0], "model_id": mid,
                            "protocol_id": pid, "status": o.status, "score": o.score, "raw_outputs": o.raw_outputs,
                            "native_spans": o.native_spans, "coverage": o.coverage, "n_tokens": o.n_tokens,
                            "input_fingerprint": o.input_fingerprint, "cache_key": keys[o.input_id], "error": o.error,
                            "seconds_per_item": o.seconds, "chunk_wall_seconds": wall, "chunk_size": len(chunk),
                            "model_fingerprint": mfp, "protocol_fingerprint": pfp, "content_key": ckeys[o.input_id],
                            "at": utcnow()}, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                log_event("predict_chunk", model=mid, protocol=pid, done=min(c + CHUNK, len(todo)), todo=len(todo),
                          items_per_s=round(len(chunk) / max(wall, 1e-9), 1))
            log_event("predict_done", model=mid, protocol=pid, seconds=round(time.time() - t_start, 1))
        del det
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return rd
