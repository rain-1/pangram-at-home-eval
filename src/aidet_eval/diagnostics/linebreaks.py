"""Line-break swap diagnostic: does a detector key on hard line wrapping rather than authorship?

Builds paired variants of the same benchmark documents that differ ONLY in line wrapping:
  * unwrap: single newlines inside paragraphs become spaces (paragraph breaks "\\n\\n" kept);
  * wrap70: each paragraph is hard-wrapped at 70 characters (paragraph breaks kept).
A detector that reads authorship should give the same verdict on both variants; one that keys on
formatting flips. Inputs are exported in the workbench scorer format ({input_id, protocol_id, text,
offset}) with protocol_id native_truncate_v1 so remote models can score them; local detectors are
scored directly. Labels never leave this module.

Usage:
  python -m aidet_eval.diagnostics.linebreaks export --benchmark artifacts/benchmark-v3 --out DIAG.jsonl.gz
  python -m aidet_eval.diagnostics.linebreaks score-local --inputs DIAG.jsonl.gz --models open_pangram_roberta,open_pangram_llama
  python -m aidet_eval.diagnostics.linebreaks analyse --inputs DIAG.jsonl.gz --run artifacts/runs/<v3 run> [--scores-dir DIR]
Remote (workbench Space): python score_benchmark.py <tag> <checkpoint> DIAG.jsonl.gz <OUT_DIR>, then copy OUT_DIR here and
pass it as --scores-dir (files <tag>--<checkpoint>--NNofMM.jsonl; keep it separate from the main external_scores).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from aidet_eval.adapters.common import hash_rank

GROUPS = [
    # (group id, description, selector on examples, transformation applied to make the "swapped" variant)
    ("pg19_human", "PG-19 human fiction (hard-wrapped)", lambda e: (e.dataset_key == "pg19") & (e.authorship == "human") & (e.role == "evaluation"), "unwrap", 200),
    ("het_gutenberg_human", "heterogeneous originals, Gutenberg (hard-wrapped, CRLF, curly)", lambda e: e.panel_ids.str.contains("heterogeneous_matched_human_controls") & (e.src_ds == "gutenberg_selected"), "unwrap,unwrap_straightq", 200),
    ("het_stdebooks_human", "heterogeneous originals, Standard Ebooks (unwrapped, curly)", lambda e: e.panel_ids.str.contains("heterogeneous_matched_human_controls") & (e.src_ds == "standardebooks"), "wrap70,straightq,wrap70_straightq", 200),
    ("hap_human", "HAP-E-2 human Gutenberg continuations (unwrapped)", lambda e: (e.dataset_key == "hap_e2") & (e.authorship == "human") & (e.role == "evaluation"), "wrap70", 200),
    ("het_gutenberg_mixed", "heterogeneous block insertions into Gutenberg (AI blocks unwrapped, human parts wrapped)", lambda e: e.panel_ids.str.contains("heterogeneous_block_replacement") & (e.src_ds == "gutenberg_selected"), "unwrap", 200),
    ("storyscope_ai", "fully AI literary stories (unwrapped, curly)", lambda e: (e.dataset_key == "storyscope") & (e.role == "evaluation"), "wrap70,straightq,wrap70_straightq", 200),
]


def unwrap(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return "\n\n".join(re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in re.split(r"\n\s*\n", t))


CURLY = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def straightq(t: str) -> str:
    return t.translate(CURLY)


def wrap70_straightq(t: str) -> str:
    return straightq(wrap70(t))


def unwrap_straightq(t: str) -> str:
    return straightq(unwrap(t))


def wrap70(t: str) -> str:
    return "\n\n".join(textwrap.fill(re.sub(r"\s+", " ", p).strip(), 70) for p in re.split(r"\n\s*\n", t))


def build(bench: Path) -> pd.DataFrame:
    ex = pd.read_parquet(bench / "examples.parquet", columns=["example_id", "text", "dataset_key", "authorship", "role", "panel_ids", "subdomain", "original_labels"])
    ex["src_ds"] = ex["original_labels"].map(lambda s: (json.loads(s) if s else {}).get("source_dataset"))
    rows = []
    for gid, desc, sel, how, cap in GROUPS:
        g = ex[sel(ex)]
        g = g.assign(r=g.example_id.map(lambda e: hash_rank("linebreaks", e))).sort_values("r").head(cap)
        fns = {"unwrap": unwrap, "wrap70": wrap70, "straightq": straightq, "wrap70_straightq": wrap70_straightq, "unwrap_straightq": unwrap_straightq}
        for r in g.itertuples(index=False):
            variants = [("original", r.text)] + [(v, fns[v](r.text)) for v in how.split(",")]
            variants = [variants[0]] + [(v, t) for v, t in variants[1:] if t != r.text]
            if len(variants) == 1:
                continue
            for variant, text in variants:
                rows.append({"input_id": f"diag-linebreaks:{r.example_id}:{variant}", "protocol_id": "native_truncate_v1",
                             "text": text, "offset": 0, "_group": gid, "_desc": desc, "_variant": variant, "_example_id": r.example_id,
                             "_authorship": r.authorship, "_newlines_per_100w": round(100 * text.count("\n") / max(1, len(text.split())), 2)})
    return pd.DataFrame(rows)


def export(bench: str, out: str) -> None:
    df = build(Path(bench))
    pub = df[["input_id", "protocol_id", "text", "offset"]]  # labels and groups never leave this module
    b = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pub.to_dict("records")).encode()
    Path(out).write_bytes(gzip.compress(b, mtime=0))
    key = Path(out).with_suffix("").with_suffix(".key.parquet")
    df.drop(columns=["text"]).to_parquet(key, index=False)
    print(json.dumps({"inputs": len(pub), "sha256": hashlib.sha256(b).hexdigest(), "groups": df.groupby("_group").size().to_dict(),
                      "label_key": str(key)}))


def score_local(inputs: str, models: list[str], out_dir: str) -> None:
    import yaml

    from aidet_eval import paths
    from aidet_eval.models.base import DetectorInput, load_detector

    rows = [json.loads(l) for l in gzip.open(inputs, "rt")]
    specs = {s["id"]: s for s in yaml.safe_load(open(paths.CONFIGS / "models.yaml"))["models"]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for m in models:
        det = load_detector(specs[m]); det.load()
        outs = det.predict([DetectorInput(r["input_id"], r["text"]) for r in rows], truncate=True)
        with open(Path(out_dir) / f"{m}.jsonl", "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"input_id": o.input_id, "status": o.status, "score": o.score,
                                    "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}) + "\n")
        print(f"scored {len(rows)} inputs with {m}")
        del det


def _load_scores(model: str, local_dir: Path, scores_dir: Path | None, spec: dict | None) -> dict[str, float]:
    f = local_dir / f"{model}.jsonl"
    if f.exists():
        return {r["input_id"]: r["score"] for r in map(json.loads, open(f)) if r["status"] == "ok"}
    if spec and scores_dir:  # remote (workbench) scorer output, same file naming as PrecomputedDetector
        out = {}
        for p in sorted(scores_dir.glob(f'{spec["tag"]}--{spec["checkpoint"]}--*of*.jsonl')):
            for r in map(json.loads, open(p)):
                if r.get("status") == "ok" and r["input_id"].startswith("diag-linebreaks:"):
                    out[r["input_id"]] = r[spec["score_field"]]
        return out
    return {}


def analyse(inputs: str, run: str, local_dir: str, scores_dir: str | None) -> None:
    import yaml

    from aidet_eval import paths

    key = pd.read_parquet(Path(inputs).with_suffix("").with_suffix(".key.parquet"))
    cal = json.loads((Path(run) / "calibration.json").read_text())["operating_points"]
    specs = {s["id"]: s for s in yaml.safe_load(open(paths.CONFIGS / "models.yaml"))["models"]}
    report = {"inputs": inputs, "threshold_source": "v3 native_truncate_v1 human_fpr_1pct (frozen calibration)", "models": {}}
    for m in sorted({o["model_id"] for o in cal}):
        thr = next(o["threshold"] for o in cal if o["model_id"] == m and o["protocol_id"] == "native_truncate_v1" and o["operating_point_id"] == "human_fpr_1pct")
        sc = _load_scores(m, Path(local_dir), Path(scores_dir) if scores_dir else None, specs.get(m))
        if not sc:
            continue
        k = key.assign(score=key.input_id.map(sc)).dropna(subset=["score"])
        k["flag"] = k.score > thr
        res = {}
        for gid, g in k.groupby("_group"):
            p = g.pivot_table(index="_example_id", columns="_variant", values=["flag", "score"], aggfunc="first")
            for sw in [c for c in p["flag"].columns if c != "original"]:
                q = p[[("flag", "original"), ("flag", sw), ("score", "original"), ("score", sw)]].dropna()
                res[f"{gid}:{sw}"] = {"desc": g._desc.iloc[0], "authorship": g._authorship.iloc[0], "swap": sw, "n_pairs": int(len(q)),
                                      "flagged_original": float(q[("flag", "original")].astype(bool).mean()), "flagged_swapped": float(q[("flag", sw)].astype(bool).mean()),
                                      "verdict_flips": int((q[("flag", "original")] != q[("flag", sw)]).sum()),
                                      "median_score_original": float(q[("score", "original")].median()), "median_score_swapped": float(q[("score", sw)].median())}
        report["models"][m] = {"threshold": thr, "groups": res}
    out = Path(run) / "diagnostic_linebreaks.json"
    out.write_text(json.dumps(report, indent=1))
    for m, r in report["models"].items():
        print(f"\n{m} (threshold {r['threshold']:.4f})")
        for gid, g in r["groups"].items():
            print(f"  {g['desc'][:58]:58s} n={g['n_pairs']:3d}  flagged: original {100*g['flagged_original']:5.1f}%  ->  {g['swap']:17s} {100*g['flagged_swapped']:5.1f}%   (flips {g['verdict_flips']})")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--benchmark", required=True); e.add_argument("--out", required=True)
    s = sub.add_parser("score-local"); s.add_argument("--inputs", required=True); s.add_argument("--models", required=True)
    s.add_argument("--out-dir", default="artifacts/diagnostics/linebreaks")
    a = sub.add_parser("analyse"); a.add_argument("--inputs", required=True); a.add_argument("--run", required=True)
    a.add_argument("--local-dir", default="artifacts/diagnostics/linebreaks"); a.add_argument("--scores-dir", default=None)
    x = ap.parse_args()
    if x.cmd == "export":
        export(x.benchmark, x.out)
    elif x.cmd == "score-local":
        score_local(x.inputs, x.models.split(","), x.out_dir)
    else:
        analyse(x.inputs, x.run, x.local_dir, x.scores_dir)


if __name__ == "__main__":
    main()
