"""Document-level predictions from input-level shards.

shared_windows_v1: document score = max(window score) over *all* frozen windows of the document.
  - all windows ok (or abstain because empty after preprocessing, with >= 1 ok) -> status ok;
  - any window error/missing -> status error, no complete score; a separately labelled
    lower-bound maximum over the successful windows is kept in `partial_max_lower_bound`;
  - every window abstains -> status abstain.
native_truncate_v1: the single native input record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_shard(path: Path) -> pd.DataFrame:
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("input_id", keep="last")  # latest record wins on resume
    return df


def doc_predictions(run_dir: Path, bench_dir: Path, model_id: str, protocol_id: str, example_ids: list[str]) -> pd.DataFrame:
    shard = read_shard(run_dir / "predictions" / model_id / f"{protocol_id}.jsonl")
    out = []
    if protocol_id == "native_truncate_v1":
        by = {r["input_id"]: r for r in shard.to_dict("records")} if len(shard) else {}
        for e in example_ids:
            r = by.get(f"{e}#native")
            if r is None:
                out.append({"example_id": e, "status": "missing", "score": None, "n_windows": 1, "windows": None, "native": None})
            else:
                out.append({"example_id": e, "status": r["status"], "score": r["score"] if r["status"] == "ok" else None,
                            "n_windows": 1, "windows": None, "native": r})
        return pd.DataFrame(out)
    win = pd.read_parquet(bench_dir / "windows" / "shared_windows_v1.parquet", columns=["window_id", "example_id", "start", "end", "n_words", "kind"])
    by = {r["input_id"]: r for r in shard.to_dict("records")} if len(shard) else {}
    groups = {e: g for e, g in win.groupby("example_id", sort=False)}
    for e in example_ids:
        g = groups.get(e)
        if g is None:
            out.append({"example_id": e, "status": "abstain", "score": None, "n_windows": 0, "windows": [], "native": None,
                        "reason": "no_windows_empty_text"})
            continue
        recs, statuses = [], []
        for w in g.itertuples(index=False):
            r = by.get(w.window_id)
            st = r["status"] if r else "missing"
            statuses.append(st)
            recs.append({"window_id": w.window_id, "start": int(w.start), "end": int(w.end), "n_words": int(w.n_words),
                         "kind": w.kind, "record": r})
        ok_scores = [x["record"]["score"] for x in recs if x["record"] and x["record"]["status"] == "ok"]
        if any(s in ("error", "missing") for s in statuses):
            status, score = "error", None
        elif not ok_scores:
            status, score = "abstain", None
        else:
            status, score = "ok", max(ok_scores)
        out.append({"example_id": e, "status": status, "score": score, "n_windows": len(recs), "windows": recs, "native": None,
                    "partial_max_lower_bound": (max(ok_scores) if (status == "error" and ok_scores) else None),
                    "n_window_errors": sum(s in ("error", "missing") for s in statuses),
                    "n_window_abstain": sum(s == "abstain" for s in statuses)})
    return pd.DataFrame(out)
