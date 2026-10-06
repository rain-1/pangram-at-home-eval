"""Download pinned model/tokenizer snapshots into models/hf and record file hashes."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from aidet_eval import paths
from aidet_eval.util import file_hashes, log_event, redact, utcnow, write_json

HF_CACHE = paths.MODELS / "hf"


def load_model_registry() -> list[dict]:
    with open(paths.CONFIGS / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def snapshot(repo: str, revision: str, allow: list[str], download: bool = True) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=allow, cache_dir=str(HF_CACHE),
                                  local_files_only=not download))


def fetch_models(model_ids: list[str] | None = None) -> dict:
    out = {}
    for m in load_model_registry():
        if model_ids and m["id"] not in model_ids:
            continue
        rec = {"model_id": m["id"], "at": utcnow(), "repos": []}
        try:
            for repo, rev, allow in ((m["checkpoint"], m["checkpoint_revision"], m["checkpoint_allow"]),
                                     (m["base_model"], m["base_revision"], m["base_allow"])):
                p = snapshot(repo, rev, allow)
                files = []
                for fp in sorted(p.rglob("*")):
                    if fp.is_file():
                        files.append({"name": str(fp.relative_to(p)), "bytes": fp.stat().st_size,
                                      **file_hashes(fp, ("sha256",))})
                rec["repos"].append({"repo": repo, "revision": rev, "path": str(p), "files": files})
            rec["status"] = "complete"
        except Exception as e:  # noqa: BLE001
            msg = redact(f"{type(e).__name__}: {e}")
            rec["status"] = "blocked" if ("401" in msg or "403" in msg or "gated" in msg.lower()) else "failed"
            rec["error"] = msg
        write_json(paths.MODELS / f"{m['id']}.snapshot.json", rec)
        log_event("model_snapshot", model=m["id"], status=rec["status"], error=rec.get("error"))
        out[m["id"]] = rec
    return out


if __name__ == "__main__":
    fetch_models()
