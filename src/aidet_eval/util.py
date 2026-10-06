"""Hashing, deterministic JSON, timestamps, logging and secret redaction."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from aidet_eval import paths


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_hashes(path: Path, algos: Iterable[str] = ("sha256",), chunk: int = 1 << 22) -> dict[str, str]:
    hs = {a: hashlib.new(a) for a in algos}
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            for h in hs.values():
                h.update(blk)
    return {a: h.hexdigest() for a, h in hs.items()}


def _check_finite(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        raise ValueError(f"non-finite number in JSON payload: {o!r}")
    return o


def canonical_json(obj: Any) -> str:
    """Deterministic, compact JSON used for fingerprints."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def write_json(path: Path, obj: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, allow_nan=False, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"),
    re.compile(r"(?i)([?&](?:token|access_token|key|sig|signature|x-amz-[a-z-]+)=)[^&\s]+"),
]


def redact(s: str) -> str:
    for p in _SECRET_PATTERNS:
        s = p.sub(lambda m: (m.group(1) if m.groups() else "") + "<redacted>", s)
    return s


def log_event(event: str, **fields: Any) -> None:
    """Append a structured, secret-redacted entry to the execution log."""
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "event": event, **fields}
    line = redact(json.dumps(rec, ensure_ascii=False, default=str))
    with open(paths.LOGS / "execution_log.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[{rec['ts']}] {event}: " + redact(json.dumps(fields, ensure_ascii=False, default=str))[:400], file=sys.stderr)


def nan_to_none(o: Any) -> Any:
    """Recursively replace NaN floats (pandas missing values) with None; reject infinities."""
    if isinstance(o, float):
        if math.isnan(o):
            return None
        if math.isinf(o):
            raise ValueError("infinite value in payload")
        return o
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return nan_to_none(o.item())
        except (ValueError, AttributeError):
            return o
    return o
