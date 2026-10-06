"""Resumable, checksum-verified acquisition of every catalogue source.

Raw files land in data/raw/<key>/files/ and are made read-only after verification.
Each source gets data/raw/<key>/_acquisition.json with an explicit status:
complete | partial | blocked | failed, the completed and remaining files, and a
continuation command. Tokens are read by huggingface_hub from its own cache and
never logged.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
import shutil
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml

from aidet_eval import paths
from aidet_eval.util import file_hashes, log_event, read_json, redact, utcnow, write_json

CONTINUE_CMD = "uv run python -m aidet_eval download --sources {key} --resume"
UA = {"User-Agent": "aidet_eval/0.1 (research evaluation; resumable downloader)"}


def load_registry(include_extra: bool = True) -> list[dict[str, Any]]:
    """Catalogue sources (exactly the 16) followed by user-requested extra sources (catalogue_name null)."""
    with open(paths.CONFIGS / "sources.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["sources"] + (cfg.get("extra_sources") or [] if include_extra else [])


def load_catalogue() -> dict[str, Any]:
    return read_json(paths.CATALOGUE)


def check_registry_covers_catalogue() -> list[str]:
    names = {s["name"] for s in load_catalogue()["sources"]}
    reg = {s["catalogue_name"] for s in load_registry(include_extra=False)}
    problems = [f"catalogue source missing from registry: {n}" for n in sorted(names - reg)]
    problems += [f"registry entry not in catalogue: {n}" for n in sorted(reg - names)]
    return problems


def _src_dir(key: str) -> Path:
    return paths.RAW / key


def _manifest_path(key: str) -> Path:
    return _src_dir(key) / "_acquisition.json"


def _make_readonly(p: Path) -> None:
    mode = p.stat().st_mode
    p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# --------------------------------------------------------------------------- HF


def _hf_plan(src: dict) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(src["repo"], revision=src["revision"], files_metadata=True)
    out = []
    for s in info.siblings:
        if not any(fnmatch.fnmatch(s.rfilename, pat) for pat in src["allow"]):
            continue
        out.append(
            {
                "name": s.rfilename,
                "size": s.size,
                "lfs_sha256": s.lfs.sha256 if s.lfs else None,
                "git_blob_sha1": s.blob_id,
                "url": f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{s.rfilename}",
            }
        )
    return out


def _hf_fetch(src: dict, f: dict, dest_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        src["repo"], f["name"], repo_type="dataset", revision=src["revision"], local_dir=str(dest_root)
    )
    return Path(p)


def _hf_verify(f: dict, p: Path) -> dict:
    if f["lfs_sha256"]:
        h = file_hashes(p, ("sha256",))
        ok = h["sha256"] == f["lfs_sha256"]
        return {"sha256": h["sha256"], "verified_against": "hf_lfs_sha256", "ok": ok}
    blob = _git_blob_sha1(p)
    h = file_hashes(p, ("sha256",))
    return {"sha256": h["sha256"], "git_blob_sha1": blob, "verified_against": "git_blob_sha1", "ok": blob == f["git_blob_sha1"]}


# ------------------------------------------------------------------------- HTTP


def _http_download(url: str, dest: Path, expected_size: int | None = None, session=None) -> None:
    """Resumable HTTP GET using a .part file and Range requests."""
    s = session or requests.Session()
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expected_size is not None and have == expected_size:
        os.replace(part, dest)
        return
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    with s.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
        if r.status_code == 416 and expected_size is not None and have == expected_size:
            os.replace(part, dest)
            return
        if have and r.status_code == 200:
            have = 0  # server ignored Range; restart
        elif r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {redact(url)}")
        with open(part, "ab" if have else "wb") as out:
            for blk in r.iter_content(1 << 20):
                out.write(blk)
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {dest.name}: {part.stat().st_size} != {expected_size}")
    os.replace(part, dest)


def _gdrive_download(file_id: str, dest: Path) -> dict:
    """Download a publicly shared Google Drive file. Returns response metadata."""
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    s = requests.Session()
    with s.get(url, headers=UA, stream=True, timeout=120) as r:
        ctype = r.headers.get("content-type", "")
        disp = r.headers.get("content-disposition", "")
        if r.status_code != 200:
            raise RuntimeError(f"Google Drive HTTP {r.status_code}")
        if "text/html" in ctype:
            body = r.text[:2000]
            raise PermissionError("Google Drive returned an HTML page instead of the file (quota, sign-in or "
                                  "removed sharing). First 200 chars: " + body[:200])
        part = dest.with_name(dest.name + ".part")
        with open(part, "wb") as out:
            for blk in r.iter_content(1 << 20):
                out.write(blk)
    os.replace(part, dest)
    return {"content_type": ctype, "content_disposition": disp}


# -------------------------------------------------------------------------- GCS


def _gcs_list(bucket: str, prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash,generation,updated),nextPageToken"}
        if token:
            params["pageToken"] = token
        r = requests.get(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o", params=params, headers=UA, timeout=60)
        r.raise_for_status()
        d = r.json()
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def _md5_b64(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while blk := f.read(1 << 22):
            h.update(blk)
    return base64.b64encode(h.digest()).decode()


# ------------------------------------------------------------------------ main


def acquire_source(src: dict, resume: bool = True, max_workers: int = 16) -> dict:
    key = src["key"]
    root = _src_dir(key)
    files_root = root / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    prev = read_json(_manifest_path(key)) if _manifest_path(key).exists() else {}
    man: dict[str, Any] = {
        "key": key,
        "catalogue_name": src["catalogue_name"],
        "kind": src["kind"],
        "registry_entry": src,
        "started_at": utcnow(),
        "retrieval_history": prev.get("retrieval_history", []),
        "continuation_command": CONTINUE_CMD.format(key=key),
    }
    plan: list[dict] = []
    try:
        if src["kind"] == "hf_dataset":
            plan = _hf_plan(src)
        elif src["kind"] == "http_files":
            plan = [dict(f) for f in src["files"]]
        elif src["kind"] == "gcs_bucket":
            for pre in src["prefixes"]:
                for it in _gcs_list(src["bucket"], pre):
                    if it["name"].endswith("/"):
                        continue
                    plan.append({"name": it["name"], "size": int(it["size"]), "gcs_md5_b64": it["md5Hash"],
                                 "gcs_generation": it["generation"], "updated": it["updated"],
                                 "url": f"https://storage.googleapis.com/{src['bucket']}/{it['name']}"})
        elif src["kind"] == "gdrive_files":
            plan = [dict(f) for f in src["files"]]
        elif src["kind"] == "github_files":
            repo = src["repo_url"].removeprefix("https://github.com/")
            plan = [{"name": f["path"], "git_blob_sha1": f["git_blob_sha1"],
                     "url": f"https://raw.githubusercontent.com/{repo}/{src['commit']}/{f['path']}"} for f in src["files"]]
        elif src["kind"] == "local_release":
            root_rel = (paths.ROOT / src["path"]).resolve()
            sums = {}
            for line in (root_rel / src["checksums_file"]).read_text().splitlines():
                h, _, name = line.partition("  ")
                sums[name.strip()] = h.strip()
            plan = [{"name": f, "src": str(root_rel / f), "sha256_expected": sums.get(f), "size": (root_rel / f).stat().st_size}
                    for f in src["files"]]
        else:
            raise ValueError(f"unknown kind {src['kind']}")
    except Exception as e:  # noqa: BLE001 - classify and record
        msg = redact(f"{type(e).__name__}: {e}")
        status = "blocked" if any(t in msg for t in ("401", "403", "Gated", "gated", "Unauthorized")) else "failed"
        man.update(status=status, error=msg, files=[], finished_at=utcnow())
        write_json(_manifest_path(key), man)
        log_event("acquire_plan_failed", source=key, status=status, error=msg)
        return man

    expected = sum(f.get("size") or 0 for f in plan)
    free = shutil.disk_usage(paths.DATA if paths.DATA.exists() else paths.ROOT).free
    man["expected_bytes"] = expected
    man["free_bytes_at_start"] = free
    if expected and expected > free * 0.9:
        man.update(status="blocked", error=f"insufficient disk: need {expected} bytes, {free} free",
                   files=plan, finished_at=utcnow())
        write_json(_manifest_path(key), man)
        return man

    prev_ok = {f["name"]: f for f in prev.get("files", []) if f.get("ok")}
    results: list[dict] = []

    def one(f: dict) -> dict:
        dest = files_root / f["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(f)
        try:
            if resume and dest.exists() and f["name"] in prev_ok:
                old = prev_ok[f["name"]]
                if dest.stat().st_size == old.get("bytes"):
                    old = dict(old)
                    old["reused"] = True
                    return old
            if dest.exists():
                dest.chmod(dest.stat().st_mode | stat.S_IWUSR)
            if src["kind"] == "hf_dataset":
                p = _hf_fetch(src, f, files_root)
                rec.update(_hf_verify(f, p))
            elif src["kind"] == "http_files":
                _http_download(f["url"], dest, f.get("size"))
                h = file_hashes(dest, ("sha256", "md5", "sha1"))
                rec.update(h)
                rec["ok"] = h["md5"] == f["md5"] and h["sha1"] == f["sha1"]
                rec["verified_against"] = "archive_org_md5_sha1"
            elif src["kind"] == "github_files":
                _http_download(f["url"], dest, None)
                rec["git_blob_sha1_measured"] = _git_blob_sha1(dest)
                rec.update(file_hashes(dest, ("sha256",)))
                rec["ok"] = rec["git_blob_sha1_measured"] == f["git_blob_sha1"]
                rec["verified_against"] = "github_git_blob_sha1_at_pinned_commit"
            elif src["kind"] == "local_release":
                shutil.copyfile(f["src"], dest)
                rec.update(file_hashes(dest, ("sha256",)))
                rec["ok"] = f["sha256_expected"] is None or rec["sha256"] == f["sha256_expected"]
                rec["verified_against"] = "release SHA256SUMS.txt" if f["sha256_expected"] else "none (checksums file itself)"
            elif src["kind"] == "gcs_bucket":
                _http_download(f["url"], dest, f.get("size"))
                rec["ok"] = _md5_b64(dest) == f["gcs_md5_b64"]
                rec["verified_against"] = "gcs_md5"
            elif src["kind"] == "gdrive_files":
                if "gsheet_id" in f:
                    url = f"https://docs.google.com/spreadsheets/d/{f['gsheet_id']}/export?format=csv"
                    r = requests.get(url, headers=UA, timeout=120)
                    if r.status_code != 200 or "text/html" in r.headers.get("content-type", ""):
                        raise PermissionError(f"sheet export HTTP {r.status_code} {r.headers.get('content-type')}")
                    dest.write_bytes(r.content)
                    rec["response"] = {"content_type": r.headers.get("content-type")}
                else:
                    rec["response"] = _gdrive_download(f["gdrive_id"], dest)
                rec.update(file_hashes(dest, ("sha256", "md5")))
                rec["ok"] = True
                rec["verified_against"] = "none_upstream_publishes_no_checksum; measured hash pinned"
            rec["bytes"] = dest.stat().st_size if dest.exists() else (files_root / f["name"]).stat().st_size
            if rec.get("ok"):
                _make_readonly(dest)
            else:
                rec["error"] = "checksum mismatch"
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = redact(f"{type(e).__name__}: {e}")[:1000]
        return rec

    t0 = time.time()
    workers = max_workers if src["kind"] == "gcs_bucket" else 4
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, f) for f in plan]
        for i, fu in enumerate(as_completed(futs)):
            results.append(fu.result())
            if src["kind"] == "gcs_bucket" and i % 2000 == 0:
                log_event("acquire_progress", source=key, done=i + 1, total=len(plan))
    results.sort(key=lambda r: r["name"])
    n_ok = sum(1 for r in results if r.get("ok"))
    errors = [r for r in results if not r.get("ok")]
    if not plan:
        status = "failed"
    elif n_ok == len(plan):
        status = "complete"
    elif n_ok == 0 and errors and all("PermissionError" in (r.get("error") or "") or "403" in (r.get("error") or "") or "401" in (r.get("error") or "") for r in errors):
        status = "blocked"
    else:
        status = "partial"
    man.update(
        status=status,
        files=results,
        n_files_expected=len(plan),
        n_files_ok=n_ok,
        remaining_files=[r["name"] for r in errors],
        bytes_ok=sum(r.get("bytes") or 0 for r in results if r.get("ok")),
        seconds=round(time.time() - t0, 1),
        finished_at=utcnow(),
    )
    man["retrieval_history"].append({"at": man["finished_at"], "status": status, "n_ok": n_ok, "n_expected": len(plan)})
    write_json(_manifest_path(key), man)
    log_event("acquire_done", source=key, status=status, n_ok=n_ok, n=len(plan),
              errors=[{"name": r["name"], "error": r.get("error")} for r in errors[:5]])
    return man


def acquire(keys: list[str] | None = None, resume: bool = True) -> list[dict]:
    problems = check_registry_covers_catalogue()
    if problems:
        raise SystemExit("registry/catalogue mismatch: " + "; ".join(problems))
    out = []
    for src in load_registry():
        if keys and src["key"] not in keys:
            continue
        log_event("acquire_start", source=src["key"], kind=src["kind"])
        out.append(acquire_source(src, resume=resume))
    return out


def acquisition_status() -> list[dict]:
    rows = []
    for src in load_registry():
        mp = _manifest_path(src["key"])
        if mp.exists():
            m = read_json(mp)
            rows.append({k: m.get(k) for k in ("key", "catalogue_name", "status", "n_files_ok", "n_files_expected",
                                               "bytes_ok", "remaining_files", "error", "continuation_command", "finished_at")})
        else:
            rows.append({"key": src["key"], "catalogue_name": src["catalogue_name"], "status": "not_started",
                         "continuation_command": CONTINUE_CMD.format(key=src["key"])})
    return rows
