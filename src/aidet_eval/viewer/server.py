"""Local results viewer: stdlib HTTP server + static single-page app.

* Summary-only scorecards load directly from results.summary.json.
* Full-text inspection uses a SQLite index built lazily from results.full.json (cached next to it,
  keyed by the full file's sha256), so the browser never receives the whole full file.
* Binds to 127.0.0.1 by default; no external requests, analytics or uploads.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
_RESP_CACHE: dict = {}  # results files are immutable per sha, so responses can be cached
_lock = threading.Lock()


def discover(roots: list[str]) -> dict[str, Path]:
    """Map run name -> directory containing results.summary.json (or a direct results file's parent)."""
    runs = {}
    for r in roots:
        p = Path(r)
        if p.is_file():
            runs[p.parent.name] = p.parent
        elif (p / "results.summary.json").exists():
            runs[p.name] = p
        elif p.is_dir():
            for d in sorted(p.iterdir()):
                if (d / "results.summary.json").exists():
                    runs[d.name] = d
    return runs


_SHA_CACHE: dict = {}


def _sha(path: Path) -> str:
    """sha256 of a results file, memoised by (path, size, mtime) so requests never re-hash 200 MB."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def index_path(run_dir: Path) -> Path | None:
    full = run_dir / "results.full.json"
    if not full.exists():
        return None
    sha = _sha(full)
    idx = run_dir / f"viewer_index_{sha[:16]}.sqlite"
    if idx.exists():
        return idx
    with _lock:
        if idx.exists():
            return idx
        data = json.loads(full.read_bytes())
        tmp = idx.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        con.execute("""create table ex (example_id text primary key, role text, panels text, dataset text, domain text, subdomain text,
                       authorship text, ai_involvement text, mechanism text, generator text, frac_bin text, length_bin text, word_count int,
                       provenance_tier text, provenance_category text, leakage text, independence text, record text)""")
        con.execute("""create table pred (example_id text, model text, protocol text, status text, score real, decisions text, record text,
                       primary key (example_id, model, protocol))""")
        rows = []
        for e in data["examples"]:
            t = e["truth"]
            rows.append((e["example_id"], e["role"], json.dumps(e["panel_ids"]), e["source"]["dataset_key"], e["domain"], e["subdomain"],
                         t["authorship"], t["ai_involvement"], t["mix_mechanism"], t["generator"],
                         (e["extensions"].get("slices") or {}).get("ai_fraction_bin"), e["length_bin"], e["word_count"],
                         e["human_provenance"]["tier"], e["human_provenance"]["category"], e["leakage_group_id"],
                         e["independence_group_id"], json.dumps(e, ensure_ascii=False)))
        con.executemany("insert into ex values (" + ",".join("?" * 18) + ")", rows)
        prow = [(p["example_id"], p["model_id"], p["protocol_id"], p["status"], p["score"],
                 json.dumps({d["operating_point_id"]: d["flag_ai"] for d in p["operating_point_decisions"]}),
                 json.dumps(p, ensure_ascii=False)) for p in data["predictions"]]
        con.executemany("insert into pred values (?,?,?,?,?,?,?)", prow)
        meta = {"calibrations": data["calibrations"], "status": data["status"], "run_id": data["run_id"]}
        con.execute("create table meta (k text, v text)")
        con.execute("insert into meta values ('meta', ?)", (json.dumps(meta),))
        for c in ("dataset", "domain", "authorship", "role"):
            con.execute(f"create index ix_{c} on ex({c})")
        con.commit()
        con.close()
        tmp.rename(idx)
    return idx


FILTER_COLS = {"role", "dataset", "domain", "subdomain", "authorship", "ai_involvement", "mechanism", "generator", "frac_bin",
               "length_bin", "provenance_tier", "provenance_category"}


class Handler(SimpleHTTPRequestHandler):
    runs: dict[str, Path] = {}

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path: Path, ctype: str):
        b = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(b)

    def end_headers(self):
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/api/runs":
                return self._json(run_list(self.runs))
            run = self.runs.get(q.get("run", ""))
            if u.path.startswith("/api/") and run is None:
                return self._json({"error": "unknown run"}, 404)
            if u.path == "/api/summary":
                return self._file(run / "results.summary.json", "application/json")
            if u.path == "/api/sensitivity":
                f = run / "calibration_sensitivity.json"
                if not f.exists():
                    return self._json({"error": "no calibration sensitivity analysis for this run"}, 404)
                return self._file(f, "application/json")
            if u.path == "/api/download":
                name = q.get("name")
                allowed = {"results.summary.cells.csv": "text/csv", "report.md": "text/markdown", "results.summary.json": "application/json"}
                if name not in allowed:
                    return self._json({"error": "not downloadable"}, 400)
                return self._file(run / name, allowed[name])
            if u.path in ("/api/examples", "/api/example", "/api/scores", "/api/meta", "/api/breakdown", "/api/roc", "/api/whatif"):
                idx = index_path(run)
                if idx is None:
                    return self._json({"error": "full results not available for this run (summary-only)"}, 404)
                ck = (str(idx), u.path, tuple(sorted(q.items())))
                if ck not in _RESP_CACHE:
                    con = sqlite3.connect(idx)
                    try:
                        _RESP_CACHE[ck] = self._query(con, u.path, q)
                    finally:
                        con.close()
                    if len(_RESP_CACHE) > 256:
                        _RESP_CACHE.pop(next(iter(_RESP_CACHE)))
                return self._json(_RESP_CACHE[ck])
            return super().do_GET()
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _query(self, con, path, q):
        if path == "/api/breakdown":
            return breakdown(con, q)
        if path == "/api/roc":
            return roc(con, q)
        if path == "/api/whatif":
            return whatif(con, q)
        if path == "/api/meta":
            return json.loads(con.execute("select v from meta where k='meta'").fetchone()[0])
        if path == "/api/example":
            r = con.execute("select record from ex where example_id=?", (q["id"],)).fetchone()
            if not r:
                return {"error": "not found"}
            preds = [json.loads(x[0]) for x in con.execute("select record from pred where example_id=?", (q["id"],))]
            return {"example": json.loads(r[0]), "predictions": preds}
        model, proto, op = q.get("model"), q.get("protocol"), q.get("op", "human_fpr_1pct")
        where, args = [], []
        for c in FILTER_COLS:
            if q.get(c):
                where.append(f"e.{c} = ?")
                args.append(q[c])
        if q.get("panel"):
            where.append("e.panels like ?")
            args.append(f'%"{q["panel"]}"%')
        if path == "/api/scores":
            sql = ("select e.authorship, p.score from ex e join pred p on p.example_id=e.example_id "
                   "where p.model=? and p.protocol=? and p.status='ok'" + "".join(f" and {w}" for w in where))
            rows = con.execute(sql, [model, proto] + args).fetchall()
            return {"scores": rows}
        # /api/examples with outcome filter
        outcome = q.get("outcome", "")
        sql = ("select e.example_id, e.role, e.dataset, e.domain, e.authorship, e.mechanism, e.generator, e.frac_bin, e.length_bin, "
               "e.word_count, e.provenance_tier, p.status, p.score, p.decisions from ex e left join pred p on p.example_id=e.example_id "
               "and p.model=? and p.protocol=?")
        where2 = list(where)
        if outcome == "false_positive":
            where2.append("e.authorship='human' and json_extract(p.decisions, '$.' || ?) = 1")
            args = args + [op]
        elif outcome == "false_negative":
            where2.append("e.ai_involvement='yes' and json_extract(p.decisions, '$.' || ?) = 0")
            args = args + [op]
        elif outcome == "error":
            where2.append("p.status != 'ok'")
        if where2:
            sql += " where " + " and ".join(where2)
        limit = min(int(q.get("limit", 200)), 1000)
        offset = int(q.get("offset", 0))
        rows = con.execute(sql + f" order by p.score desc limit {limit} offset {offset}", [model, proto] + args).fetchall()
        cols = ["example_id", "role", "dataset", "domain", "authorship", "mechanism", "generator", "frac_bin", "length_bin",
                "word_count", "provenance_tier", "status", "score", "decisions"]
        out = [dict(zip(cols, r)) for r in rows]
        if outcome == "disagreement" and q.get("model2"):
            m2 = {r[0]: json.loads(r[1]) for r in con.execute("select example_id, decisions from pred where model=? and protocol=?", (q["model2"], proto))}
            out = [r for r in out if r["decisions"] and r["example_id"] in m2 and json.loads(r["decisions"]).get(op) is not None
                   and m2[r["example_id"]].get(op) is not None and json.loads(r["decisions"]).get(op) != m2[r["example_id"]].get(op)]
            for r in out:
                r["decisions_model2"] = json.dumps(m2[r["example_id"]])
        return {"rows": out, "limit": limit, "offset": offset}


def serve(runs_arg: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    roots = [x for x in runs_arg.split(",") if x]
    Handler.runs = discover(roots)
    if not Handler.runs:
        raise SystemExit(f"no runs with results.summary.json found under {roots}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"aidet_eval viewer: http://{host}:{port}/  ({len(Handler.runs)} runs: {', '.join(Handler.runs)})")
    httpd.serve_forever()


# ------------------------------------------------------------------ analysis endpoints
BREAKDOWN_FIELDS = {"subdomain": "e.subdomain", "dataset": "e.dataset", "generator": "e.generator", "mechanism": "e.mechanism",
                    "length_bin": "e.length_bin", "frac_bin": "e.frac_bin", "domain": "e.domain", "provenance_tier": "e.provenance_tier"}


def _wilson(k: int, n: int) -> tuple[float, float] | None:
    if n == 0:
        return None
    from aidet_eval.metrics import wilson

    return wilson(k, n)


def breakdown(con, q):
    """Official decisions grouped by one field, per authorship class. Intervals: Wilson (assumes independence)."""
    by = BREAKDOWN_FIELDS.get(q.get("by", "subdomain"), "e.subdomain")
    where, args = ["p.model=?", "p.protocol=?", "e.role='evaluation'"], [q.get("model"), q.get("protocol")]
    if q.get("panel"):
        where.append("e.panels like ?")
        args.append(f'%"{q["panel"]}"%')
    if q.get("domain"):
        where.append("e.domain = ?")
        args.append(q["domain"])
    sql = (f"select coalesce({by}, '(none)'), e.authorship, p.status, json_extract(p.decisions, '$.' || ?) "
           f"from ex e join pred p on p.example_id=e.example_id where " + " and ".join(where))
    agg: dict = {}
    for key, auth, status, flag in con.execute(sql, [q.get("op", "human_fpr_1pct")] + args):
        a = agg.setdefault(key, {}).setdefault(auth, {"n": 0, "scored": 0, "flagged": 0})
        a["n"] += 1
        if status == "ok" and flag is not None:
            a["scored"] += 1
            a["flagged"] += int(flag)
    rows = []
    for key, d in agg.items():
        row = {"category": key}
        for auth, a in d.items():
            ci = _wilson(a["flagged"], a["scored"])
            row[auth] = {**a, "rate": (a["flagged"] / a["scored"]) if a["scored"] else None, "ci": ci}
        rows.append(row)
    rows.sort(key=lambda r: -sum(v["n"] for k, v in r.items() if k != "category"))
    return {"by": q.get("by", "subdomain"), "rows": rows, "interval": "Wilson 95%, assumes independent documents (clustering ignored)"}


def roc(con, q):
    """ROC curves per group: negatives = human, positives = ai (task=fully) or ai+mixed (task=any)."""
    import numpy as np

    from aidet_eval.metrics import auroc

    task = q.get("task", "fully")
    group = BREAKDOWN_FIELDS.get(q.get("by", "domain"), "e.domain")
    where, args = ["p.model=?", "p.protocol=?", "e.role='evaluation'", "p.status='ok'"], [q.get("model"), q.get("protocol")]
    humans = q.get("humans", "audited")
    sql = (f"select coalesce({group}, '(none)'), e.authorship, e.panels, p.score from ex e join pred p on p.example_id=e.example_id where "
           + " and ".join(where))
    data: dict = {}
    for key, auth, panels, score in con.execute(sql, args):
        if '"editlens_in_family_reference"' in panels:
            continue  # baselines' own family: not independent evidence
        if auth == "human":
            if humans == "audited" and '"audited_human_core"' not in panels and '"student_human"' not in panels:
                continue
            lab = 0
        elif auth == "ai" or (auth == "mixed" and task == "any"):
            lab = 1
        else:
            continue
        data.setdefault(key, []).append((score, lab))
    meta = json.loads(con.execute("select v from meta where k='meta'").fetchone()[0])
    thr = next((c["threshold"] for c in meta["calibrations"] if c["model_id"] == q.get("model") and c["protocol_id"] == q.get("protocol")
                and c["operating_point_id"] == q.get("op", "human_fpr_1pct")), None)
    out = []
    for key, pts in sorted(data.items()):
        a = np.array(pts, dtype=float)
        s, y = a[:, 0], a[:, 1].astype(int)
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            continue
        order = np.argsort(-s, kind="mergesort")
        s2, y2 = s[order], y[order]
        last = np.r_[np.flatnonzero(np.diff(s2)), len(s2) - 1]  # end of each tie block
        tpr = np.cumsum(y2)[last] / npos
        fpr = np.cumsum(1 - y2)[last] / nneg
        tpr, fpr = np.r_[0, tpr], np.r_[0, fpr]
        keep = np.unique(np.r_[np.linspace(0, len(fpr) - 1, 300).astype(int), np.searchsorted(fpr, np.geomspace(1e-3, 1, 120))]).clip(0, len(fpr) - 1)
        op = None
        if thr is not None:
            op = {"fpr": float(((s > thr) & (y == 0)).sum() / nneg), "tpr": float(((s > thr) & (y == 1)).sum() / npos)}
        out.append({"group": key, "n_pos": npos, "n_neg": nneg, "auroc": auroc(s, y),
                    "fpr": [float(fpr[i]) for i in keep], "tpr": [float(tpr[i]) for i in keep], "operating_point": op})
    return {"task": task, "humans": humans, "threshold": thr, "curves": out}


_RUN_META: dict = {}


def run_list(runs: dict) -> list[dict]:
    """Runs with profile/version/status so the viewer can default to the newest full run and label smoke runs."""
    out = []
    for name, d in runs.items():
        f = d / "results.summary.json"
        key = (str(f), f.stat().st_mtime_ns) if f.exists() else None
        if key and key not in _RUN_META:
            sm = json.loads(f.read_bytes())
            b = sm["benchmark"]
            _RUN_META[key] = {"benchmark_id": b["benchmark_id"], "profile": b["profile"], "status": sm["status"],
                              "data_origin": sm["data_origin"], "n_evaluation": b["realised_counts"]["evaluation"],
                              "n_calibration": b["realised_counts"]["calibration"]}
        meta = _RUN_META.get(key, {})
        out.append({"run": name, "has_full": (d / "results.full.json").exists(), "has_summary": f.exists(), **meta})
    import re as _re

    def order(r):
        m = _re.search(r"-v(\d+)", r.get("benchmark_id", ""))
        return (r.get("profile") != "full", r.get("data_origin") != "measured", -(int(m.group(1)) if m else 0), r["run"])
    return sorted(out, key=order)


WHATIF_PANELS = [("human_fpr", "audited_human_core", "human"), ("student_fpr", "student_human", "human"),
                 ("fully_ai", "fully_ai_core", "ai"), ("sentences", "controlled_replacement", "mixed"),
                 ("blocks", "heterogeneous_block_replacement", "mixed"), ("cowriting", "aitdna_real_cowriting", "mixed")]


def whatif(con, q):
    """For each false-alarm budget: threshold from the human calibration documents (official order-statistic rule,
    flag iff score > threshold), then headline rates on the evaluation panels. Exploratory except 1%/5%."""
    import math

    import numpy as np

    from aidet_eval.metrics import wilson

    protocol = q.get("protocol")
    budgets = [float(x) for x in q.get("budgets", "0.005,0.01,0.02,0.05,0.1").split(",")]
    models = [r[0] for r in con.execute("select distinct model from pred where protocol=?", (protocol,))]
    out = {"budgets": budgets, "models": sorted(models), "rows": []}
    for m in sorted(models):
        cal = np.array([r[0] for r in con.execute(
            "select p.score from pred p join ex e on e.example_id=p.example_id where p.model=? and p.protocol=? and e.role='calibration' and p.status='ok'",
            (m, protocol))], dtype=float)
        ev = con.execute("select e.panels, e.authorship, p.score from pred p join ex e on e.example_id=p.example_id "
                         "where p.model=? and p.protocol=? and e.role='evaluation' and p.status='ok'", (m, protocol)).fetchall()
        groups = {k: np.array([sc for pan, au, sc in ev if f'"{panel}"' in pan and au == cls], dtype=float) for k, panel, cls in WHATIF_PANELS}
        srt = np.sort(cal)[::-1]
        for a in budgets:
            k = math.floor(a * len(srt))
            thr = float(srt[k]) if 0 <= k < len(srt) else None
            row = {"model": m, "budget": a, "threshold": thr, "calibration_fpr": float((cal > thr).mean()) if thr is not None else None}
            for key, arr in groups.items():
                if thr is None or len(arr) == 0:
                    row[key] = None
                    continue
                x = int((arr > thr).sum())
                lo, hi = wilson(x, len(arr))
                row[key] = {"rate": x / len(arr), "k": x, "n": int(len(arr)), "ci": [lo, hi]}
            out["rows"].append(row)
    return out
