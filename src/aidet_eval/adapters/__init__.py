"""Source normalisation driver: raw release -> data/normalised/<key>.parquet + data/inventory/<key>.measured.json."""

from __future__ import annotations

import time

import pandas as pd

from aidet_eval import paths
from aidet_eval.acquire import load_registry
from aidet_eval.adapters import corpora, extra_sources, hf_sources
from aidet_eval.adapters.common import JSON_COLS, Rec, dumps
from aidet_eval.util import log_event, utcnow, write_json

ADAPTERS = {
    "llmtrace_detection": hf_sources.normalise_llmtrace_detection,
    "llmtrace_classification": hf_sources.normalise_llmtrace_classification,
    "raid": hf_sources.normalise_raid,
    "damasha": hf_sources.normalise_damasha,
    "aitdna": hf_sources.normalise_aitdna,
    "beemo": hf_sources.normalise_beemo,
    "imdb": hf_sources.normalise_imdb,
    "xsum": hf_sources.normalise_xsum,
    "scientific_papers": corpora.normalise_scientific_papers,
    "wikitext": hf_sources.normalise_wikitext,
    "pg19": corpora.normalise_pg19,
    "persuade": corpora.normalise_persuade,
    "no_robots": hf_sources.normalise_no_robots,
    "mage": hf_sources.normalise_mage,
    "editlens_iclr": hf_sources.normalise_editlens_iclr,
    "coauthor": corpora.normalise_coauthor,
    # extra (user-requested) sources for benchmark-v2
    "storyscope": extra_sources.normalise_storyscope,
    "hap_e2": extra_sources.normalise_hap_e2,
    "lamp": extra_sources.normalise_lamp,
    "cnn_dailymail": extra_sources.normalise_cnn_dailymail,
    "generate_heterogeneous": extra_sources.normalise_generate_heterogeneous,
}
# derived pools (no acquisition; computed from another normalised source)
DERIVED = {
    "wikitext_detok": extra_sources.normalise_wikitext_detok,
    "scientific_papers_detok": extra_sources.normalise_scientific_papers_detok,
}


def records_to_frame(rows: list) -> pd.DataFrame:
    if rows and isinstance(rows[0], Rec):
        df = pd.DataFrame([r.to_row() for r in rows])
        for c in JSON_COLS:
            if c in df:
                df[c] = df[c].map(lambda v: None if v is None else dumps(v))
        return df
    return pd.DataFrame(rows)


def normalise(keys: list[str] | None = None) -> None:
    import os

    os.chdir(paths.ROOT)
    paths.NORMALISED.mkdir(parents=True, exist_ok=True)
    paths.INVENTORY.mkdir(parents=True, exist_ok=True)
    todo = [s for s in load_registry() if not keys or s["key"] in keys]
    todo += [{"key": k, "derived": True} for k in DERIVED if not keys or k in keys]
    for src in todo:
        t0 = time.time()
        log_event("normalise_start", source=src["key"])
        rows, inv = (DERIVED if src.get("derived") else ADAPTERS)[src["key"]](src)
        df = records_to_frame(rows)
        out = paths.NORMALISED / f"{src['key']}.parquet"
        df.to_parquet(out, index=False)
        summary = {"source": src["key"], "created_at": utcnow(), "normalised_path": str(out.relative_to(paths.ROOT)),
                   "n_candidate_records": int(len(df)), **inv.to_dict()}
        if "eligibility" in df:
            summary["candidate_eligibility"] = df["eligibility"].value_counts().to_dict()
            summary["candidate_authorship"] = df["authorship"].value_counts().to_dict()
            summary["candidate_domain"] = df["domain"].astype(str).value_counts().to_dict()
        write_json(paths.INVENTORY / f"{src['key']}.measured.json", summary)
        log_event("normalise_done", source=src["key"], n=int(len(df)), seconds=round(time.time() - t0, 1))
