"""Consolidated machine-readable inventory of all catalogue sources + catalogue audit.

Writes data/inventory/inventory.json (one entry per catalogue source) and
data/inventory/source_audit.json (discrepancies/corrections; the original catalogue is unchanged).
"""

from __future__ import annotations

from aidet_eval import paths
from aidet_eval.acquire import load_catalogue, load_registry
from aidet_eval.util import read_json, utcnow, write_json

# Evidence-based decisions recorded during acquisition/normalisation (see execution log).
DECISIONS = {
    "llmtrace_detection": dict(eligibility="primary: controlled_replacement (fill_gaps), fully_ai, source-matched human references",
                               provenance="human rows are web-derived benchmark-provided references; dates not audited",
                               rights="Apache-2.0 dataset card; underlying web sources not individually cleared"),
    "llmtrace_classification": dict(eligibility="primary: fully_ai candidates and matched human references (linked to detection via topic_id/text)",
                                    provenance="benchmark-provided; not automatically independent of the detection release",
                                    rights="Apache-2.0 dataset card"),
    "raid": dict(eligibility="primary: fully_ai (clean attack=='none' labelled training release); human rows as benchmark-provided references; attacks reserve; test.csv unlabelled (never scored)",
                 provenance="RAID human source documents; collection dates not independently audited", rights="MIT dataset card"),
    "damasha": dict(eligibility="diagnostic panel only (mechanism/domain/generator unknown at row level); attacked files reserve",
                    provenance="reuses MAGE/M4/XSum/essay corpora per dataset card", rights="CC-BY-4.0 dataset card"),
    "aitdna": dict(eligibility="primary: aitdna_real_cowriting (all 362 unique texts)", provenance="logged writing sessions (edit-log lineage)",
                   rights="CC-BY-SA-4.0 per dataset card (catalogue listed GPL-2.0)"),
    "beemo": dict(eligibility="primary: beemo_matched_editing (triplets)", provenance="human_output from No Robots annotators; human_edits are expert edits of AI output (not human-only)",
                  rights="field-dependent (CC-BY-NC-4.0 / MIT / generator terms)"),
    "imdb": dict(eligibility="primary: audited_human_core reviews + calibration", provenance="2011 corpus release (pre-LLM); sentiment labels are not authorship labels",
                 rights="IMDb terms; research corpus"),
    "xsum": dict(eligibility="primary: audited_human_core news + calibration", provenance="2018 BBC article corpus (pre-LLM); 233 documents overlap EditLens training data",
                 rights="BBC content rights not cleared; research use"),
    "scientific_papers": dict(eligibility="primary: audited_human_core research (abstracts + body passages) + calibration; @xmath/@xcite passages excluded",
                              provenance="2018 arXiv/PubMed corpus; text lowercased and tokenised by original extraction",
                              rights="per-article licences not audited; research use"),
    "wikitext": dict(eligibility="primary: audited_human_core expository + calibration", provenance="2016 WikiText-103 (Wikipedia Good/Featured articles); tokenised",
                     rights="CC-BY-SA (Wikipedia)"),
    "pg19": dict(eligibility="primary: audited_human_core creative (historical literature) + calibration", provenance="books published before 1919",
                 rights="public-domain Project Gutenberg texts; Apache-2.0 dataset packaging"),
    "persuade": dict(eligibility="primary: student_human + calibration", provenance="PERSUADE 2.0 original release (grades 6-12 US students); collection dates not published",
                     rights="CC-BY-NC-SA-4.0; student writing: do not redistribute texts"),
    "no_robots": dict(eligibility="POISON (from benchmark-v3): excluded from evaluation and calibration; its human responses set the pooled threshold. Used in v1/v2 as modern_human_instructions + calibration",
                      provenance="curator-attested human annotators (2023)", rights="CC-BY-NC-4.0"),
    "mage": dict(eligibility="reserve (in-distribution regression source; hashes kept for overlap audits)",
                 provenance="label mapping verified 1=human, 0=machine", rights="Apache-2.0 dataset card"),
    "editlens_iclr": dict(eligibility="in-family reference panel (test split only); train/val define Pangram training exposure",
                          provenance="EditLens generation pipeline; not independent evidence for EditLens-trained baselines",
                          rights="CC-BY-NC-SA-4.0, gated (existing authorised access)"),
    "coauthor": dict(eligibility="reserve (dataset licence not stated; legacy GPT-3; replay not independently validated)",
                     provenance="logged co-writing sessions (2021, GPT-3)", rights="dataset licence not stated; interface code MIT"),
}

AUDIT_NOTES = [
    {"source": "aitdna", "field": "license", "catalogue": "GPL-2.0", "observed": "cc-by-sa-4.0 (dataset card at pinned revision)"},
    {"source": "aitdna", "field": "rows", "catalogue": "362 unique texts, 3,258 displayed rows", "observed": "9 views x 362 rows; span view concatenation equals document text for all 362"},
    {"source": "damasha", "field": "clean rows", "catalogue": "paper clean_combined 95,859", "observed": "96,692 rows in DAMASHA_Final_No_ADV.csv"},
    {"source": "damasha", "field": "attacked rows", "catalogue": "671,013 including six attack variants", "observed": "6 x 96,692 attacked + 96,692 clean = 676,844"},
    {"source": "damasha", "field": "row metadata", "catalogue": "domain/generator/mechanism mapping to validate", "observed": "release has only hybrid_text and 'has_pair ' (trailing space); no row-level domain/generator/mechanism"},
    {"source": "coauthor", "field": "sessions", "catalogue": "1445 sessions, 63 writers", "observed": "1447 session files; metadata sheet covers 830 sessions with 57 distinct worker_ids"},
    {"source": "coauthor", "field": "license", "catalogue": "not verified", "observed": "dataset licence not stated on project site; interface repo MIT"},
    {"source": "scientific_papers", "field": "acquisition", "catalogue": "HF armanc/scientific_papers", "observed": "HF repo is a legacy loader script only; author README links Google Drive + archive.org mirror (used, MD5/SHA1 verified)"},
    {"source": "scientific_papers", "field": "text form", "catalogue": "article/abstract", "observed": "original extraction is lowercased and space-tokenised; arXiv contains @xmath/@xcite placeholders"},
    {"source": "pg19", "field": "acquisition", "catalogue": "HF deepmind/pg19", "observed": "HF repo is a legacy loader; original GCS bucket deepmind-gutenberg used (28,753 objects, MD5 verified)"},
    {"source": "wikitext", "field": "articles", "catalogue": "rows are not documents", "observed": "28,478 train articles reconstructed from top-level headings (published figure 28,475)"},
    {"source": "llmtrace_detection", "field": "data_type semantics", "catalogue": "domain_field data_type", "observed": "'story' human sources are film plot summaries; 'article' are web blog articles; 'factual' are Wikipedia paragraphs"},
    {"source": "llmtrace_detection", "field": "fill_gaps generators", "catalogue": "-", "observed": "all 13,674 English fill_gaps mixtures generated by gemini-2.5-flash"},
    {"source": "raid", "field": "books domain", "catalogue": "english prose domain", "observed": "book plot summaries (mapped to narrative_summary, not creative)"},
    {"source": "no_robots", "field": "prompt_id", "catalogue": "group_field prompt_id", "observed": "prompt_id present; one prompt_id repeated within train"},
    {"source": "beemo", "field": "duplicates", "catalogue": "deduplicate repeated human references", "observed": "2,187 rows, 2,187 unique prompt_id and human_output (no repeats within Beemo); all linked to No Robots"},
    {"source": "xsum", "field": "overlap", "catalogue": "may overlap detector benchmarks", "observed": "233 XSum documents exactly/near-duplicated in EditLens train/val (known Pangram training exposure)"},
    {"source": "persuade", "field": "release", "catalogue": "over 25,000 essays", "observed": "25,995 essays (train 15,593 + password-protected test 10,402); rows are discourse elements"},
    {"source": "imdb", "field": "grouping", "catalogue": "retain source identity", "observed": "HF parquet lacks movie IDs; same-movie reviews cannot be grouped"},
]


def run_inventory(keys: list[str] | None = None) -> dict:
    cat = {s["name"]: s for s in load_catalogue()["sources"]}
    out = []
    for src in load_registry():
        if keys and src["key"] not in keys:
            continue
        acq_p = paths.RAW / src["key"] / "_acquisition.json"
        inv_p = paths.INVENTORY / f"{src['key']}.measured.json"
        acq = read_json(acq_p) if acq_p.exists() else {"status": "not_started"}
        inv = read_json(inv_p) if inv_p.exists() else {}
        c = cat.get(src["catalogue_name"]) or {"name": src["key"], "dataset_id": src.get("repo") or src.get("repo_url") or src.get("path"),
                                                "url": src.get("repo_url"), "note": "extra source added for benchmark-v2 at user request; not a catalogue entry"}
        files = acq.get("files", [])
        entry = {
            "source_id": src["key"], "name": src["catalogue_name"],
            "upstream": c.get("url") or c.get("dataset_id"), "resolved_location": src.get("repo") or src.get("bucket") or src.get("repo_url") or src.get("path") or [f.get("url") if isinstance(f, dict) else f for f in src.get("files", [])],
            "repository_migration": src.get("loader_repo") and f"catalogue id {c.get('dataset_id')} is a legacy loader; data acquired from original release location",
            "resolved_revision": src.get("revision") or src.get("loader_revision"),
            "retrieved_at": acq.get("finished_at"),
            "acquisition_status": acq.get("status"),
            "download_completeness": {"n_files_expected": acq.get("n_files_expected"), "n_files_ok": acq.get("n_files_ok"),
                                      "bytes_ok": acq.get("bytes_ok"), "remaining_files": acq.get("remaining_files", [])},
            "continuation_command": acq.get("continuation_command"),
            "file_list": [{k: f.get(k) for k in ("name", "bytes", "sha256", "verified_against", "ok")} for f in files[:60]],
            "n_files": len(files),
            "acquisition_manifest": str(acq_p.relative_to(paths.ROOT)),
            "reported_counts": {k: c[k] for k in ("reported_rows", "rows", "unique_texts", "paper_counts", "book_counts", "sessions",
                                                  "released_rows_as_viewed", "reported_scale", "writers") if k in c},
            "measured_counts": {"rows_by_file": inv.get("measured_rows_by_file"), "rows_total": inv.get("measured_rows_total"),
                                **(inv.get("measured_counts") or {})},
            "schema": inv.get("schemas"), "facts": inv.get("facts"), "notes": inv.get("notes"),
            "candidate_pool": {"path": inv.get("normalised_path"), "n": inv.get("n_candidate_records"),
                               "eligibility": inv.get("candidate_eligibility"), "authorship": inv.get("candidate_authorship"),
                               "domain": inv.get("candidate_domain")},
            "span_encoding": (inv.get("facts") or {}).get("span_encoding"),
            "applicable_terms": c.get("license_as_listed") or c.get("license_note"),
            "catalogue_entry_exists": src["catalogue_name"] is not None,
            **{f"decision_{k}": v for k, v in DECISIONS.get(src["key"], {"eligibility": src.get("primary_role"), "provenance": "see adapter notes",
                                                                         "rights": "see measured inventory / dataset card"}).items()},
            "catalogue_entry": c,
        }
        if src["catalogue_name"] is None:
            entry["name"] = src["key"]
        out.append(entry)
    doc = {"created_at": utcnow(), "n_sources": len(out), "sources": out,
           "statuses": {e["source_id"]: e["acquisition_status"] for e in out}}
    write_json(paths.INVENTORY / "inventory.json", doc)
    write_json(paths.INVENTORY / "source_audit.json", {"created_at": utcnow(), "catalogue_sha256": __import__("aidet_eval.util", fromlist=["x"]).file_hashes(paths.CATALOGUE)["sha256"],
                                                       "note": "Corrections/discrepancies vs the original catalogue (kept unchanged in spec/)", "findings": AUDIT_NOTES})
    print(f"inventory: {len(out)} sources -> {paths.INVENTORY / 'inventory.json'}")
    for e in out:
        print(f"  {e['source_id']:24s} {e['acquisition_status']:9s} rows={e['measured_counts'].get('rows_total')}")
    return doc
