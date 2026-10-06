"""Adapters for archive/file-based sources: Scientific Papers, PG-19, PERSUADE 2.0, CoAuthor."""

from __future__ import annotations

import ast
import collections
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from aidet_eval import spans as S
from aidet_eval.adapters.common import Inventory, Rec, contiguous_excerpt, hash_rank, paragraph_spans
from aidet_eval.util import sha256_text

RAW = Path("data/raw")
XMATH = re.compile(r"@xmath\d*|@xcite")


# ===================================================================== Scientific Papers

def _sci_items(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            yield json.loads(line)


def normalise_scientific_papers(src):
    inv = Inventory(src["key"])
    rows = []
    for cfg in ("arxiv", "pubmed"):
        zf = zipfile.ZipFile(RAW / "scientific_papers" / "files" / f"{cfg}-dataset.zip")
        for split, member in (("train", "train.txt"), ("validation", "val.txt"), ("test", "test.txt")):
            name = f"{cfg}-dataset/{member}"
            # pass 1: rank all article ids (full population) -> keep 3000 per split/config
            ids = []
            n = 0
            for d in _sci_items(zf, name):
                n += 1
                ids.append((hash_rank("sci", cfg, d["article_id"]), d["article_id"]))
                if n == 1:
                    inv.schemas[name] = sorted(d.keys())
            inv.rows_by_file[name] = n
            keep = {a for _, a in sorted(ids)[: (4000 if split == "train" else 1500)]}
            for d in _sci_items(zf, name):
                aid = d["article_id"]
                if aid not in keep:
                    continue
                grp = f"sci_article:{cfg}:{aid}"
                common = dict(dataset_key="scientific_papers", dataset_id="armancohan/long-summarization",
                              revision="archive.org md5 " + ("6242aaf5cfcc7814473eee8b779c1b9f" if cfg == "arxiv" else "3ae396f2b690253e7379a038b410300c"),
                              original_row_id=aid, config=cfg, original_split=split, source_group_id=grp,
                              leakage_keys=[grp], independence_key=grp, domain="research",
                              rights={"licence": "unknown; per-article arXiv/PMC licences not audited (research use)"})
                # abstract (HF loader convention: join paragraphs with \n, remove <S> </S>)
                raw_abs = "\n".join(d["abstract_text"])
                abs_text = raw_abs.replace("<S>", "").replace("</S>", "")
                abs_text_stripped = abs_text.strip()
                ra = Rec(**common, text=abs_text_stripped, variant="abstract", subdomain="abstract",
                         original_labels={"article_id": aid, "part": "abstract"}, original_text_sha256=sha256_text(raw_abs),
                         canonicalisation={"transformations": ["join abstract_text with \\n", "remove <S> and </S> markers (HF loader convention)", "strip outer whitespace"],
                                           "offset_mapping": "not needed (no spans on human-only text)"})
                # body passage
                body = "\n".join(d["article_text"])
                ex = contiguous_excerpt(paragraph_spans(body), body, f"sci|{cfg}|{aid}", 120, 450)
                recs = [ra]
                if ex:
                    a, b = ex
                    rb = Rec(**common, text=body[a:b], variant="body", subdomain="body_passage",
                             original_labels={"article_id": aid, "part": "body"}, parent_ids=[f"sci:{cfg}:{aid}:article_text"],
                             parent_excerpt={"parent_id": f"sci:{cfg}:{aid}:article_text", "start": a, "end": b,
                                             "parent_sha256": sha256_text(body), "policy": "paragraph-aligned contiguous 120-450 words, hash-chosen start"})
                    recs.append(rb)
                for r in recs:
                    r.authorship, r.ai_involvement, r.authorship_basis = "human", "no", "historical_release_date"
                    r.ai_char_spans, r.annotation_completeness, r.annotation_provenance = [], "complete", "pre_llm_release"
                    r.human_provenance_category = "historical_pre_llm_corpus"
                    r.provenance_tier = "independently_audited"
                    r.verified_dates = {"corpus_release": "2018 (Cohan et al., NAACL 2018)"}
                    r.human_provenance_evidence = {"evidence": f"{cfg} papers in the 2018 long-summarization corpus",
                                                   "limitations": "scholarly authorship; extraction pipeline lowercases and tokenises text"}
                    r.artefact_flags = ["tokenised_lowercased_extraction"]
                    if XMATH.search(r.text):
                        r.artefact_flags.append("math_or_citation_placeholder")
                        r.eligibility, r.eligibility_reasons = "excluded", ["extraction_artifact_math_or_citation_placeholder"]
                    if S.word_count(r.text) < 40:
                        r.eligibility, r.eligibility_reasons = "excluded", r.eligibility_reasons + ["too_short_after_extraction"]
                    rows.append(r)
                inv.count("artefact:xmath_in_abstract", cfg, int(bool(XMATH.search(abs_text))))
    inv.notes.append("Text is lowercased and space-tokenised by the original extraction; kept verbatim and flagged.")
    return rows, inv


# ================================================================================= PG-19

def normalise_pg19(src):
    inv = Inventory(src["key"])
    rows = []
    md = pd.read_csv(RAW / "pg19" / "files" / "metadata.csv", header=None, names=["book_id", "short_book_title", "publication_date", "url"], dtype=str)
    meta = {r.book_id: r for r in md.itertuples(index=False)}
    by_split = collections.defaultdict(list)
    for p in sorted((RAW / "pg19" / "files").glob("*/*.txt")):
        by_split[p.parent.name].append(p)
    for split, files in by_split.items():
        inv.rows_by_file[f"{split}/*.txt"] = len(files)
        ranked = sorted((hash_rank("pg19", p.stem), p) for p in files)[: (5000 if split == "train" else len(files))]
        for _, p in ranked:
            bid = p.stem
            text = p.read_text(encoding="utf-8")
            m = meta.get(bid)
            title = m.short_book_title if m else None
            author = title.rsplit(" by ", 1)[1].strip() if title and " by " in title else None
            # skip the first and last 10% (front/back matter), paragraph-aligned excerpt
            n = len(text)
            lo, hi = int(n * 0.1), int(n * 0.9)
            paras = [(a, b) for a, b in paragraph_spans(text, "\n\n") if a >= lo and b <= hi]
            ex = contiguous_excerpt(paras, text, f"pg19|{bid}", 150, 450)
            if ex is None:
                inv.count("no_excerpt", split)
                continue
            a, b = ex
            rec = Rec(dataset_key="pg19", dataset_id="deepmind/pg19 (gs://deepmind-gutenberg)", revision="gcs object md5s in acquisition manifest",
                      original_row_id=bid, text=text[a:b], config="default", original_split=split,
                      source_group_id=f"pg19_book:{bid}", leakage_keys=[f"pg19_book:{bid}"],
                      independence_key=f"pg19_author:{author}" if author else f"pg19_book:{bid}",
                      domain="creative", subdomain="historical_literature", source_urls=[m.url] if m else [],
                      parent_ids=[f"pg19:{bid}"],
                      parent_excerpt={"parent_id": f"pg19:{bid}", "start": a, "end": b, "parent_sha256": sha256_text(text),
                                      "policy": "paragraph-aligned contiguous 150-450 words within middle 80% of book, hash-chosen start"},
                      original_labels={"book_id": bid, "short_book_title": title, "publication_date": m.publication_date if m else None},
                      rights={"licence": "Apache-2.0 (dataset card); underlying Project Gutenberg public-domain texts"},
                      extensions={"author_parsed_from_title": author})
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "historical_publication"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "pre_llm_release"
            rec.human_provenance_category = "historical_publication"
            rec.provenance_tier = "independently_audited"
            rec.verified_dates = {"edition_publication_date": m.publication_date if m else None, "note": "PG-19 metadata edition date; books published before 1919"}
            rec.human_provenance_evidence = {"evidence": "Project Gutenberg book in PG-19 (published before 1919)",
                                             "limitations": "PG-19 removed boilerplate and substituted offensive words; translations possible"}
            rec.artefact_flags = ["pg19_hard_wrapped_lines"] if "\n" in rec.text else []
            rows.append(rec)
    inv.facts["books_in_metadata"] = len(md)
    inv.facts["author_grouping"] = "author parsed from 'Title by Author' short title when present; else book id"
    return rows, inv


# ============================================================================== PERSUADE

def normalise_persuade(src):
    inv = Inventory(src["key"])
    rows = []
    usecols = ["essay_id", "essay_id_comp", "competition_set", "full_text", "holistic_essay_score", "provider", "task",
               "prompt_name", "grade_level", "ell_status", "essay_word_count"]
    tr = pd.read_csv(RAW / "persuade" / "files" / "persuade_train_download", dtype=str, usecols=usecols, keep_default_na=False)
    inv.rows_by_file["persuade_corpus_2.0_train.csv (discourse rows)"] = len(tr)
    zf = zipfile.ZipFile(RAW / "persuade" / "files" / "persuade_test_download")
    with zf.open("persuade_corpus_2.0_test.csv", pwd=b"persuade_test") as fh:
        te = pd.read_csv(fh, dtype=str, usecols=lambda c: c in usecols, keep_default_na=False)
    inv.rows_by_file["persuade_corpus_2.0_test.csv (discourse rows)"] = len(te)
    for split, df in (("train", tr), ("test", te)):
        ess = df.drop_duplicates("essay_id")
        inv.count("essays", split, len(ess))
        for c in ("prompt_name", "task", "provider", "grade_level", "holistic_essay_score"):
            if c in ess:
                for v, n in ess[c].value_counts().items():
                    inv.count(f"{split}:{c}", v, int(n))
        dup_text = ess.groupby("essay_id")["full_text"].nunique()
        inv.count("essay_ids_with_inconsistent_full_text", split, int((dup_text > 1).sum()))
        for r in ess.itertuples(index=False):
            rec = Rec(dataset_key="persuade", dataset_id="scrosseye/persuade_corpus_2.0", revision="gdrive sha256 in acquisition manifest",
                      original_row_id=r.essay_id, text=r.full_text, config="persuade_2.0", original_split=split,
                      source_group_id=f"persuade_essay:{r.essay_id}", leakage_keys=[f"persuade_essay:{r.essay_id}"],
                      independence_key=f"persuade_essay:{r.essay_id}", domain="student", subdomain=r.task,
                      original_labels={"essay_id": r.essay_id, "essay_id_comp": r.essay_id_comp, "competition_set": r.competition_set,
                                       "prompt_name": r.prompt_name, "task": r.task, "holistic_essay_score": r.holistic_essay_score,
                                       "grade_level": r.grade_level, "ell_status": r.ell_status, "provider": r.provider},
                      rights={"licence": "CC-BY-NC-SA-4.0", "sensitivity": "student writing; do not redistribute texts"},
                      extensions={"proficiency_score": r.holistic_essay_score, "grade_level": r.grade_level})
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "corpus_collection"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "pre_llm_collection"
            rec.human_provenance_category = "student_essay_corpus"
            rec.provenance_tier = "independently_audited"
            rec.verified_dates = {"collection": None, "note": "collection dates not published per essay; not inferred"}
            rec.human_provenance_evidence = {"evidence": "PERSUADE 2.0 original release (Crossley et al., 2024), grades 6-12 US students",
                                             "limitations": "writer IDs not released; each essay treated as its own writer unit"}
            rows.append(rec)
    inv.notes.append("Rows are discourse elements; essays deduplicated by essay_id. Test zip password published in the author README.")
    return rows, inv


# ============================================================================== CoAuthor

def _apply_ops(doc: list, owners: list, ops: list, owner: str):
    pos = 0
    for op in ops:
        if "retain" in op:
            pos += op["retain"]
        elif "insert" in op:
            ins = op["insert"]
            doc[pos:pos] = list(ins)
            owners[pos:pos] = [owner] * len(ins)
            pos += len(ins)
        elif "delete" in op:
            del doc[pos:pos + op["delete"]]
            del owners[pos:pos + op["delete"]]


def normalise_coauthor(src):
    inv = Inventory(src["key"])
    rows = []
    zf = zipfile.ZipFile(RAW / "coauthor" / "files" / "coauthor_sessions_download")
    meta = pd.read_csv(RAW / "coauthor" / "files" / "coauthor_metadata.csv", dtype=str)
    worker = dict(zip(meta["session_id"], meta["worker_id"]))
    names = [n for n in zf.namelist() if n.endswith(".jsonl")]
    inv.rows_by_file["sessions"] = len(names)
    replay_ok = 0
    for n in sorted(names):
        sid = Path(n).stem
        events = [json.loads(x) for x in zf.read(n).decode("utf-8").splitlines() if x.strip()]
        inv.count("events", "total", len(events))
        doc, owners = [], []
        consistent = True
        for ev in events:
            name = ev.get("eventName")
            inv.count("eventName", name)
            if name == "system-initialize":
                init = ev.get("currentDoc") or ""
                doc, owners = list(init), ["prompt"] * len(init)
            td = ev.get("textDelta")
            if isinstance(td, dict) and td.get("ops"):
                owner = "api" if ev.get("eventSource") == "api" or name == "suggestion-select" else "user"
                try:
                    _apply_ops(doc, owners, td["ops"], owner)
                except Exception:  # noqa: BLE001
                    consistent = False
            cc = ev.get("currentCursor")
            if isinstance(cc, int) and cc > len(doc):
                consistent = False
        replay_ok += consistent
        inv.count("replay_consistent", consistent)
        text = "".join(doc)
        ai = [i for i, o in enumerate(owners) if o == "api"]
        rec = Rec(dataset_key="coauthor", dataset_id="coauthor.stanford.edu (chi2022-coauthor-v1.0)", revision="gdrive sha256 in acquisition manifest",
                  original_row_id=sid, text=text, config="v1.0", original_split="all", source_group_id=f"coauthor_session:{sid}",
                  leakage_keys=[f"coauthor_session:{sid}"], independence_key=f"coauthor_writer:{worker.get(sid, sid)}",
                  domain="creative_or_argumentative", subdomain="coauthor_session", generator="GPT-3 (2021)",
                  original_labels={"worker_id": worker.get(sid)},
                  rights={"licence": "not stated for the dataset (interface code MIT)"})
        rec.authorship = "mixed" if ai else "unknown"
        rec.ai_involvement = "yes" if ai else "unknown"
        rec.mix_mechanism = "interactive_cowriting_legacy_gpt3"
        rec.ai_char_spans = [[a, b] for a, b in S.union((i, i + 1) for i in ai)] if consistent else None
        rec.annotation_completeness = "partial"
        rec.annotation_provenance = "edit_log_replay_unverified"
        rec.eligibility = "reserve"
        rec.eligibility_reasons = ["reserve_source: rights not stated; legacy GPT-3 generator; replay not independently validated"]
        rows.append(rec)
    inv.facts["sessions"] = len(names)
    inv.facts["sessions_with_metadata"] = len(meta)
    inv.facts["distinct_workers_in_metadata"] = int(meta["worker_id"].nunique())
    inv.facts["replay_consistent_sessions"] = replay_ok
    return rows, inv
