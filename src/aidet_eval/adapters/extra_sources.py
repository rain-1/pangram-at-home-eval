"""Adapters for sources added for benchmark-v2 (user request, 2026-10-06), plus detokenised derivatives."""

from __future__ import annotations

import ast
import collections
import glob
import json
import re
from pathlib import Path

import pandas as pd

from aidet_eval import spans as S
from aidet_eval.adapters.common import Inventory, Rec, contiguous_excerpt, hash_rank, paragraph_spans
from aidet_eval.detok import detok_checked
from aidet_eval.util import sha256_text

RAW = Path("data/raw")
NORM = Path("data/normalised")


def _human(r: Rec, category: str, tier: str, evidence: dict, dates: dict | None = None) -> Rec:
    r.authorship, r.ai_involvement, r.authorship_basis = "human", "no", category
    r.ai_char_spans, r.annotation_completeness, r.annotation_provenance = [], "complete", "source_assertion"
    r.human_provenance_category, r.provenance_tier, r.human_provenance_evidence, r.verified_dates = category, tier, evidence, dates
    return r


def _ai(r: Rec, generator: str, task: str | None) -> Rec:
    r.authorship, r.ai_involvement, r.authorship_basis = "ai", "yes", "dataset_generation_record"
    r.generator = generator
    r.ai_char_spans, r.annotation_completeness, r.annotation_provenance = [[0, len(r.text)]], "complete", "dataset_label_assertion"
    r.extensions["generation_task"] = task
    return r


# ================================================================== StoryScope
STORY_MODELS = {"story_gpt": "gpt-5.4", "story_claude": "claude-sonnet-4.6", "story_gemini": "gemini-3-flash",
                "story_deepseek": "deepseek-v3.2", "story_kimi": "kimi-k2.5"}


def normalise_storyscope(src):
    inv = Inventory("storyscope")
    rows = []
    for split in ("test", "dev"):
        df = pd.read_parquet(RAW / "storyscope" / "files" / f"stories_{split}.parquet")
        inv.rows_by_file[f"stories_{split}.parquet"] = len(df)
        inv.schemas[split] = list(df.columns)
        cols = list(STORY_MODELS)
        for r in df.itertuples(index=False):
            # one generator per prompt (hash-assigned) so prompts stay independent; balanced in expectation
            col = cols[int(hash_rank("storyscope", r.prompt_id), 16) % len(cols)]
            story = getattr(r, col)
            inv.count("assigned_generator", STORY_MODELS[col])
            if not isinstance(story, str) or not story.strip():
                inv.count("missing_story", STORY_MODELS[col])
                continue
            paras = [(a, b) for a, b in paragraph_spans(story) if not story[a:b].lstrip().startswith(("#", "**", "*", "---"))]
            ex = contiguous_excerpt(paras, story, f"storyscope|{r.prompt_id}", 150, 450)
            if ex is None:
                inv.count("no_excerpt", split)
                continue
            a, b = ex
            rec = Rec(dataset_key="storyscope", dataset_id=src["repo"], revision=src["revision"], original_row_id=f"{r.prompt_id}:{col}",
                      text=story[a:b], config="default", original_split=split, source_group_id=f"storyscope_prompt:{r.prompt_id}",
                      leakage_keys=[f"storyscope_prompt:{r.prompt_id}"], independence_key=f"storyscope_prompt:{r.prompt_id}",
                      domain="creative", subdomain="literary_short_story",
                      original_labels={"prompt_id": r.prompt_id, "title": r.title, "model_column": col,
                                       "human_author_of_source_story": r.human_author, "human_anthology": r.human_anthology},
                      parent_ids=[f"storyscope:{r.prompt_id}:{col}"],
                      parent_excerpt={"parent_id": f"storyscope:{r.prompt_id}:{col}", "start": a, "end": b, "parent_sha256": sha256_text(story),
                                      "policy": "paragraph-aligned contiguous 150-450 words, hash-chosen start, markdown heading lines skipped"},
                      rights={"licence": "MIT (dataset card)"})
            rows.append(_ai(rec, STORY_MODELS[col], "short_story_from_prompt (prompt reverse-engineered from a published story)"))
    inv.notes.append("Human source stories are not released (copyright); prompts are LLM reverse-engineered from them.")
    return rows, inv


# ==================================================================== HAP-E-2
HAP_MODELS = ["gpt-5-mini-2025-08-07", "claude-haiku-4-5-20251001", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
              "Meta-Llama-3-70B-Instruct", "Meta-Llama-3-8B-Instruct", "gemma-2-27b-it", "gemma-2-9b-it"]


def normalise_hap_e2(src):
    inv = Inventory("hap_e2")
    rows = []
    base = RAW / "hap_e2" / "files" / "text_data"
    human = pd.read_parquet(base / "hape2-text_chunk_2.parquet")
    inv.rows_by_file["chunk_2"] = len(human)
    for r in human[human["doc_id"].str.startswith("fic_")].itertuples(index=False):
        doc = r.doc_id.split("@")[0]
        if not isinstance(r.text, str) or not r.text.strip():
            continue
        rec = Rec(dataset_key="hap_e2", dataset_id=src["repo"], revision=src["revision"], original_row_id=r.doc_id, text=r.text,
                  config="fic", original_split="all", source_group_id=f"hap_doc:{doc}", leakage_keys=[f"hap_doc:{doc}"],
                  independence_key=f"hap_doc:{doc}", domain="creative", subdomain="fiction_continuation",
                  original_labels={"doc_id": r.doc_id, "role": "human chunk_2 (true continuation)"}, rights={"licence": "MIT; underlying Project Gutenberg public-domain fiction"})
        rows.append(_human(rec, "benchmark_provided_source_document", "benchmark_provided",
                           {"evidence": "true continuation of a Project Gutenberg public-domain novel/short story (HAP-E-2 chunk_2)",
                            "limitations": "book identifiers not exposed; may overlap PG-19 books (near-duplicate linking applies)"}))
    for m in HAP_MODELS:
        df = pd.read_parquet(base / f"hape2-text_{m}.parquet")
        inv.rows_by_file[m] = len(df)
        fic = df[df["doc_id"].str.startswith("fic_")]
        inv.count("fiction_rows", m, len(fic))
        for r in fic.itertuples(index=False):
            if not isinstance(r.text, str) or not r.text.strip():
                inv.count("null_text", m)
                continue
            doc = r.doc_id.split("@")[0]
            rec = Rec(dataset_key="hap_e2", dataset_id=src["repo"], revision=src["revision"], original_row_id=r.doc_id, text=r.text,
                      config="fic", original_split="all", source_group_id=f"hap_doc:{doc}", leakage_keys=[f"hap_doc:{doc}"],
                      independence_key=f"hap_doc:{doc}", domain="creative", subdomain="fiction_continuation",
                      original_labels={"doc_id": r.doc_id, "model": m}, rights={"licence": "MIT; generator terms apply"})
            rows.append(_ai(rec, m, "continue_human_fiction_in_style (~500 words)"))
    inv.notes.append("Base (non-instruct) model files not ingested; chunk_1 (the prompt seed) not used as an evaluation text.")
    return rows, inv


# ======================================================================= LAMP
def normalise_lamp(src):
    inv = Inventory("lamp")
    rows = []
    data = json.loads((RAW / "lamp" / "files" / "Writing_Alignment" / "LAMP" / "LAMP-train-val-test.json").read_text())
    inv.rows_by_file["LAMP-train-val-test.json"] = len(data)
    gen = {"gpt4o": "gpt-4o", "claude3.5-sonnet": "claude-3.5-sonnet", "llama370B": "llama-3-70b", "llama3.1_405B": "llama-3.1-405b"}
    for i, d in enumerate(data):
        inv.count("type", d["type"])
        inv.count("source", d["source"])
        inv.count("editor", d["editor"])
        creative = d["type"] in ("Literary Fiction", "Creative NonFiction", "fantasy")
        dom = "creative" if creative else "nonfiction_personal"
        g = gen.get(d["source"], d["source"])
        common = dict(dataset_key="lamp", dataset_id="salesforce/creativity_eval:Writing_Alignment/LAMP", revision=src["commit"],
                      original_row_id=str(i), config="LAMP-train-val-test", original_split=d.get("data-split"),
                      source_group_id=f"lamp_para:{i}", leakage_keys=[f"lamp_para:{i}"], independence_key=f"lamp_editor:{d['editor']}",
                      domain=dom, subdomain=d["type"], rights={"licence": "BSD-3-Clause (repo); paper CC BY 4.0"})
        labels = {"id": d["id"], "type": d["type"], "source": d["source"], "editor": d["editor"], "instruction": d["instruction"]}
        a = Rec(**common, text=d["preedit"], variant="preedit", original_labels={**labels, "field": "preedit"})
        rows.append(_ai(a, g, "write paragraph from instruction (instruction reverse-engineered from published writing)"))
        e = Rec(**common, text=d["postedit"], variant="postedit", original_labels={**labels, "field": "postedit"})
        e.authorship, e.ai_involvement = "mixed", "yes"
        e.authorship_basis = "documented_process(MFA writer edited the LLM paragraph)"
        e.mix_mechanism = "human_edited_ai"
        e.generator = g
        e.ai_char_spans, e.annotation_completeness, e.annotation_provenance = None, "process_only", "process_label_only"
        e.extensions["pair_id"] = f"lamp:{i}"
        a.extensions["pair_id"] = f"lamp:{i}"
        rows.append(e)
    inv.notes.append("Edited paragraphs carry process labels only; fine-grained edit text pairs are not used as span gold.")
    return rows, inv


# ============================================================== CNN/DailyMail
DM_BOILER = re.compile(r"^(By \. |PUBLISHED: |Last updated at )|\| UPDATED: ")


def normalise_cnn_dailymail(src):
    inv = Inventory("cnn_dailymail")
    rows = []
    for split in ("train", "validation", "test"):
        files = sorted(glob.glob(str(RAW / "cnn_dailymail" / "files" / "3.0.0" / f"{split}-*.parquet")))
        df = pd.concat([pd.read_parquet(f, columns=["article", "id"]) for f in files], ignore_index=True)
        inv.rows_by_file[f"3.0.0/{split}"] = len(df)
        ranked = sorted(((hash_rank("cnn", r), i) for i, r in enumerate(df["id"])))[:6000]
        for _, i in ranked:
            t, aid = df.iloc[i]["article"], df.iloc[i]["id"]
            outlet = "cnn" if t.lstrip().startswith(("(CNN)", "(CNN) --", "(CNN) -")) or t[:120].find("(CNN)") >= 0 else "dailymail_or_unknown"
            rec = Rec(dataset_key="cnn_dailymail", dataset_id=src["repo"], revision=src["revision"], original_row_id=aid, text=t,
                      config="3.0.0", original_split=split, source_group_id=f"cnndm:{aid}", leakage_keys=[f"cnndm:{aid}"],
                      independence_key=f"cnndm:{aid}", domain="news", subdomain=f"{outlet}_article",
                      original_labels={"id": aid, "outlet_heuristic": outlet}, rights={"licence": "Apache-2.0 dataset packaging; news content rights not cleared (research use)"})
            _human(rec, "historical_pre_llm_corpus", "independently_audited",
                   {"evidence": "CNN/Daily Mail articles 2007-2015 (Hermann et al. 2015; See et al. 2017)"},
                   {"corpus_release": "2015-2017; articles 2007-2015"})
            if DM_BOILER.search(t[:400]):
                rec.eligibility, rec.eligibility_reasons = "excluded", ["byline_or_timestamp_boilerplate_lead"]
            inv.count("outlet_heuristic", outlet)
            rows.append(rec)
    inv.facts["candidate_policy"] = "hash-ranked 6000 per split; DailyMail byline/timestamp boilerplate leads excluded"
    return rows, inv


# ===================================================== generate-heterogeneous
HET_DOMAIN = {"gutenberg_selected": ("creative", "historical_fiction"), "standardebooks": ("creative", "historical_fiction"),
              "jmlr_pre2015": ("research", "ml_paper_body"), "beigebook": ("expository", "economic_report"),
              "wikitext": ("expository", "wikipedia_article_body")}


def _gutenberg_key(ref: str | None) -> str | None:
    m = re.search(r"gutenberg\.org/(?:ebooks|files|cache/epub)/(\d+)", ref or "")
    return f"pg19_book:{m.group(1)}" if m else None


def normalise_generate_heterogeneous(src):
    inv = Inventory("generate_heterogeneous")
    rows = []
    base = RAW / "generate_heterogeneous" / "files" / "data"
    for cfg in ("mixed", "human_controls"):
        df = pd.read_parquet(base / cfg / "test.parquet")
        inv.rows_by_file[f"{cfg}/test.parquet"] = len(df)
        inv.schemas[cfg] = list(df.columns)
        for c in ("source_dataset", "reported_model", "cohort"):
            for v, n in df[c].astype(str).value_counts().items():
                inv.count(f"{cfg}:{c}", v, int(n))
        for r in df.itertuples(index=False):
            dom, sub = HET_DOMAIN.get(r.source_dataset, (None, r.source_dataset))
            keys = [f"het_source:{r.source_id}", f"het_group:{r.group_id}"]
            gk = _gutenberg_key(r.source_reference)
            if gk:
                keys.append(gk)
            rec = Rec(dataset_key="generate_heterogeneous", dataset_id="open-text-detector/heterogeneous-ai-spans (local release v1.3.0)",
                      revision="v1.3.0 SHA256SUMS", original_row_id=r.id, text=r.text, config=cfg, original_split="test",
                      source_group_id=f"het_source:{r.source_id}", leakage_keys=keys, independence_key=f"het_group:{r.group_id}",
                      domain=dom, subdomain=sub, source_urls=[r.source_reference] if r.source_reference else [],
                      original_labels={"id": r.id, "kind": r.kind, "cohort": r.cohort, "source_dataset": r.source_dataset,
                                       "source_title": r.source_title, "source_author": r.source_author,
                                       "model_identity_status": r.model_identity_status, "human_verified_flag": bool(r.human_verified)},
                      rights={"licence": f"mixed-source terms (release LICENSE.md); source: {r.source_license}"},
                      extensions={"user_training_note": "same release's train split is user training data; test split is group-isolated"})
            if r.kind == "mixed":
                iv = [[int(s["start"]), int(s["end"])] for s in r.spans if s["label"] == "ai"]
                rec.authorship, rec.ai_involvement, rec.authorship_basis = "mixed", "yes", "construction_provenance(splice)"
                rec.mix_mechanism = "block_replacement"
                rep_m = r.reported_model if isinstance(r.reported_model, str) and r.reported_model else None
                req_m = r.requested_model if isinstance(r.requested_model, str) and r.requested_model else None
                rec.generator = rep_m or (f"{req_m} (requested; identity unconfirmed)" if req_m else None)
                rec.extensions["generator_identity_status"] = r.model_identity_status
                rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = iv, "complete", "insertion_provenance"
                if abs(S.total_length([tuple(x) for x in iv]) / max(len(r.text), 1) - float(r.ai_fraction_chars)) > 1e-6:
                    rec.eligibility, rec.eligibility_reasons = "quarantined", ["ai_fraction_mismatch_with_release"]
            else:
                _human(rec, "benchmark_provided_source_document", "benchmark_provided",
                       {"evidence": f"original {r.source_dataset} passage used as the matched control",
                        "limitations": "release flag human_verified=False; sources are pre-LLM publications"})
            rows.append(rec)
    return rows, inv


# ============================================================ detokenised derivatives
REFLIST = re.compile(r"(et al\b.*){3,}|(\b(19|20)\d\d\b.*){8,}", re.S)


def _derive_detok(key: str, kind: str, keep) -> tuple[list[dict], Inventory]:
    inv = Inventory(f"{key}_detok")
    df = pd.read_parquet(NORM / f"{key}.parquet")
    df = df[keep(df)]
    out = []
    flags = collections.Counter()
    for r in df.to_dict("records"):
        text, info = detok_checked(r["text"], kind)
        r = dict(r)
        r["dataset_key"] = f"{key}_detok"
        r["example_id"] = r["example_id"].replace(f"{key}:", f"{key}_detok:", 1)
        canon = json.loads(r["canonicalisation"]) if r["canonicalisation"] else {"transformations": []}
        canon["transformations"] = canon.get("transformations", []) + [f"lossless detokenisation ({info['version']}; {info['n_edits']} recorded edits)"]
        canon["detok"] = {k: info[k] for k in ("version", "n_edits", "flags", "lossless")}
        canon["detok_edits_vs_previous_text"] = info["edits"]
        r["canonicalisation"] = json.dumps(canon)
        r["original_text_sha256"] = r["original_text_sha256"]
        r["text"] = text
        r["text_sha256"] = sha256_text(text)
        r["word_count"] = S.word_count(text)
        r["char_count"] = len(text)
        af = json.loads(r["artefact_flags"]) if r["artefact_flags"] else []
        af = [a for a in af if a not in ("wikitext_tokenised_punctuation", "tokenised_lowercased_extraction")]
        af += ["detokenised"] + (["lowercased_source"] if kind == "sci" else []) + info["flags"]
        r["artefact_flags"] = json.dumps(af)
        if kind == "sci" and REFLIST.search(text):
            r["eligibility"] = "excluded"
            r["eligibility_reasons"] = json.dumps(["reference_list_like_passage"])
            flags["reference_list_excluded"] += 1
        for f in info["flags"]:
            flags[f] += 1
        out.append(r)
    inv.rows_by_file[f"{key}.parquet (eligible)"] = len(out)
    inv.counters["flags"] = flags
    return out, inv


def normalise_wikitext_detok(src):
    return _derive_detok("wikitext", "wikitext", lambda d: d["eligibility"] == "eligible")


def normalise_scientific_papers_detok(src):
    return _derive_detok("scientific_papers", "sci", lambda d: (d["eligibility"] == "eligible") & (d["config"] == "pubmed"))
