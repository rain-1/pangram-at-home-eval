"""Normalisation adapters for Hugging Face-hosted catalogue sources.

Each adapter streams the *full* downloaded release to measure its inventory, and emits
canonical candidate records. For very large releases the candidate pool is a deterministic
population-wide hash-ranked sample per stratum (never "the first N rows").
"""

from __future__ import annotations

import collections
import glob
import html
import json
import re
from pathlib import Path

import pandas as pd

from aidet_eval import spans as S
from aidet_eval.adapters.common import Inventory, Rec, contiguous_excerpt, hash_rank, paragraph_spans
from aidet_eval.overlap import fingerprint as overlap_fp
from aidet_eval.util import sha256_text

# --- domain mappings (evidence: inspected samples; see data/inventory/source_audit.json) -------------
LLMTRACE_DOMAIN = {
    "paper_abstract": ("research", "abstract"),
    "news": ("news", "news_article"),
    "review": ("reviews", "product_review"),
    "factual": ("expository", "wikipedia_paragraph"),
    "story": ("narrative_summary", "movie_plot_summary"),  # human sources are film plot summaries, not fiction
    "article": ("web_article", "blog_article"),  # not mapped to research without evidence
    "question": ("qa_forum", "answer"),
    "short_form": ("short_form", "tweet"),
    "poetry": ("poetry", "poetry_labelled"),
}
RAID_DOMAIN = {
    "abstracts": ("research", "abstract"),
    "news": ("news", "news_article"),
    "reviews": ("reviews", "movie_review"),
    "wiki": ("expository", "wikipedia_article_lead"),
    "books": ("narrative_summary", "book_plot_summary"),
    "poetry": ("poetry", "poem"),
    "recipes": ("recipes", "recipe"),
    "reddit": ("forum", "reddit_post"),
}
CORE_DOMAINS = ["research", "news", "reviews", "expository", "creative"]


def _files(key: str) -> Path:
    return Path("data/raw") / key / "files"


# ============================================================================ LLMTrace

def _llmtrace_rows(key: str, dataset_id: str, revision: str, inv: Inventory, keep, detection: bool):
    out = []
    for split in ("train", "valid", "test"):
        path = _files(key) / f"{split}.jsonl"
        n = 0
        for chunk in pd.read_json(path, lines=True, chunksize=50000, dtype=False):
            n += len(chunk)
            if not inv.schemas:
                inv.schemas[f"{split}.jsonl"] = {c: str(t) for c, t in chunk.dtypes.items()}
            for c in ("lang", "label", "data_type", "prompt_type", "model"):
                for v, cnt in chunk[c].astype(str).value_counts().items():
                    inv.count(c, v, int(cnt))
            for (lang, label, pt), cnt in chunk.groupby(["lang", "label", chunk["prompt_type"].astype(str)]).size().items():
                inv.count("lang|label|prompt_type", f"{lang}|{label}|{pt}", int(cnt))
            eng = chunk[chunk["lang"] == "eng"]
            for idx, r in eng.iterrows():
                rid = f"{split}:{idx}"
                if not keep(r, rid):
                    continue
                out.append(_llmtrace_record(key, dataset_id, revision, split, idx, r, detection))
        inv.rows_by_file[f"{split}.jsonl"] = n
    return out


def _llmtrace_record(key, dataset_id, revision, split, idx, r, detection):
    dom, sub = LLMTRACE_DOMAIN.get(r["data_type"], (None, r["data_type"]))
    text = r["text"]
    label = r["label"]
    pt = None if str(r.get("prompt_type")) in ("nan", "None") else r.get("prompt_type")
    rec = Rec(dataset_key=key, dataset_id=dataset_id, revision=revision, original_row_id=str(idx), text=text,
              config="eng", original_split=split, source_group_id=f"llmtrace_topic:{r['topic_id']}",
              leakage_keys=[f"llmtrace_topic:{r['topic_id']}"], independence_key=f"llmtrace_topic:{r['topic_id']}",
              domain=dom, subdomain=sub, language="en",
              original_labels={"label": label, "model": r["model"], "data_type": r["data_type"], "prompt_type": pt,
                               "topic_id": r["topic_id"],
                               **({"ai_char_intervals": r["ai_char_intervals"]} if detection else {})},
              rights={"licence": "Apache-2.0 (dataset card)", "underlying_sources": "web-derived; not individually cleared"})
    if label == "human":
        rec.authorship, rec.ai_involvement = "human", "no"
        rec.authorship_basis = "dataset_label"
        rec.ai_char_spans = []
        rec.annotation_completeness = "complete"
        rec.annotation_provenance = "dataset_label_assertion"
        rec.human_provenance_category = "benchmark_provided_web_source"
        rec.provenance_tier = "benchmark_provided"
        rec.human_provenance_evidence = {"evidence": "LLMTrace label=human; web-derived source text, dates not audited"}
    elif label == "ai":
        rec.authorship, rec.ai_involvement = "ai", "yes"
        rec.authorship_basis = "dataset_label"
        rec.generator = r["model"]
        rec.ai_char_spans = [[0, len(text)]]
        rec.annotation_completeness = "complete"
        rec.annotation_provenance = "dataset_label_assertion"
        rec.extensions["generation_task"] = pt
    elif label == "mixed":
        rec.authorship, rec.ai_involvement = "mixed", "yes"
        rec.generator = r["model"]
        rec.extensions["generation_task"] = pt
        try:
            iv = json.loads(r["ai_char_intervals"]) if isinstance(r["ai_char_intervals"], str) else r["ai_char_intervals"]
        except Exception:  # noqa: BLE001
            iv = None
        rec.ai_char_spans = iv
        if pt == "fill_gaps":
            rec.mix_mechanism = "sentence_replacement"
            rec.authorship_basis = "documented_generation_process"
            rec.annotation_completeness = "complete"
            rec.annotation_provenance = "insertion_provenance"
            rec.extensions["mechanism_evidence"] = "prompt asks the model to fill <SENTENCE> gaps in a human text"
        elif pt == "human_modification":
            rec.mix_mechanism = "human_edited_ai"
            rec.authorship_basis = "documented_generation_process"
            rec.annotation_completeness = "partial"
            rec.annotation_provenance = "dataset_intervals_method_unverified"
        else:
            rec.mix_mechanism = f"llmtrace_{pt}_mixed"
            rec.authorship_basis = "dataset_label"
            rec.annotation_completeness = "partial"
            rec.annotation_provenance = "dataset_intervals_method_unverified"
    return rec


def normalise_llmtrace_detection(src):
    inv = Inventory(src["key"])
    rows = _llmtrace_rows(src["key"], src["repo"], src["revision"], inv, lambda r, rid: True, detection=True)
    inv.facts["span_encoding"] = "ai_char_intervals: JSON list of [start,end) code-point offsets (verified on non-ASCII text; no non-BMP chars present so UTF-16 vs code point not distinguishable on this release)"
    inv.facts["fill_gaps_generators"] = "gemini-2.5-flash only (measured)"
    return rows, inv


def normalise_llmtrace_classification(src):
    inv = Inventory(src["key"])
    per_cell = collections.Counter()
    ranked: dict[tuple, list] = collections.defaultdict(list)

    # stream everything for inventory; keep hash-ranked English candidates per (label,data_type,model,prompt_type)
    def keep(r, rid):
        cell = (r["label"], r["data_type"], r["model"], str(r.get("prompt_type")))
        ranked[cell].append((hash_rank("llmtrace_classification", rid), rid))
        return True

    rows = _llmtrace_rows(src["key"], src["repo"], src["revision"], inv, keep, detection=False)
    cap = {"human": 400, "ai": 60}
    allowed = set()
    for cell, lst in ranked.items():
        lst.sort()
        allowed.update(rid for _, rid in lst[:cap.get(cell[0], 60)])
    rows = [r for r in rows if f"{r.original_split}:{r.original_row_id}" in allowed]
    inv.facts["candidate_policy"] = "English rows; hash-ranked (seed 42) cap 400 human / 60 ai per (label,data_type,model,prompt_type) cell"
    return rows, inv


# ================================================================================ RAID

def normalise_raid(src):
    import pyarrow as pa
    import pyarrow.csv as pacsv

    inv = Inventory(src["key"])
    rows = []
    cols = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain", "title", "prompt", "generation"]
    for fname in ("train.csv", "extra.csv"):
        path = _files("raid") / fname
        opts = pacsv.ConvertOptions(column_types={c: pa.string() for c in cols}, strings_can_be_null=False)
        reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=1 << 26),
                                 parse_options=pacsv.ParseOptions(newlines_in_values=True), convert_options=opts)
        n = 0
        best: dict[tuple, list] = collections.defaultdict(list)
        human_ids: dict[str, tuple] = {}
        stash: dict[str, dict] = {}
        for batch in reader:
            df = batch.to_pandas()
            n += len(df)
            if fname not in inv.schemas:
                inv.schemas[fname] = list(df.columns)
            for c in ("model", "domain", "attack", "decoding", "repetition_penalty"):
                for v, cnt in df[c].value_counts().items():
                    inv.count(f"{fname}:{c}", v, int(cnt))
            clean = df[(df["attack"] == "none") & df["domain"].isin(RAID_DOMAIN)]
            for r in clean.itertuples(index=False):
                cell = (r.domain, r.model, r.decoding, r.repetition_penalty)
                key = hash_rank("raid", r.id)
                lst = best[cell]
                cap = 2500 if r.model == "human" else 25
                if len(lst) < cap or key < lst[-1][0]:
                    lst.append((key, r.id))
                    lst.sort()
                    if len(lst) > cap:
                        drop = lst.pop()[1]
                        stash.pop(drop, None)
                    stash[r.id] = r._asdict()
        inv.rows_by_file[fname] = n
        for cell, lst in best.items():
            for _, rid in lst:
                r = stash[rid]
                rows.append(_raid_record(src, fname, r))
    # test.csv: unlabelled public test rows -> inventory only
    tn = 0
    for batch in pacsv.open_csv(_files("raid") / "test.csv", read_options=pacsv.ReadOptions(block_size=1 << 26),
                                parse_options=pacsv.ParseOptions(newlines_in_values=True)):
        tn += batch.num_rows
    inv.rows_by_file["test.csv"] = tn
    inv.notes.append("test.csv has only id,generation (labels withheld upstream); counted, never scored.")
    inv.notes.append("Human label rule verified: model=='human' rows have empty decoding/prompt and source_id==id.")
    inv.facts["candidate_policy"] = "attack=='none' English prose domains; hash-ranked cap 2500 human / 25 per (domain,model,decoding,repetition_penalty) AI cell"
    inv.facts["label_status"] = "labelled training release (train.csv/extra.csv): NOT unseen evaluation for RAID-trained detectors"
    return rows, inv


def _raid_record(src, fname, r):
    dom, sub = RAID_DOMAIN[r["domain"]]
    text = r["generation"]
    split = fname.replace(".csv", "")
    rec = Rec(dataset_key="raid", dataset_id=src["repo"], revision=src["revision"], original_row_id=r["id"],
              text=text, config="raid", original_split=split, source_group_id=f"raid_source:{r['source_id']}",
              leakage_keys=[f"raid_source:{r['source_id']}"], independence_key=f"raid_source:{r['source_id']}",
              domain=dom, subdomain=sub,
              original_labels={k: r[k] for k in ("model", "domain", "attack", "decoding", "repetition_penalty", "source_id", "adv_source_id")},
              rights={"licence": "MIT (dataset card)", "underlying_sources": "domain-specific public corpora"},
              extensions={"title": r["title"], "release_role": "labelled_training_release"})
    if raid_authorship(r["model"]) == "human":
        rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "dataset_label(model=='human')"
        rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "dataset_label_assertion"
        rec.human_provenance_category = "benchmark_provided_source_document"
        rec.provenance_tier = "benchmark_provided"
        rec.human_provenance_evidence = {"evidence": "RAID human source document; collection dates not independently audited"}
    else:
        rec.authorship, rec.ai_involvement, rec.authorship_basis = "ai", "yes", "dataset_label(model!='human')"
        rec.generator = r["model"]
        rec.generation_parameters = {"decoding": r["decoding"], "repetition_penalty": r["repetition_penalty"]}
        rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [[0, len(text)]], "complete", "dataset_label_assertion"
        rec.extensions["generation_task"] = "continuation_from_title_prompt"
        rec.extensions["prompt"] = r["prompt"]
    return rec


# ============================================================================= DAMASHA

def normalise_damasha(src):
    inv = Inventory(src["key"])
    rows = []
    f = _files("damasha") / "DAMASHA_Final_No_ADV.csv"
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    inv.rows_by_file[f.name] = len(df)
    inv.schemas[f.name] = list(df.columns)
    positions = collections.Counter()
    for idx, r in df.iterrows():
        tagged = r["hybrid_text"]
        rec = Rec(dataset_key="damasha", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(idx),
                  text="", config="clean", original_split="all", domain=None, subdomain="unknown",
                  original_labels={"has_pair": r["has_pair "]},
                  rights={"licence": "CC-BY-4.0 (dataset card)", "underlying_sources": "reuses MAGE/M4/XSum/essay corpora per card"})
        try:
            text, iv = S.strip_tags_with_offsets(tagged, "<AI_Start>", "</AI_End>")
            rec.text = text
            rec.ai_char_spans = [list(x) for x in iv]
            rec.annotation_completeness = "complete"
            rec.annotation_provenance = "dataset_inline_tags"
            rec.canonicalisation = {"transformations": ["remove <AI_Start>/</AI_End> tags"],
                                    "offset_mapping": "tag removal; spans computed on tag-free text (exact)"}
            rec.original_text_sha256 = sha256_text(tagged)
            pos = S.block_position(iv, len(text)) if iv else "none"
            positions[pos] += 1
        except S.SpanError as e:
            rec.text = tagged
            rec.eligibility, rec.eligibility_reasons = "quarantined", [f"tag_parse:{e}"]
        rec.authorship, rec.ai_involvement = "mixed", "yes"
        rec.authorship_basis = "dataset_inline_tags"
        rec.mix_mechanism = "unknown_tagged_ai_segments"
        rec.generator = None
        rec.leakage_keys = []
        rec.independence_key = f"damasha:{idx}"
        if rec.eligibility == "eligible":
            rec.eligibility_reasons = ["mechanism_domain_generator_not_in_release"]
        rows.append(rec)
    # attacked files: inventory only (reserve robustness material)
    for p in sorted(glob.glob(str(_files("damasha") / "attacked*.csv"))):
        a = pd.read_csv(p, dtype=str, keep_default_na=False, usecols=["attack_name", "tag_count"])
        inv.rows_by_file[Path(p).name] = len(a)
        inv.schemas[Path(p).name] = ["hybrid_text", "attacked_text", "attack_name", "tag_count", "has_pair "]
        for v, c in a["attack_name"].value_counts().items():
            inv.count("attack_name", v, int(c))
    inv.counters["structural_ai_block_position(not_mechanism_evidence)"] = positions
    inv.notes.append("Release rows carry no domain, generator or mechanism fields; the paper's sentence-replacement subset cannot be identified row-wise.")
    inv.notes.append("Measured clean rows 96,692 vs paper clean_combined 95,859.")
    return rows, inv


# ============================================================================== AITDNA

def normalise_aitdna(src):
    inv = Inventory(src["key"])
    rows = []
    views = {}
    for v in ("original", "document", "span", "token", "membership", "boundary", "intent", "content", "sentence"):
        p = glob.glob(str(_files("aitdna") / v / "*.parquet"))[0]
        views[v] = pd.read_parquet(p)
        inv.rows_by_file[f"{v}/test.parquet"] = len(views[v])
        inv.schemas[v] = list(views[v].columns)
    span = views["span"]
    doc = views["document"]
    task_dom = {"Creative Writing": "creative", "Explanatory Writing": "expository", "Argumentative Writing": "argumentative",
                "Peer Review": "research_peer_review"}
    for i in range(len(span)):
        md = dict(span.iloc[i]["metadata"])
        segs = list(span.iloc[i]["data"])
        text = "".join(s["text"] for s in segs)
        doc_text = "".join(s["text"] for s in doc.iloc[i]["data"])
        doc_label = [s["author"] for s in doc.iloc[i]["data"]]
        pos, ai, hu = 0, [], []
        for s in segs:
            a, b = pos, pos + len(s["text"])
            if b > a:
                (ai if s["author"] == "Bot" else hu).append((a, b))
            pos = b
        human_only = bool(md["human_only"])
        rec = Rec(dataset_key="aitdna", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(i), text=text,
                  config="span", original_split="test", source_group_id=f"aitdna_session:{i}",
                  leakage_keys=[f"aitdna_session:{i}"], independence_key=f"aitdna_author:{md['author']}",
                  domain=task_dom.get(md["task"], md["task"]), subdomain=md["task"],
                  original_labels={**{k: (v if not hasattr(v, "item") else v.item()) for k, v in md.items()},
                                   "document_view_label": doc_label},
                  rights={"licence": "CC-BY-SA-4.0 (dataset card; catalogue listed GPL-2.0)"},
                  extensions={"setting": md["setting"], "temperature": md["temperature"], "writer": md["author"]})
        rec.ai_char_spans = [list(x) for x in S.union(ai)]
        rec.human_char_spans = [list(x) for x in S.union(hu)]
        rec.annotation_completeness = "complete"
        rec.annotation_provenance = "edit_log_lineage"
        rec.human_provenance_category = "logged_writing_session"
        rec.provenance_tier = "logged_session"
        rec.human_provenance_evidence = {"evidence": "keystroke/edit log of the writing session (AITDNA span view)",
                                         "limitations": "provenance is per logged interface; external copy-paste origin not observable"}
        if doc_text != text:
            rec.eligibility, rec.eligibility_reasons = "quarantined", ["span_view_does_not_match_document_view"]
        if human_only:
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "logged_session(human_only=True)"
        else:
            rec.authorship, rec.ai_involvement = "mixed", "yes"
            rec.authorship_basis = "logged_session(AI assistant used)"
            rec.mix_mechanism = "interactive_cowriting"
            rec.generator = md["model"] or None
            if not ai:
                rec.extensions["no_surviving_ai_characters"] = True
        rows.append(rec)
    md = pd.DataFrame([dict(m) for m in span["metadata"]])
    for c in ("human_only", "task", "setting", "model"):
        for v, n in md[c].astype(str).value_counts().items():
            inv.count(c, v, int(n))
    inv.facts["distinct_texts"] = len(span)
    inv.facts["distinct_authors"] = int(md["author"].nunique())
    inv.facts["views_are_same_texts"] = "all 9 views have 362 rows with identical metadata; span concat == document text for all rows"
    inv.notes.append("Document view uses a 50%-AI threshold; we use exact span-view provenance and process-level ai_involvement.")
    return rows, inv


# =============================================================================== Beemo

def normalise_beemo(src):
    inv = Inventory(src["key"])
    df = pd.read_parquet(glob.glob(str(_files("beemo") / "data" / "*.parquet"))[0])
    inv.rows_by_file["data/train.parquet"] = len(df)
    inv.schemas["data/train.parquet"] = list(df.columns)
    for c in ("category", "model"):
        for v, n in df[c].value_counts().items():
            inv.count(c, v, int(n))
    inv.facts["unique_prompt_id"] = int(df["prompt_id"].nunique())
    inv.facts["unique_human_output"] = int(df["human_output"].nunique())
    rows = []
    for _, r in df.iterrows():
        pid = r["prompt_id"]
        base = dict(dataset_key="beemo", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(r["id"]),
                    config="default", original_split="train", source_group_id=f"norobots_prompt:{pid}",
                    leakage_keys=[f"norobots_prompt:{pid}"], independence_key=f"norobots_prompt:{pid}",
                    domain="instruction", subdomain=r["category"],
                    rights={"licence": "field-dependent: human_output CC-BY-NC-4.0 (No Robots); human_edits MIT subject to model terms; model_output subject to generator terms"})
        common_labels = {"id": int(r["id"]), "category": r["category"], "model": r["model"], "prompt_id": pid}
        h = Rec(**base, text=r["human_output"], variant="human_output",
                original_labels={**common_labels, "field": "human_output"})
        h.authorship, h.ai_involvement, h.authorship_basis = "human", "no", "dataset_field(human_output from No Robots)"
        h.ai_char_spans, h.annotation_completeness, h.annotation_provenance = [], "complete", "dataset_field_assertion"
        h.human_provenance_category = "curator_attested_annotator"
        h.provenance_tier = "curator_attested"
        h.human_provenance_evidence = {"evidence": "No Robots annotator response reused as Beemo human_output"}
        a = Rec(**base, text=r["model_output"], variant="model_output", original_labels={**common_labels, "field": "model_output"})
        a.authorship, a.ai_involvement, a.authorship_basis = "ai", "yes", "dataset_field(model_output)"
        a.generator = r["model"]
        a.ai_char_spans, a.annotation_completeness, a.annotation_provenance = [[0, len(r["model_output"])]], "complete", "dataset_field_assertion"
        a.extensions["generation_task"] = r["category"]
        e = Rec(**base, text=r["human_edits"], variant="human_edits", original_labels={**common_labels, "field": "human_edits"})
        e.authorship, e.ai_involvement = "mixed", "yes"
        e.authorship_basis = "documented_process(expert edited model_output)"
        e.mix_mechanism = "human_edited_ai"
        e.generator = r["model"]
        e.ai_char_spans = None  # no causal surviving-span gold; diffs are not used as gold
        e.annotation_completeness = "process_only"
        e.annotation_provenance = "process_label_only"
        for x in (h, a, e):
            x.extensions["triplet_id"] = f"beemo:{r['id']}"
            rows.append(x)
    inv.notes.append("llama-3.1-70b_edits and gpt-4o_edits (AI edits of AI output) inventoried only; not used.")
    return rows, inv


# ============================================================================ No Robots

def normalise_no_robots(src):
    inv = Inventory(src["key"])
    rows = []
    for split in ("train", "test"):
        p = glob.glob(str(_files("no_robots") / "data" / f"{split}-*.parquet"))[0]
        df = pd.read_parquet(p)
        inv.rows_by_file[f"data/{split}.parquet"] = len(df)
        inv.schemas[f"data/{split}.parquet"] = list(df.columns)
        for v, n in df["category"].value_counts().items():
            inv.count(f"{split}:category", v, int(n))
        for ri, r in df.iterrows():
            msgs = list(r["messages"])
            assistant = [m for m in msgs if m["role"] == "assistant"]
            inv.count("n_assistant_turns", len(assistant))
            if not assistant:
                continue
            text = assistant[0]["content"]  # first assistant turn: one response per prompt
            rec = Rec(dataset_key="no_robots", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(ri),
                      text=text, config="default", original_split=split, source_group_id=f"norobots_prompt:{r['prompt_id']}",
                      leakage_keys=[f"norobots_prompt:{r['prompt_id']}"], independence_key=f"norobots_prompt:{r['prompt_id']}",
                      domain="instruction", subdomain=r["category"],
                      original_labels={"category": r["category"], "prompt_id": r["prompt_id"], "assistant_turn": 0,
                                       "n_assistant_turns": len(assistant)},
                      rights={"licence": "CC-BY-NC-4.0"})
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "curator_attestation"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "curator_attestation"
            rec.human_provenance_category = "curator_attested_annotator"
            rec.provenance_tier = "curator_attested"
            rec.verified_dates = {"collection": "2023 (dataset release; per-item dates not published)"}
            rec.human_provenance_evidence = {"evidence": "HuggingFaceH4 card: skilled annotators, vendor instructed not to use LLMs",
                                             "limitations": "attestation, not logged provenance"}
            if r["category"] == "Coding":
                rec.eligibility, rec.eligibility_reasons = "excluded", ["non_prose_code_category"]
            rows.append(rec)
    inv.notes.append("train_sft/test_sft files are identical-prompt SFT copies (same prompt_ids) and are not separately ingested.")
    return rows, inv


# ================================================================================ IMDb

BR_RE = re.compile(r"<br\s*/?>", re.I)


def normalise_imdb(src):
    inv = Inventory(src["key"])
    rows = []
    for split in ("train", "test", "unsupervised"):
        df = pd.read_parquet(glob.glob(str(_files("imdb") / "plain_text" / f"{split}-*.parquet"))[0])
        inv.rows_by_file[f"plain_text/{split}.parquet"] = len(df)
        inv.schemas[split] = list(df.columns)
        for v, n in df["label"].value_counts().items():
            inv.count(f"{split}:sentiment_label", v, int(n))
        ranked = sorted(((hash_rank("imdb", split, i), i) for i in range(len(df))))[:6000]
        for _, i in ranked:
            orig = df.iloc[i]["text"]
            text = BR_RE.sub("\n", orig)
            rec = Rec(dataset_key="imdb", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(i), text=text,
                      config="plain_text", original_split=split, source_group_id=f"imdb:{split}:{i}",
                      leakage_keys=[], independence_key=f"imdb:{split}:{i}",
                      domain="reviews", subdomain="movie_review",
                      original_labels={"sentiment_label": int(df.iloc[i]["label"]), "note": "sentiment, not authorship"},
                      rights={"licence": "other (IMDb terms; research corpus)"}, original_text_sha256=sha256_text(orig))
            rec.canonicalisation = {"transformations": ["replace <br />/<br/>/<br> tags with \\n"],
                                    "offset_mapping": "not needed (no spans on human-only text)"}
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "historical_release_date"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "pre_llm_release"
            rec.human_provenance_category = "historical_pre_llm_corpus"
            rec.provenance_tier = "independently_audited"
            rec.verified_dates = {"corpus_release": "2011 (Maas et al., ACL 2011)"}
            rec.human_provenance_evidence = {"evidence": "Large Movie Review Dataset released 2011, before modern LLMs",
                                             "limitations": "per-review authorship not verified; movie IDs not in HF release so same-movie reviews cannot be grouped"}
            rows.append(rec)
    inv.facts["candidate_policy"] = "hash-ranked 6000 per split"
    return rows, inv


# ================================================================================ XSum

def normalise_xsum(src):
    inv = Inventory(src["key"])
    rows = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(glob.glob(str(_files("xsum") / "data" / f"{split}-*.parquet"))[0])
        inv.rows_by_file[f"data/{split}.parquet"] = len(df)
        inv.schemas[split] = list(df.columns)
        ranked = sorted(((hash_rank("xsum", r), i) for i, r in enumerate(df["id"])))[:6000]
        for _, i in ranked:
            r = df.iloc[i]
            rec = Rec(dataset_key="xsum", dataset_id=src["repo"], revision=src["revision"], original_row_id=str(r["id"]),
                      text=r["document"], config="default", original_split=split, source_group_id=f"xsum:{r['id']}",
                      leakage_keys=[f"xsum:{r['id']}"], independence_key=f"xsum:{r['id']}",
                      domain="news", subdomain="bbc_article",
                      source_urls=[f"https://www.bbc.com/news/{r['id']}"],
                      original_labels={"id": r["id"]}, rights={"licence": "unknown; BBC content rights not cleared (research use)"})
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "historical_release_date"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "pre_llm_release"
            rec.human_provenance_category = "historical_pre_llm_corpus"
            rec.provenance_tier = "independently_audited"
            rec.verified_dates = {"corpus_release": "2018 (Narayan et al., EMNLP 2018); BBC articles 2010-2017"}
            rec.human_provenance_evidence = {"evidence": "XSum BBC article bodies (document field, not summaries)"}
            rows.append(rec)
    inv.facts["candidate_policy"] = "hash-ranked 6000 per split"
    return rows, inv


# ============================================================================ WikiText

HEADING_RE = re.compile(r"^ (= )+[^=].*?( =)+ $")


def normalise_wikitext(src):
    inv = Inventory(src["key"])
    rows = []
    for split in ("train", "validation", "test"):
        files = sorted(glob.glob(str(_files("wikitext") / "wikitext-103-raw-v1" / f"{split}-*.parquet")))
        lines = pd.concat([pd.read_parquet(f) for f in files])["text"].tolist()
        inv.rows_by_file[f"wikitext-103-raw-v1/{split}"] = len(lines)
        # article boundaries: a top-level heading " = Title = \n" preceded by an empty line
        arts, cur, title = [], [], None
        for k, ln in enumerate(lines):
            is_top = ln.startswith(" = ") and not ln.startswith(" = = ") and ln.rstrip("\n").endswith(" = ")
            if is_top and (k == 0 or lines[k - 1] == ""):
                if cur:
                    arts.append((title, "".join(cur)))
                title, cur = ln.strip().strip("=").strip(), [ln]
            else:
                cur.append(ln)
        if cur:
            arts.append((title, "".join(cur)))
        inv.count("articles_reconstructed", split, len(arts))
        ranked = sorted((hash_rank("wikitext", split, t, j), j) for j, (t, _) in enumerate(arts))[:5000]
        for _, j in ranked:
            t, body = arts[j]
            # paragraphs = non-heading lines
            paras = [(a, b) for a, b in paragraph_spans(body) if not body[a:b].startswith("=")]
            ex = contiguous_excerpt(paras, body, f"wikitext|{split}|{t}", 120, 450)
            if ex is None:
                continue
            a, b = ex
            text = body[a:b]
            rec = Rec(dataset_key="wikitext", dataset_id=src["repo"], revision=src["revision"], original_row_id=f"{j}",
                      text=text, config="wikitext-103-raw-v1", original_split=split,
                      source_group_id=f"wikipedia_title:{t}", leakage_keys=[f"wikipedia_title:{t}"],
                      independence_key=f"wikipedia_title:{t}", domain="expository", subdomain="wikipedia_article_body",
                      original_labels={"article_title": t}, parent_ids=[f"wikitext:{split}:article:{j}"],
                      parent_excerpt={"parent_id": f"wikitext:{split}:article:{j}", "start": a, "end": b,
                                      "parent_sha256": sha256_text(body), "policy": "paragraph-aligned contiguous 120-450 words, hash-chosen start"},
                      rights={"licence": "CC-BY-SA (Wikipedia text via WikiText)"},
                      artefact_flags=["wikitext_tokenised_punctuation", "wikitext_at_markers"] if "@" in text else ["wikitext_tokenised_punctuation"])
            rec.authorship, rec.ai_involvement, rec.authorship_basis = "human", "no", "historical_release_date"
            rec.ai_char_spans, rec.annotation_completeness, rec.annotation_provenance = [], "complete", "pre_llm_release"
            rec.human_provenance_category = "historical_pre_llm_corpus"
            rec.provenance_tier = "independently_audited"
            rec.verified_dates = {"corpus_release": "2016 (Merity et al.); Wikipedia Good/Featured articles"}
            rec.human_provenance_evidence = {"evidence": "WikiText-103 2016 release",
                                             "limitations": "collaborative human authorship; pre-LLM snapshot"}
            rows.append(rec)
    inv.notes.append("Raw lines are pre-tokenised (spaces around punctuation, @-@ markers); kept verbatim and flagged.")
    return rows, inv


# ================================================================================ MAGE

def normalise_mage(src):
    inv = Inventory(src["key"])
    rows = []
    for fname in ("train.csv", "valid.csv", "test.csv", "test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"):
        df = pd.read_csv(_files("mage") / fname, dtype={"text": str, "src": str})
        inv.rows_by_file[fname] = len(df)
        inv.schemas[fname] = list(df.columns)
        for v, n in df["label"].value_counts().items():
            inv.count(f"{fname}:label", v, int(n))
        if "src" in df:
            hum = df["src"].astype(str).str.contains("human")
            inv.count("label_check", f"{fname}: src~human & label==1", int((hum & (df.label == 1)).sum()))
            inv.count("label_check", f"{fname}: src~human & label==0", int((hum & (df.label == 0)).sum()))
            for v, n in df["src"].value_counts().items():
                inv.count("src", v, int(n))
        # reserve: keep only hashes for overlap auditing
        for i, t in enumerate(df["text"].astype(str)):
            rows.append({"dataset_key": "mage", "file": fname, "row": i, "label": int(df["label"].iloc[i]),
                     "authorship": mage_authorship(df["label"].iloc[i]),
                         "src": str(df["src"].iloc[i]) if "src" in df else None, "text_sha256": sha256_text(t),
                         "match_fp": overlap_fp(t)})
    inv.facts["label_mapping_verified"] = "1 = human, 0 = machine (all *_human src rows have label 1)"
    return rows, inv


# ======================================================================= EditLens ICLR

def normalise_editlens_iclr(src):
    inv = Inventory(src["key"])
    rows = []
    for p in sorted(glob.glob(str(_files("editlens_iclr") / "data" / "*.parquet"))):
        name = Path(p).name.split("-")[0]
        df = pd.read_parquet(p)
        inv.rows_by_file[f"data/{name}"] = len(df)
        inv.schemas[name] = list(df.columns)
        for c in ("text_type", "model", "source"):
            for v, n in df[c].astype(str).value_counts().items():
                inv.count(f"{name}:{c}", v, int(n))
        for i, r in df.iterrows():
            tt = r["text_type"]
            rec = Rec(dataset_key="editlens_iclr", dataset_id=src["repo"], revision=src["revision"],
                      original_row_id=str(i), text=r["text"] or "", config="default", original_split=name,
                      source_group_id=f"editlens_source:{r['source_id']}", leakage_keys=[f"editlens_source:{r['source_id']}"],
                      independence_key=f"editlens_source:{r['source_id']}", domain="editlens_in_family", subdomain=str(r["source"]),
                      original_labels={"text_id": r["text_id"], "text_type": tt, "model": r["model"], "source": r["source"],
                                       "cosine_score": r["cosine_score"], "soft_ngrams_score": r["soft_ngrams_score"]},
                      rights={"licence": "CC-BY-NC-SA-4.0 (gated; access accepted previously by account holder)"},
                      extensions={"source_text_sha256": sha256_text(r["source_text"]) if isinstance(r["source_text"], str) else None})
            if tt == "human_written":
                rec.authorship, rec.ai_involvement, rec.ai_char_spans = "human", "no", []
                rec.annotation_completeness = "complete"
                rec.provenance_tier = "in_family_reference"
                rec.human_provenance_category = "benchmark_provided_source_document"
            elif tt == "ai_generated":
                rec.authorship, rec.ai_involvement = "ai", "yes"
                rec.ai_char_spans, rec.annotation_completeness = [[0, len(rec.text)]], "complete"
                rec.generator = r["model"]
            else:
                rec.authorship, rec.ai_involvement = "mixed", "yes"
                rec.mix_mechanism = "ai_edited_human"
                rec.annotation_completeness = "process_only"
                rec.generator = r["model"]
            rec.authorship_basis = f"dataset_text_type={tt}"
            rec.annotation_provenance = "dataset_label_assertion"
            rec.eligibility = "eligible" if name.startswith("test") else "reserve"
            if name == "train" or name == "val":
                rec.eligibility_reasons = ["baseline_training_or_validation_split"]
            rows.append(rec)
    inv.notes.append("In-family: EditLens baselines were trained on this dataset's train split; test results are not independent evidence.")
    return rows, inv


# ---- explicit label conventions (unit-tested) ----------------------------------------
def mage_authorship(label: int) -> str:
    """MAGE convention verified on the release: 1 = human, 0 = machine (NOT the generic 1 = AI)."""
    return {1: "human", 0: "ai"}[int(label)]


def raid_authorship(model: str) -> str:
    """RAID: rows with model == 'human' are human-written; every other model value is AI-generated."""
    return "human" if model == "human" else "ai"
