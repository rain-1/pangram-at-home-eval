"""Select and freeze a source-disjoint benchmark version.

Pipeline (no detector scores are read anywhere in this module):
 1. load normalised candidate pools;
 2. leakage groups = union-find over native keys, exact match-fingerprints and verified MinHash
    near-duplicates; independence groups from writer/book/session/source keys;
 3. training-exposure audit (EditLens train/val + optional user registry);
 4. stratified, hash-ranked (seed 42) panel selection; AI/mixed panels first, then source-matched
    human references, then trusted human evaluation controls, then leakage-disjoint calibration;
 5. write examples.parquet, manifest.json (targets/eligible/selected/shortfalls), overlap evidence,
    benchmark card; hash everything into a benchmark version and mark it frozen.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aidet_eval import overlap, paths
from aidet_eval import spans as S
from aidet_eval.adapters.common import hash_rank
from aidet_eval.util import fingerprint, log_event, nan_to_none, read_json, sha256_text, utcnow, write_json

POOL_SOURCES = ["llmtrace_detection", "llmtrace_classification", "raid", "damasha", "aitdna", "beemo", "imdb", "xsum",
                "scientific_papers", "wikitext", "pg19", "persuade", "no_robots", "editlens_iclr"]
RESERVE_ONLY = {"mage": "reserve: in-distribution regression source; training exposure of future models unknown",
                "coauthor": "reserve: dataset licence not stated; legacy GPT-3; replay not independently validated"}
CORE = ["research", "news", "reviews", "expository", "creative"]
TRUSTED_HUMAN_SOURCE = {"research": "scientific_papers", "news": "xsum", "reviews": "imdb", "expository": "wikitext",
                        "creative": "pg19", "student": "persuade", "instruction": "no_robots"}
FRAC_BINS = [("gt0_le5pct", 0.0, 0.05), ("gt5_le10pct", 0.05, 0.10), ("gt10_le25pct", 0.10, 0.25),
             ("gt25_le50pct", 0.25, 0.50), ("gt50_lt100pct", 0.50, 1.0)]
LENGTH_BINS = [("lt50w", 0, 50), ("50_149w", 50, 150), ("150_299w", 150, 300), ("300_599w", 300, 600),
               ("600_1199w", 600, 1200), ("ge1200w", 1200, 10**9)]


def jl(x):
    """NaN/None-safe json.loads for nullable JSON-string columns."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return json.loads(x)


def frac_bin(f) -> str:
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return "unknown"
    if f >= 1.0:
        return "full_ai"
    if f <= 0.0:
        return "zero"
    for name, lo, hi in FRAC_BINS:
        if lo < f <= hi or (name == "gt50_lt100pct" and lo < f < hi):
            return name
    return "unknown"


def length_bin(wc: int) -> str:
    for name, lo, hi in LENGTH_BINS:
        if lo <= wc < hi:
            return name
    return "unknown"


def block_len_bin(spans_json) -> str | None:
    sp = jl(spans_json)
    if not sp:
        return None
    mean = np.mean([b - a for a, b in sp])
    return "lt100c" if mean < 100 else "100_299c" if mean < 300 else "300_999c" if mean < 1000 else "ge1000c"


def block_count_bin(n) -> str | None:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return None
    n = int(n)
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3_4" if n <= 4 else "ge5"


# ----------------------------------------------------------------------------- loading

V2_EXTRA_POOLS = ["storyscope", "hap_e2", "lamp", "cnn_dailymail", "generate_heterogeneous", "wikitext_detok", "scientific_papers_detok"]


def pool_sources(plan: dict) -> list[str]:
    return POOL_SOURCES + (V2_EXTRA_POOLS if plan.get("v2") else [])


def trusted_sources(plan: dict) -> dict:
    return (plan.get("v2") or {}).get("trusted_human_sources") or TRUSTED_HUMAN_SOURCE


def load_pools(sources: list[str] = POOL_SOURCES) -> pd.DataFrame:
    frames = []
    for k in sources:
        p = paths.NORMALISED / f"{k}.parquet"
        if not p.exists():
            raise SystemExit(f"missing normalised source {k}; run `python -m aidet_eval normalise --sources {k}`")
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    if df["example_id"].duplicated().any():
        d = df[df["example_id"].duplicated()]["example_id"].head().tolist()
        raise SystemExit(f"duplicate example ids: {d}")
    return df


# ----------------------------------------------------------------------------- grouping

def build_groups(df: pd.DataFrame, out_dir: Path, verification: str = "minhash_estimate") -> pd.DataFrame:
    uf = overlap.UnionFind()
    df["match_fp"] = [overlap.fingerprint(t) for t in df["text"]]
    ntok = [len(overlap.match_tokens(t)) for t in df["text"]]
    for eid, keys, fp, fp_ok in zip(df["example_id"], df["leakage_keys"], df["match_fp"],
                                    [n >= overlap.POLICY["min_tokens_for_exact_fingerprint_link"] for n in ntok]):
        uf.find(eid)
        for k in jl(keys) or []:
            uf.union(eid, k)
        if fp_ok:
            uf.union(eid, f"fp:{fp}")  # short texts (< 8 match tokens) are not linked by fingerprint alone
    log_event("build_neardup_start", n=len(df))
    pairs, stats = overlap.near_duplicate_pairs(df["example_id"].tolist(), df["text"].tolist(), verification=verification)
    for a, b, _ in pairs:
        uf.union(a, b)
    df["leakage_group_id"] = [uf.find(e) for e in df["example_id"]]
    # stable, readable group ids
    gid = {g: "lg_" + sha256_text(g)[:16] for g in df["leakage_group_id"].unique()}
    df["leakage_group_id"] = df["leakage_group_id"].map(gid)
    # independence: writer/book-author/session/source key, merged with leakage group
    uf2 = overlap.UnionFind()
    for eid, ik, lg in zip(df["example_id"], df["independence_key"], df["leakage_group_id"]):
        uf2.union(eid, lg)
        if ik:
            uf2.union(eid, f"ind:{ik}")
    df["independence_group_id"] = ["ig_" + sha256_text(uf2.find(e))[:16] for e in df["example_id"]]
    pdf = pd.DataFrame(pairs, columns=["a", "b", "est_jaccard"])
    ds = df.set_index("example_id")["dataset_key"]
    if len(pdf):
        pdf["dataset_a"] = pdf["a"].map(ds)
        pdf["dataset_b"] = pdf["b"].map(ds)
    pdf.to_parquet(out_dir / "overlap" / "near_duplicate_pairs.parquet", index=False)
    fp_dups = df.groupby("match_fp")["dataset_key"].agg(lambda s: sorted(set(s)))
    cross_exact = collections.Counter("|".join(v) for v in fp_dups if len(v) > 1)
    exact_within = int((df.groupby("match_fp").size() > 1).sum())
    cross_near = collections.Counter("|".join(sorted({a, b})) for a, b in zip(pdf.get("dataset_a", []), pdf.get("dataset_b", [])) if a != b)
    stats.update(policy=overlap.POLICY, exact_fingerprint_groups_with_duplicates=exact_within,
                 cross_dataset_exact_links=dict(cross_exact), cross_dataset_near_dup_pairs=dict(cross_near),
                 n_leakage_groups=int(df["leakage_group_id"].nunique()), n_records=int(len(df)))
    write_json(out_dir / "overlap" / "overlap_stats.json", stats)
    log_event("build_groups_done", **{k: v for k, v in stats.items() if not isinstance(v, dict)})
    return df


# ----------------------------------------------------------------------------- exposure

def training_exposure(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per-model exposure status: known_overlap | possible_overlap | no_detected_overlap | unknown."""
    is_train = (df["dataset_key"] == "editlens_iclr") & df["original_split"].isin(["train", "val"])
    train_groups = set(df.loc[is_train, "leakage_group_id"])
    src_fps = set()
    el = pd.read_parquet(paths.NORMALISED / "editlens_iclr.parquet", columns=["original_split", "extensions"])
    for sp, ext in zip(el["original_split"], el["extensions"]):
        if sp in ("train", "val") and ext:
            h = (jl(ext) or {}).get("source_text_sha256")
            if h:
                src_fps.add(h)
    known = df["leakage_group_id"].isin(train_groups) & ~is_train
    known |= df["text_sha256"].isin(src_fps)
    by_ds = df[known].groupby("dataset_key").size().to_dict()
    possible_ds = {d for d, n in by_ds.items() if n > 0 and d != "editlens_iclr"}
    status = np.where(is_train, "known_overlap",
                      np.where(known, "known_overlap",
                               np.where(df["dataset_key"].isin(possible_ds), "possible_overlap", "no_detected_overlap")))
    status = np.where(df["dataset_key"].eq("editlens_iclr") & ~is_train, "possible_overlap", status)
    for m in ("open_pangram_roberta", "open_pangram_llama"):
        df[f"exposure__{m}"] = status
    # optional user registry for future models
    reg_path = paths.CONFIGS / "training_registry.yaml"
    user = {}
    if reg_path.exists():
        reg = yaml.safe_load(reg_path.read_text()) or {}
        for model_id, entries in (reg.get("models") or {}).items():
            hit = pd.Series(False, index=df.index)
            for e in entries or []:
                vals = set(Path(e["path"]).read_text().split())
                col = {"text_sha256": "text_sha256", "source_group_id": "source_group_id", "example_id": "example_id"}[e["kind"]]
                hit |= df[col].isin(vals)
            hit_groups = set(df.loc[hit, "leakage_group_id"])
            df[f"exposure__{model_id}"] = np.where(df["leakage_group_id"].isin(hit_groups), "known_overlap", "no_detected_overlap")
            user[model_id] = int(hit.sum())
    audit = {"pangram_editlens": {"registry": "pangram/editlens_iclr train+val (text, source_text hashes, near-duplicates, shared source ids)",
                                  "known_overlap_by_dataset": {k: int(v) for k, v in by_ds.items()},
                                  "datasets_marked_possible_overlap": sorted(possible_ds) + ["editlens_iclr (test splits: same generation pipeline)"],
                                  "caveat": "no_detected_overlap is not proof of no pretraining exposure (RoBERTa/Llama base pretraining corpora unknown)"},
             "user_registry_hits": user}
    write_json(out_dir / "overlap" / "training_exposure_audit.json", audit)
    return df


# ----------------------------------------------------------------------------- selection

class Selector:
    def __init__(self, df: pd.DataFrame, seed: int):
        self.df = df
        self.seed = seed
        self.group_role: dict[str, str] = {}  # leakage group -> evaluation|calibration
        self.used_ind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.picks: list[dict] = []
        self.cells: list[dict] = []

    def rank(self, ids: pd.Series, salt: str) -> pd.Series:
        return ids.map(lambda e: hash_rank(salt, e, seed=self.seed))

    def take(self, pool: pd.DataFrame, n: int, role: str, panel: str, cell: dict, strata: list[str] | None = None,
             one_per_ind: bool = True, allow_shared_group: bool = False) -> pd.DataFrame:
        """Hash-ranked selection with optional round-robin over strata; group-role exclusivity enforced."""
        eligible = pool[pool["eligibility"] == "eligible"]
        mask = np.array([self.group_role.get(g, role) == role for g in eligible["leakage_group_id"]], dtype=bool)
        ok = eligible.loc[mask]
        if not allow_shared_group:
            taken_groups = {p["leakage_group_id"] for p in self.picks if p["panel"] == panel}
            ok = ok.loc[~ok["leakage_group_id"].isin(taken_groups).to_numpy(dtype=bool)]
        ok = ok.assign(sel_rank=self.rank(ok["example_id"], panel)).sort_values("sel_rank")
        chosen, seen_groups, seen_ids = [], set(), set()
        ind_used = self.used_ind[panel]

        def admissible(r, strict):
            if r.leakage_group_id in seen_groups and (strict or not allow_shared_group):
                return False
            if r.example_id in seen_ids:
                return False
            if strict and one_per_ind and ind_used[r.independence_group_id] > 0:
                return False
            return True

        if strata:
            buckets = collections.OrderedDict()
            for key, grp in ok.groupby(strata, dropna=False, sort=True):
                buckets[key] = list(grp.itertuples(index=False))
            for strict in (True, False):
                progress = True
                while len(chosen) < n and progress:
                    progress = False
                    for key in list(buckets):
                        lst = buckets[key]
                        while lst and not admissible(lst[0], strict):
                            lst.pop(0)  # blocked items stay available to the relaxed pass (buckets are rebuilt)
                        if lst and len(chosen) < n:
                            r = lst.pop(0)
                            chosen.append(r)
                            seen_ids.add(r.example_id)
                            seen_groups.add(r.leakage_group_id)
                            ind_used[r.independence_group_id] += 1
                            progress = True
                if len(chosen) >= n:
                    break
                # rebuild buckets for the relaxed pass
                rest = ok[~ok["example_id"].isin(seen_ids)] if allow_shared_group else ok[~ok["leakage_group_id"].isin(seen_groups)]
                buckets = collections.OrderedDict((k, list(g.itertuples(index=False))) for k, g in rest.groupby(strata, dropna=False, sort=True))
        else:
            for strict in (True, False):
                for r in ok.itertuples(index=False):
                    if len(chosen) >= n:
                        break
                    if admissible(r, strict):
                        chosen.append(r)
                        seen_ids.add(r.example_id)
                        seen_groups.add(r.leakage_group_id)
                        ind_used[r.independence_group_id] += 1
                if len(chosen) >= n:
                    break
        out = pd.DataFrame(chosen, columns=ok.columns) if chosen else ok.head(0)
        for r in out.itertuples(index=False):
            self.group_role[r.leakage_group_id] = role
            self.picks.append({"example_id": r.example_id, "panel": panel, "role": role, "rank": r.sel_rank,
                               "leakage_group_id": r.leakage_group_id, "cell": json.dumps(cell, sort_keys=True)})
        self.cells.append({"panel": panel, "role": role, **{f"cell_{k}": v for k, v in cell.items()},
                           "requested": int(n), "eligible": int(len(eligible)),
                           "eligible_after_group_constraints": int(len(ok)), "selected": int(len(out)),
                           "missing": int(max(0, n - len(out))),
                           "excluded_by_reason": _reasons(pool)})
        return out


def _reasons(pool: pd.DataFrame) -> dict:
    c = collections.Counter()
    for e, rs in zip(pool["eligibility"], pool["eligibility_reasons"]):
        if e != "eligible":
            for r in jl(rs) or [e]:
                c[f"{e}:{r}"] += 1
    return dict(c)


def select(df: pd.DataFrame, plan: dict, profile: str) -> tuple[Selector, dict]:
    seed = int(plan.get("seed", 42))
    v2 = plan.get("v2") or {}
    # poisoned sources: excluded from every panel and from calibration, with an explicit reason
    for key, reason in (plan.get("poisoned_sources") or {}).items():
        ds, _, variant = key.partition(":")
        m = df["dataset_key"].eq(ds) & (df["variant"].eq(variant) if variant else True)
        df.loc[m, "eligibility"] = "excluded"
        df.loc[m, "eligibility_reasons"] = json.dumps([f"poisoned_source: {reason}"])
    trusted = trusted_sources(plan)
    sel = Selector(df, seed)
    scale = 1.0 if profile == "full" else 0.01
    q = lambda n: max(1, int(round(n * scale))) if n else 0  # noqa: E731

    is_eng = df["language"].eq("en")
    d = df[is_eng]
    df["n_blocks"] = df["span_count"]
    d = df[is_eng].assign(frac_bin=df["ai_fraction_char"].map(frac_bin), nblk_bin=df["span_count"].map(block_count_bin))

    # ---- fully AI (dataset x generator x task balance within domain)
    ai_sources = ["raid", "llmtrace_detection", "llmtrace_classification"] + ((v2.get("fully_ai_creative") or {}).get("sources") or [])
    ai = d[(d["authorship"] == "ai") & d["dataset_key"].isin(ai_sources)]
    ai = ai.assign(task=ai["extensions"].map(lambda e: (jl(e) or {}).get("generation_task")),
                   dataset_family=ai["dataset_key"].map(lambda k: "llmtrace" if k.startswith("llmtrace") else k))
    for dom in CORE + ["narrative_summary"]:
        pool = ai[ai["domain"] == dom]
        sel.take(pool, q(300), "evaluation", "fully_ai_core" if dom in CORE else "fully_ai_noncore",
                 {"domain": dom}, strata=["dataset_family", "generator"], one_per_ind=True)

    # ---- controlled replacement (LLMTrace fill_gaps; exact insertion spans)
    rep = d[(d["mix_mechanism"] == "sentence_replacement") & (d["annotation_completeness"] == "complete")]
    for dom in CORE + ["narrative_summary"]:
        pool = rep[rep["domain"] == dom]
        sel.take(pool, q(200), "evaluation", "controlled_replacement" if dom in CORE else "controlled_replacement_noncore",
                 {"domain": dom, "mechanism": "sentence_replacement"}, strata=["frac_bin", "nblk_bin"])

    # ---- DAMASHA tagged mixtures: mechanism/domain/generator unknown (separately named diagnostic)
    if "damasha_tagged_mixed_unknown_mechanism" not in (v2.get("drop_panels") or []):
        dam = d[d["dataset_key"] == "damasha"].copy()
        dam["eligibility"] = "eligible"  # eligibility reason 'metadata absent' does not bar this diagnostic panel
        sel.take(dam[df.loc[dam.index, "eligibility"] != "quarantined"], q(250), "evaluation", "damasha_tagged_mixed_unknown_mechanism",
                 {"mechanism": "unknown_tagged_ai_segments"}, strata=["frac_bin", "nblk_bin"])

    # ---- v2: block replacement into human documents (generate-heterogeneous test split) + matched controls
    if v2.get("heterogeneous_block_replacement"):
        cfg = v2["heterogeneous_block_replacement"]
        het = d[d["dataset_key"] == cfg["source"]]
        hm = het[het["authorship"] == "mixed"].assign(src_ds=lambda x: x["original_labels"].map(lambda s_: json.loads(s_)["source_dataset"]))
        # 594 test documents come from 91 books/reports: allow several excerpts per group after a one-per-group
        # pass (clustered intervals account for it); a group never crosses roles.
        chosen = sel.take(hm, q(cfg["target_mixed"]), "evaluation", "heterogeneous_block_replacement",
                          {"mechanism": "block_replacement"}, strata=["src_ds", "generator"], allow_shared_group=True)
        if cfg.get("include_matched_controls"):
            src_keys = set(chosen["source_group_id"])
            ctl = het[(het["authorship"] == "human") & het["source_group_id"].isin(src_keys)]
            for r in ctl.itertuples(index=False):
                sel.picks.append({"example_id": r.example_id, "panel": "heterogeneous_matched_human_controls", "role": "evaluation",
                                  "rank": None, "leakage_group_id": r.leakage_group_id, "cell": json.dumps({"unit": "matched_control"})})
                sel.group_role[r.leakage_group_id] = "evaluation"
            sel.cells.append({"panel": "heterogeneous_matched_human_controls", "role": "evaluation", "cell_unit": "matched_controls_added",
                              "requested": int(len(chosen)), "eligible": int(len(ctl)), "eligible_after_group_constraints": int(len(ctl)),
                              "selected": int(len(ctl)), "missing": int(len(chosen) - len(ctl)), "excluded_by_reason": {}})

    # ---- v2: LAMP (MFA writers editing LLM paragraphs): AI draft + human-edited version, as pairs
    if v2.get("lamp_human_edited_ai"):
        lp = d[d["dataset_key"] == "lamp"]
        drafts = lp[lp["variant"] == "preedit"]
        chosen = sel.take(drafts, q(v2["lamp_human_edited_ai"]["target_pairs"]), "evaluation", "lamp_human_edited_ai",
                          {"unit": "pair"}, strata=["subdomain", "generator"], one_per_ind=False)
        pids = set(chosen["extensions"].map(lambda e: json.loads(e)["pair_id"]))
        edits = lp[(lp["variant"] == "postedit") & lp["extensions"].map(lambda e: json.loads(e)["pair_id"] in pids)]
        for r in edits.itertuples(index=False):
            sel.picks.append({"example_id": r.example_id, "panel": "lamp_human_edited_ai", "role": "evaluation", "rank": None,
                              "leakage_group_id": r.leakage_group_id, "cell": json.dumps({"unit": "pair_member"})})
        sel.cells.append({"panel": "lamp_human_edited_ai", "role": "evaluation", "cell_unit": "edited_members_added",
                          "requested": int(len(chosen)), "eligible": int(len(edits)), "eligible_after_group_constraints": int(len(edits)),
                          "selected": int(len(edits)), "missing": int(len(chosen) - len(edits)), "excluded_by_reason": {}})

    # ---- AITDNA: all eligible unique final texts
    aitdna = d[d["dataset_key"] == "aitdna"]
    sel.take(aitdna, len(aitdna) if profile == "full" else q(362), "evaluation", "aitdna_real_cowriting",
             {"dataset": "aitdna"}, one_per_ind=False)

    # ---- Beemo matched triplets (select by triplet, keep all three)
    bm = d[d["dataset_key"] == "beemo"]
    poisoned_beemo_human = "beemo:human_output" in (plan.get("poisoned_sources") or {})
    anchor = "model_output" if poisoned_beemo_human else "human_output"
    siblings = ["human_edits"] if poisoned_beemo_human else ["model_output", "human_edits"]
    trip = bm[bm["variant"] == anchor].assign(category=lambda x: x["subdomain"])
    chosen_h = sel.take(trip, q(200), "evaluation", "beemo_matched_editing", {"unit": "triplet"},
                        strata=["category"])
    tids = set(chosen_h["extensions"].map(lambda e: json.loads(e)["triplet_id"]))
    rest = bm[bm["variant"].isin(siblings) & bm["extensions"].map(lambda e: json.loads(e)["triplet_id"] in tids)]
    for r in rest.itertuples(index=False):
        sel.picks.append({"example_id": r.example_id, "panel": "beemo_matched_editing", "role": "evaluation", "rank": None,
                          "leakage_group_id": r.leakage_group_id, "cell": json.dumps({"unit": "triplet_member"})})
    sel.cells.append({"panel": "beemo_matched_editing", "role": "evaluation", "cell_unit": "triplet_members_added",
                      "requested": len(siblings) * len(chosen_h), "eligible": int(len(rest)), "eligible_after_group_constraints": int(len(rest)),
                      "selected": int(len(rest)), "missing": int(len(siblings) * len(chosen_h) - len(rest)), "excluded_by_reason": {}})

    # ---- EditLens in-family reference (test split only)
    el = d[(d["dataset_key"] == "editlens_iclr") & (d["original_split"] == "test")]
    el = el.assign(tt=el["original_labels"].map(lambda s: json.loads(s)["text_type"]))
    for tt in ("human_written", "ai_generated", "ai_edited"):
        sel.take(el[el["tt"] == tt], q(80), "evaluation", "editlens_in_family_reference", {"text_type": tt},
                 strata=["subdomain"])

    # ---- source-matched human references for selected AI/mixed groups (benchmark-provided)
    sel_ai_groups = {p["leakage_group_id"] for p in sel.picks if p["panel"] in
                     ("fully_ai_core", "fully_ai_noncore", "controlled_replacement", "controlled_replacement_noncore")}
    ref = d[(d["authorship"] == "human") & d["dataset_key"].isin(["raid", "llmtrace_detection", "llmtrace_classification", "hap_e2"])
            & d["leakage_group_id"].isin(sel_ai_groups)]
    for dom in CORE + ["narrative_summary"]:
        sel.take(ref[ref["domain"] == dom], q(120), "evaluation", "benchmark_source_matched_human_references",
                 {"domain": dom}, strata=["dataset_key"], one_per_ind=False, allow_shared_group=False)

    # ---- trusted human controls (evaluation), then calibration (disjoint groups)
    hum = d[(d["authorship"] == "human")]
    beemo_groups = {p["leakage_group_id"] for p in sel.picks if p["panel"] == "beemo_matched_editing"}
    for dom in CORE:
        src = trusted[dom]
        pool = hum[(hum["dataset_key"] == src) & (hum["provenance_tier"] == "independently_audited")]
        strata = ["subdomain", "config"] if src.startswith("scientific_papers") else ["original_split"]
        sel.take(pool, q(1000), "evaluation", "audited_human_core", {"domain": dom, "dataset": src}, strata=strata)
    stu = hum[hum["dataset_key"] == "persuade"].assign(prompt=lambda x: x["original_labels"].map(lambda s: json.loads(s)["prompt_name"]))
    sel.take(stu, q(1000), "evaluation", "student_human", {"domain": "student"}, strata=["prompt"])
    if "no_robots" not in (plan.get("poisoned_sources") or {}):
        nr = hum[(hum["dataset_key"] == "no_robots") & ~hum["leakage_group_id"].isin(beemo_groups)]
        sel.take(nr, q(500), "evaluation", "modern_human_instructions", {"domain": "instruction"}, strata=["subdomain"])

    calib_pools = {}
    for dom, src in trusted.items():
        pool = hum[(hum["dataset_key"] == src)]
        if dom == "instruction":
            pool = pool[~pool["leakage_group_id"].isin(beemo_groups)]
        strata = ["subdomain", "config"] if src.startswith("scientific_papers") else (["subdomain"] if src == "no_robots" else ["original_split"])
        if src == "persuade":
            pool = pool.assign(prompt=pool["original_labels"].map(lambda s: json.loads(s)["prompt_name"]))
            strata = ["prompt"]
        calib_pools[dom] = sel.take(pool, q(500), "calibration", "calibration_human", {"domain": dom, "dataset": src}, strata=strata)
    return sel, {"scale": scale}


# ----------------------------------------------------------------------------- build

def build(config: str, profile: str = "full", out: str | None = None) -> Path:
    os.chdir(paths.ROOT)
    plan_path = Path(config)
    plan = yaml.safe_load(plan_path.read_text())
    bench_id = plan["benchmark_id"] + ("" if profile == "full" else "-smoke")
    ver = plan["benchmark_id"].rsplit("-", 1)[-1]  # e.g. "v1", "v2"
    out_dir = Path(out) if out else paths.ARTIFACTS / (f"benchmark-{ver}" if profile == "full" else f"benchmark-smoke-{ver}")
    sources = pool_sources(plan)
    verification = (plan.get("v2") or {}).get("near_dup_verification", "minhash_estimate")
    if (out_dir / "FROZEN").exists():
        raise SystemExit(f"{out_dir} is frozen; build a new version directory instead of overwriting it")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "overlap").mkdir(parents=True)
    t0 = utcnow()
    cache_key = fingerprint({"files": {k: [(paths.NORMALISED / f"{k}.parquet").stat().st_size,
                                            (paths.NORMALISED / f"{k}.parquet").stat().st_mtime_ns] for k in sources},
                             "policy": overlap.POLICY, "verification": verification, "code": sha256_text(Path(overlap.__file__).read_text())})
    cache = paths.DATA / "cache" / f"grouped_{cache_key[:16]}"
    if (cache / "pool.parquet").exists():
        df = pd.read_parquet(cache / "pool.parquet")
        shutil.copytree(cache / "overlap", out_dir / "overlap", dirs_exist_ok=True)
        log_event("build_loaded_grouped_cache", n=len(df), cache=str(cache))
    else:
        df = load_pools(sources)
        log_event("build_loaded", n=len(df))
        df = build_groups(df, out_dir, verification)
        cache.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache / "pool.parquet", index=False)
        shutil.copytree(out_dir / "overlap", cache / "overlap", dirs_exist_ok=True)
    df = training_exposure(df, out_dir)
    df["length_bin"] = df["word_count"].map(length_bin)
    df["ai_fraction_bin"] = [("full_ai" if a == "ai" else frac_bin(f)) if a in ("ai", "mixed") else None
                             for a, f in zip(df["authorship"], df["ai_fraction_char"])]
    df["block_count_bin"] = [block_count_bin(n) if a == "mixed" else None for a, n in zip(df["authorship"], df["span_count"])]
    df["block_length_bin"] = [block_len_bin(s) if a == "mixed" else None for a, s in zip(df["authorship"], df["ai_char_spans"])]
    df["block_position"] = [S.block_position([tuple(x) for x in jl(s)], n) if (a == "mixed" and jl(s)) else None
                            for a, s, n in zip(df["authorship"], df["ai_char_spans"], df["char_count"])]
    # candidate-level group table (all candidates; evidence for overlap)
    df[["example_id", "dataset_key", "leakage_group_id", "independence_group_id", "match_fp", "text_sha256",
        "exposure__open_pangram_roberta"]].to_parquet(out_dir / "overlap" / "candidate_groups.parquet", index=False)

    sel, meta = select(df, plan, profile)
    picks = pd.DataFrame(sel.picks)
    # roles must be consistent per example and per leakage group
    role_per_ex = picks.groupby("example_id")["role"].nunique()
    assert (role_per_ex == 1).all(), "example assigned to both calibration and evaluation"
    panels = picks.groupby("example_id")["panel"].agg(lambda s: sorted(set(s)))
    role = picks.groupby("example_id")["role"].first()
    ex = df[df["example_id"].isin(panels.index)].copy()
    ex["panel_ids"] = ex["example_id"].map(lambda e: json.dumps(panels[e]))
    ex["role"] = ex["example_id"].map(role)
    ex = ex.sort_values(["role", "example_id"]).reset_index(drop=True)
    g_role = ex.groupby("leakage_group_id")["role"].nunique()
    assert (g_role == 1).all(), "leakage group spans calibration and evaluation"

    cells = pd.DataFrame(sel.cells)
    selection_hash = sha256_text(json.dumps(sorted(ex["example_id"]), ensure_ascii=False))
    text_gold_fp = fingerprint(nan_to_none([[e, t, s, a, m] for e, t, s, a, m in zip(ex["example_id"], ex["text_sha256"], ex["ai_char_spans"],
                                                                                     ex["authorship"], ex["mix_mechanism"])]))
    calib_ids = sorted(ex.loc[ex["role"] == "calibration", "example_id"])
    calib_hash = sha256_text(json.dumps(calib_ids))
    ex.to_parquet(out_dir / "examples.parquet", index=False)
    picks.to_parquet(out_dir / "selection_log.parquet", index=False)
    write_json(out_dir / "selection_cells.json", nan_to_none(sel.cells))

    acq = []
    from aidet_eval.acquire import acquisition_status

    for a in acquisition_status():
        acq.append({k: a.get(k) for k in ("key", "catalogue_name", "status", "n_files_ok", "n_files_expected")})
    eval_ex = ex[ex["role"] == "evaluation"]
    manifest = {
        "benchmark_id": bench_id, "benchmark_version": "1.0.0", "profile": profile,
        "purpose": plan.get("purpose", "development_regression"),
        "release_role": "development/regression benchmark (repeatedly inspected); not a confirmation release",
        "seed": int(plan.get("seed", 42)), "created_at": t0, "frozen_at": utcnow(),
        "resolved_plan": plan, "plan_sha256": sha256_text(plan_path.read_text()),
        "domain_mappings": {"core_domains": CORE, "trusted_human_sources": trusted_sources(plan),
                            "llmtrace": "see aidet_eval.adapters.hf_sources.LLMTRACE_DOMAIN",
                            "raid": "see aidet_eval.adapters.hf_sources.RAID_DOMAIN",
                            "notes": ["LLMTrace 'story' and RAID 'books' are plot summaries -> domain narrative_summary (not creative)",
                                      "LLMTrace 'article' (web blogs) not mapped to research",
                                      "creative AI/mixed cells have no evidence-supported source -> explicit shortfall"]},
        "reserve_sources": RESERVE_ONLY,
        "selection_policy": {"sampling": "hash-ranked (sha256(seed|panel|example_id)), round-robin over strata, one per leakage group per panel, one per independence group first",
                             "order": ["fully_ai", "controlled_replacement", "damasha", "aitdna", "beemo", "editlens_in_family", "matched_human_refs", "trusted_human_eval", "calibration"],
                             "no_detector_scores_used": True},
        "length_function": "word_count = number of maximal non-whitespace runs (regex \\S+)",
        "length_bins": LENGTH_BINS, "ai_fraction_bins": FRAC_BINS,
        "excerpt_policy": "pg19, scientific_papers body, wikitext: one paragraph-aligned contiguous excerpt per parent, hash-chosen start; other sources whole documents",
        "offset_convention": "Unicode code points, zero-based, half-open [start,end) into canonical text",
        "counts": {"examples": int(len(ex)), "evaluation": int(len(eval_ex)), "calibration": int(len(calib_ids)),
                   "evaluation_leakage_groups": int(eval_ex["leakage_group_id"].nunique()),
                   "evaluation_independence_groups": int(eval_ex["independence_group_id"].nunique()),
                   "by_panel": picks.groupby("panel").size().to_dict(),
                   "by_role_authorship": {f"{a}|{b}": int(n) for (a, b), n in ex.groupby(["role", "authorship"]).size().items()}},
        "cells": sel.cells,
        "fingerprints": {"selection_sha256": selection_hash, "text_gold_fingerprint": text_gold_fp,
                         "calibration_membership_sha256": calib_hash},
        "calibration": {"ids_sha256": calib_hash, "n": len(calib_ids),
                        "by_domain": ex[ex["role"] == "calibration"].groupby("domain").size().to_dict(),
                        "weights": "empirical (equal target 500 per domain; actual counts above)"},
        "acquisition_status": acq,
        "overlap": read_json(out_dir / "overlap" / "overlap_stats.json"),
        "training_exposure": read_json(out_dir / "overlap" / "training_exposure_audit.json"),
    }
    manifest = nan_to_none(json.loads(json.dumps(manifest, default=str).replace("NaN", "null")))
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    ex_hash = sha256_text((out_dir / "examples.parquet").read_bytes().hex()[:0] + selection_hash + text_gold_fp)
    manifest["benchmark_hash"] = sha256_text(manifest_bytes.decode() + ex_hash)
    write_json(out_dir / "manifest.json", json.loads(json.dumps(manifest, default=str)))
    (out_dir / "FROZEN").write_text(json.dumps({"benchmark_hash": manifest["benchmark_hash"], "frozen_at": manifest["frozen_at"],
                                                "note": "Do not modify. Any change to selection, gold, sampling or domains requires a new version."}, indent=1))
    log_event("build_done", out=str(out_dir), **{k: v for k, v in manifest["counts"].items() if not isinstance(v, dict)})
    return out_dir


def validate_benchmark(bench: str) -> bool:
    d = Path(bench)
    ex = pd.read_parquet(d / "examples.parquet")
    problems = []
    if ex["example_id"].duplicated().any():
        problems.append("duplicate example ids")
    if (ex.groupby("leakage_group_id")["role"].nunique() > 1).any():
        problems.append("leakage group spans calibration and evaluation")
    for r in ex.itertuples(index=False):
        if sha256_text(r.text) != r.text_sha256:
            problems.append(f"text hash mismatch {r.example_id}")
        try:
            S.validate_text(r.text)
            S.validate_intervals(jl(r.ai_char_spans), len(r.text))
        except S.SpanError as e:
            problems.append(f"{r.example_id}: {e}")
        if r.eligibility != "eligible" and r.dataset_key != "damasha":
            problems.append(f"ineligible example selected {r.example_id}")
    man = read_json(d / "manifest.json")
    if man["counts"]["examples"] != len(ex):
        problems.append("manifest count mismatch")
    ok = not problems
    write_json(d / "validation.json", {"validated_at": utcnow(), "ok": ok, "problems": problems[:200], "n_problems": len(problems)})
    print("benchmark valid" if ok else f"{len(problems)} problems: {problems[:5]}")
    return ok
