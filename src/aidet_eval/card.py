"""Generate a benchmark card (Markdown) from a frozen benchmark manifest and the source inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aidet_eval import paths
from aidet_eval.util import read_json


V1_LIMITS = [
    "- No evidence-supported creative-writing AI or mixed source exists in the catalogue releases: creative fully-AI, replacement and matched-reference cells are empty (plot summaries are a separate `narrative_summary` domain).",
    "- All controlled sentence-replacement mixtures come from LLMTrace `fill_gaps`, generated only by gemini-2.5-flash.",
    "- Research and expository trusted human controls (Scientific Papers, WikiText) are pre-tokenised by their original releases; arXiv passages with @xmath/@xcite placeholders were excluded.",
    "- DAMASHA rows lack domain/generator/mechanism metadata; they form a separately named diagnostic panel only.",
    "- RAID material is the labelled training release; AI texts are from 2023-era generators. LLMTrace/RAID human references are benchmark-provided, not independently audited.",
    "- IMDb lacks movie IDs in the HF release, so same-movie reviews cannot be grouped; PERSUADE releases no writer IDs (each essay is its own unit).",
    "- AITDNA has 95 human-only texts: too few for precise low-FPR estimates.",
    "- This is a development/regression benchmark that will be inspected repeatedly; a separately frozen confirmation release is needed for unbiased final claims.",
    "- Rights: several sources are non-commercial (CC-BY-NC/NC-SA) or have uncleared underlying content; texts are not to be redistributed (use `export --redact`).",
]
V2_LIMITS = [
    "- v2 changes vs v1: DAMASHA panel dropped; XSum removed from evaluation and calibration (user will train on it); news human controls now CNN/DailyMail; WikiText and PubMed human texts losslessly detokenised (PubMed stays lowercase); creative fully-AI (StoryScope, HAP-E-2), creative matched human continuations (HAP-E-2), block replacement (generate-heterogeneous v1.3.0 test split) and LAMP human-edited AI added; near-duplicates verified with exact shingle Jaccard.",
    "- CNN/DailyMail also overlaps EditLens training data (173 candidate records known_overlap); an excluding-known-overlap view is provided.",
    "- Beige Book (Federal Reserve report) human controls are flagged far more often than other human text by both baselines; expository block-replacement recall is dominated by this genre effect.",
    "- generate-heterogeneous: test split only (group-isolated from its train split, which is user training data); 300 mixtures from 91 books/reports, so excerpts are clustered (cluster-bootstrap intervals). Generators are Claude-family only.",
    "- StoryScope human source stories are not released; its AI stories are paired only by prompt. HAP-E-2 AI texts are style continuations seeded by human text.",
    "- LAMP and Beemo edited texts carry process labels only (no span gold).",
    "- Sentence-level replacement (LLMTrace fill_gaps) is still a single generator (gemini-2.5-flash), and has no creative cell.",
    "- PubMed human research text remains lowercase after detokenisation; RAID material is the labelled training release; IMDb/PERSUADE grouping limits as in v1.",
    "- Development/regression benchmark; not a confirmation release. Several sources are non-commercial; do not redistribute texts.",
]


def write_card(bench: str, out: str | None = None) -> Path:
    d = Path(bench)
    m = read_json(d / "manifest.json")
    prot = read_json(d / "windows" / "protocols.json")
    inv = read_json(paths.INVENTORY / "inventory.json")
    ex = pd.read_parquet(d / "examples.parquet", columns=["example_id", "role", "panel_ids", "domain", "dataset_key", "authorship",
                                                          "mix_mechanism", "length_bin", "provenance_tier", "word_count"])
    ex["panels"] = ex["panel_ids"].map(json.loads)
    L = [f"# Benchmark card: {m['benchmark_id']} v{m['benchmark_version']} ({m['profile']})", "",
         f"- **Purpose:** {m['purpose']} — {m['release_role']}",
         f"- **Benchmark hash:** `{m['benchmark_hash']}`",
         f"- **Selection / text+gold / calibration fingerprints:** `{m['fingerprints']['selection_sha256'][:16]}` / `{m['fingerprints']['text_gold_fingerprint'][:16]}` / `{m['fingerprints']['calibration_membership_sha256'][:16]}`",
         f"- **Window manifest (shared_windows_v1):** `{prot['protocols'][0]['window_manifest_sha256']}` — {prot['n_windows']} windows",
         f"- **Frozen:** {m['frozen_at']} · seed {m['seed']}", "",
         "## What it measures", "",
         "Where a detector succeeds and fails on two independent axes, **domain/category** and **human / AI / mixed authorship**, "
         "with mixture mechanism as a third axis. The primary comparison is domain-specific detection (fully-AI recall, "
         "AI-insertion recall) against domain-specific human false-positive rates at operating points frozen on disjoint human "
         "calibration documents. There is deliberately no single overall score.", "",
         "## Composition (realised)", "",
         f"Evaluation inputs: **{m['counts']['evaluation']}** unique texts in {m['counts']['evaluation_leakage_groups']} leakage groups / "
         f"{m['counts']['evaluation_independence_groups']} independence groups. Calibration: **{m['counts']['calibration']}** human documents, leakage-group-disjoint from evaluation.", "",
         "| panel | role | n | authorship | domains |", "|---|---|---|---|---|"]
    for p in sorted({q for ps in ex["panels"] for q in ps}):
        s = ex[ex["panels"].map(lambda ps: p in ps)]
        L.append(f"| {p} | {s['role'].iloc[0]} | {len(s)} | { {k: int(v) for k, v in s['authorship'].value_counts().items()} } | { {k: int(v) for k, v in s['domain'].value_counts().items()} } |")
    L += ["", "### Requested vs eligible vs selected (every cell)", "", "| panel | role | cell | requested | eligible | selected | missing |", "|---|---|---|---|---|---|---|"]
    for c in m["cells"]:
        cell = {k[5:]: v for k, v in c.items() if k.startswith("cell_")}
        L.append(f"| {c['panel']} | {c['role']} | {cell} | {c['requested']} | {c['eligible']} | {c['selected']} | {c['missing']} |")
    L += ["", "## Domain mapping decisions", ""] + [f"- {n}" for n in m["domain_mappings"]["notes"]] + [
        f"- Trusted (independently audited / curator-attested) human sources per domain: {m['domain_mappings']['trusted_human_sources']}", "",
        "## Label and span conventions", "",
        f"- Offsets: {m['offset_convention']}. `ai_char_spans: null` = unavailable; `[]` = exhaustively empty.",
        "- Human spans are the complement of AI spans **only** under complete annotation; process-level labels (Beemo expert edits, EditLens AI edits) carry no span gold.",
        "- `ai_fraction_char` = |union of AI intervals| / canonical length (whitespace included), null when not identifiable.",
        f"- Length function: {m['length_function']}; bins {[b[0] for b in m['length_bins']]}.",
        f"- Excerpts: {m['excerpt_policy']}.", "",
        "## Grouping, overlap and contamination", "",
        f"- Leakage groups: union-find over native keys (topic/source/prompt/book/article/essay/session IDs), match-normalised fingerprints (>= 8 tokens) and MinHash near-duplicates (policy: `{json.dumps(m['overlap'].get('policy'))}`).",
        f"- Near-duplicate statistics: {m['overlap'].get('verified_pairs')} verified pairs from {m['overlap'].get('lsh_candidate_pairs')} LSH candidates; cross-dataset exact links {m['overlap'].get('cross_dataset_exact_links')}.",
        "- Post-freeze audit with exact shingle Jaccard: all 20 selected-example pairs with J >= 0.8 were already grouped (none crosses calibration/evaluation); 2 EditLens test-split items have J >= 0.8 to EditLens training texts but are marked `possible_overlap` rather than `known_overlap` (MinHash estimate noise). Later versions should use `verification='exact_jaccard'`.",
        f"- Pangram training exposure: {json.dumps(m['training_exposure']['pangram_editlens']['known_overlap_by_dataset'])} candidate records are known overlaps with EditLens train/val; datasets marked possible_overlap: {m['training_exposure']['pangram_editlens']['datasets_marked_possible_overlap']}. `no_detected_overlap` is not proof of no pretraining exposure.",
        "- Optional registry for future models: `configs/training_registry.yaml` (text_sha256 / source_group_id / example_id lists) marks `known_overlap` and enables an independent view.", "",
        "## Sources (all 16 catalogue entries)", "", "| source | acquisition | measured rows | role |", "|---|---|---|---|"]
    for s in inv["sources"]:
        L.append(f"| {s['source_id']} | {s['acquisition_status']} | {s['measured_counts'].get('rows_total')} | {s['decision_eligibility']} |")
    L += ["", "Reserve sources: " + "; ".join(f"{k}: {v}" for k, v in m["reserve_sources"].items()), "",
          "## Known limitations", ""] + (V2_LIMITS if m["resolved_plan"].get("v2") else V1_LIMITS) + [
          ""]
    path = Path(out) if out else d / "BENCHMARK_CARD.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    print(write_card(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
