# Benchmark card: ai-detector-panels-v2 v1.0.0 (full)

- **Purpose:** development_regression — development/regression benchmark (repeatedly inspected); not a confirmation release
- **Benchmark hash:** `0393c8cb5081ffd59d76b62ba3ec6f2a2883d889a0a16243f0eb7387b51023e5`
- **Selection / text+gold / calibration fingerprints:** `8ae811c6f4e2987a` / `055a63346bbb3d04` / `695f0154f24682ec`
- **Window manifest (shared_windows_v1):** `3ae7adfb8c4b7509029c5b060b88550025724401c3aa1f3d82cb407ce7ccd34d` — 42399 windows
- **Frozen:** 2026-10-06T10:03:55+00:00 · seed 42

## What it measures

Where a detector succeeds and fails on two independent axes, **domain/category** and **human / AI / mixed authorship**, with mixture mechanism as a third axis. The primary comparison is domain-specific detection (fully-AI recall, AI-insertion recall) against domain-specific human false-positive rates at operating points frozen on disjoint human calibration documents. There is deliberately no single overall score.

## Composition (realised)

Evaluation inputs: **12022** unique texts in 10244 leakage groups / 9893 independence groups. Calibration: **3500** human documents, leakage-group-disjoint from evaluation.

| panel | role | n | authorship | domains |
|---|---|---|---|---|
| aitdna_real_cowriting | evaluation | 362 | {'mixed': 267, 'human': 95} | {'argumentative': 121, 'creative': 116, 'expository': 115, 'research_peer_review': 10} |
| audited_human_core | evaluation | 5000 | {'human': 5000} | {'news': 1000, 'reviews': 1000, 'creative': 1000, 'research': 1000, 'expository': 1000} |
| beemo_matched_editing | evaluation | 600 | {'mixed': 200, 'human': 200, 'ai': 200} | {'instruction': 600} |
| benchmark_source_matched_human_references | evaluation | 720 | {'human': 720} | {'creative': 120, 'news': 120, 'narrative_summary': 120, 'reviews': 120, 'research': 120, 'expository': 120} |
| calibration_human | calibration | 3500 | {'human': 3500} | {'news': 500, 'reviews': 500, 'instruction': 500, 'student': 500, 'creative': 500, 'research': 500, 'expository': 500} |
| controlled_replacement | evaluation | 800 | {'mixed': 800} | {'expository': 200, 'research': 200, 'news': 200, 'reviews': 200} |
| controlled_replacement_noncore | evaluation | 200 | {'mixed': 200} | {'narrative_summary': 200} |
| editlens_in_family_reference | evaluation | 240 | {'ai': 80, 'mixed': 80, 'human': 80} | {'editlens_in_family': 240} |
| fully_ai_core | evaluation | 1500 | {'ai': 1500} | {'creative': 300, 'reviews': 300, 'news': 300, 'expository': 300, 'research': 300} |
| fully_ai_noncore | evaluation | 300 | {'ai': 300} | {'narrative_summary': 300} |
| heterogeneous_block_replacement | evaluation | 300 | {'mixed': 300} | {'research': 137, 'creative': 115, 'expository': 48} |
| heterogeneous_matched_human_controls | evaluation | 300 | {'human': 300} | {'research': 137, 'creative': 115, 'expository': 48} |
| lamp_human_edited_ai | evaluation | 200 | {'mixed': 100, 'ai': 100} | {'nonfiction_personal': 116, 'creative': 84} |
| modern_human_instructions | evaluation | 500 | {'human': 500} | {'instruction': 500} |
| student_human | evaluation | 1000 | {'human': 1000} | {'student': 1000} |

### Requested vs eligible vs selected (every cell)

| panel | role | cell | requested | eligible | selected | missing |
|---|---|---|---|---|---|---|
| fully_ai_core | evaluation | {'domain': 'research'} | 300 | 5888 | 300 | 0 |
| fully_ai_core | evaluation | {'domain': 'news'} | 300 | 5246 | 300 | 0 |
| fully_ai_core | evaluation | {'domain': 'reviews'} | 300 | 5767 | 300 | 0 |
| fully_ai_core | evaluation | {'domain': 'expository'} | 300 | 5397 | 300 | 0 |
| fully_ai_core | evaluation | {'domain': 'creative'} | 300 | 12636 | 300 | 0 |
| fully_ai_noncore | evaluation | {'domain': 'narrative_summary'} | 300 | 5605 | 300 | 0 |
| controlled_replacement | evaluation | {'domain': 'research', 'mechanism': 'sentence_replacement'} | 200 | 1444 | 200 | 0 |
| controlled_replacement | evaluation | {'domain': 'news', 'mechanism': 'sentence_replacement'} | 200 | 1730 | 200 | 0 |
| controlled_replacement | evaluation | {'domain': 'reviews', 'mechanism': 'sentence_replacement'} | 200 | 1216 | 200 | 0 |
| controlled_replacement | evaluation | {'domain': 'expository', 'mechanism': 'sentence_replacement'} | 200 | 1734 | 200 | 0 |
| controlled_replacement | evaluation | {'domain': 'creative', 'mechanism': 'sentence_replacement'} | 200 | 0 | 0 | 200 |
| controlled_replacement_noncore | evaluation | {'domain': 'narrative_summary', 'mechanism': 'sentence_replacement'} | 200 | 1746 | 200 | 0 |
| heterogeneous_block_replacement | evaluation | {'mechanism': 'block_replacement'} | 300 | 594 | 300 | 0 |
| heterogeneous_matched_human_controls | evaluation | {'unit': 'matched_controls_added'} | 300 | 300 | 300 | 0 |
| lamp_human_edited_ai | evaluation | {'unit': 'pair'} | 100 | 1282 | 100 | 0 |
| lamp_human_edited_ai | evaluation | {'unit': 'edited_members_added'} | 100 | 100 | 100 | 0 |
| aitdna_real_cowriting | evaluation | {'dataset': 'aitdna'} | 362 | 362 | 362 | 0 |
| beemo_matched_editing | evaluation | {'unit': 'triplet'} | 200 | 2187 | 200 | 0 |
| beemo_matched_editing | evaluation | {'unit': 'triplet_members_added'} | 400 | 400 | 400 | 0 |
| editlens_in_family_reference | evaluation | {'text_type': 'human_written'} | 80 | 2038 | 80 | 0 |
| editlens_in_family_reference | evaluation | {'text_type': 'ai_generated'} | 80 | 2046 | 80 | 0 |
| editlens_in_family_reference | evaluation | {'text_type': 'ai_edited'} | 80 | 2031 | 80 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'research'} | 120 | 381 | 120 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'news'} | 120 | 404 | 120 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'reviews'} | 120 | 378 | 120 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'expository'} | 120 | 367 | 120 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'creative'} | 120 | 185 | 120 | 0 |
| benchmark_source_matched_human_references | evaluation | {'domain': 'narrative_summary'} | 120 | 368 | 120 | 0 |
| audited_human_core | evaluation | {'domain': 'research', 'dataset': 'scientific_papers_detok'} | 1000 | 13465 | 1000 | 0 |
| audited_human_core | evaluation | {'domain': 'news', 'dataset': 'cnn_dailymail'} | 1000 | 15927 | 1000 | 0 |
| audited_human_core | evaluation | {'domain': 'reviews', 'dataset': 'imdb'} | 1000 | 18000 | 1000 | 0 |
| audited_human_core | evaluation | {'domain': 'expository', 'dataset': 'wikitext_detok'} | 1000 | 5117 | 1000 | 0 |
| audited_human_core | evaluation | {'domain': 'creative', 'dataset': 'pg19'} | 1000 | 5138 | 1000 | 0 |
| student_human | evaluation | {'domain': 'student'} | 1000 | 25995 | 1000 | 0 |
| modern_human_instructions | evaluation | {'domain': 'instruction'} | 500 | 9450 | 500 | 0 |
| calibration_human | calibration | {'domain': 'research', 'dataset': 'scientific_papers_detok'} | 500 | 13465 | 500 | 0 |
| calibration_human | calibration | {'domain': 'news', 'dataset': 'cnn_dailymail'} | 500 | 15927 | 500 | 0 |
| calibration_human | calibration | {'domain': 'reviews', 'dataset': 'imdb'} | 500 | 18000 | 500 | 0 |
| calibration_human | calibration | {'domain': 'expository', 'dataset': 'wikitext_detok'} | 500 | 5117 | 500 | 0 |
| calibration_human | calibration | {'domain': 'creative', 'dataset': 'pg19'} | 500 | 5138 | 500 | 0 |
| calibration_human | calibration | {'domain': 'student', 'dataset': 'persuade'} | 500 | 25995 | 500 | 0 |
| calibration_human | calibration | {'domain': 'instruction', 'dataset': 'no_robots'} | 500 | 9450 | 500 | 0 |

## Domain mapping decisions

- LLMTrace 'story' and RAID 'books' are plot summaries -> domain narrative_summary (not creative)
- LLMTrace 'article' (web blogs) not mapped to research
- creative AI/mixed cells have no evidence-supported source -> explicit shortfall
- Trusted (independently audited / curator-attested) human sources per domain: {'research': 'scientific_papers_detok', 'news': 'cnn_dailymail', 'reviews': 'imdb', 'expository': 'wikitext_detok', 'creative': 'pg19', 'student': 'persuade', 'instruction': 'no_robots'}

## Label and span conventions

- Offsets: Unicode code points, zero-based, half-open [start,end) into canonical text. `ai_char_spans: null` = unavailable; `[]` = exhaustively empty.
- Human spans are the complement of AI spans **only** under complete annotation; process-level labels (Beemo expert edits, EditLens AI edits) carry no span gold.
- `ai_fraction_char` = |union of AI intervals| / canonical length (whitespace included), null when not identifiable.
- Length function: word_count = number of maximal non-whitespace runs (regex \S+); bins ['lt50w', '50_149w', '150_299w', '300_599w', '600_1199w', 'ge1200w'].
- Excerpts: pg19, scientific_papers body, wikitext: one paragraph-aligned contiguous excerpt per parent, hash-chosen start; other sources whole documents.

## Grouping, overlap and contamination

- Leakage groups: union-find over native keys (topic/source/prompt/book/article/essay/session IDs), match-normalised fingerprints (>= 8 tokens) and MinHash near-duplicates (policy: `{"normalisation": "lowercase; tokens = [a-z0-9]+ runs", "shingle_words": 5, "num_perm": 64, "lsh_bands": 16, "lsh_rows": 4, "jaccard_threshold": 0.8, "hash": "xxhash64 seed 0", "min_tokens_for_near_dup": 20, "min_tokens_for_exact_fingerprint_link": 8}`).
- Near-duplicate statistics: 32256 verified pairs from 125706 LSH candidates; cross-dataset exact links {'scientific_papers|scientific_papers_detok': 13858, 'wikitext|wikitext_detok': 5108, 'llmtrace_classification|llmtrace_detection': 2954, 'beemo|no_robots': 2187, 'cnn_dailymail|editlens_iclr': 155, 'editlens_iclr|xsum': 199, 'llmtrace_detection|raid': 22, 'llmtrace_classification|raid': 5, 'editlens_iclr|llmtrace_classification': 1, 'editlens_iclr|llmtrace_detection': 12, 'cnn_dailymail|llmtrace_detection': 8, 'beemo|damasha|no_robots|raid|scientific_papers|storyscope|xsum': 1, 'llmtrace_classification|llmtrace_detection|raid': 1, 'raid|scientific_papers': 2, 'cnn_dailymail|llmtrace_classification': 1, 'imdb|raid': 2}.
- Post-freeze audit with exact shingle Jaccard: all 20 selected-example pairs with J >= 0.8 were already grouped (none crosses calibration/evaluation); 2 EditLens test-split items have J >= 0.8 to EditLens training texts but are marked `possible_overlap` rather than `known_overlap` (MinHash estimate noise). Later versions should use `verification='exact_jaccard'`.
- Pangram training exposure: {"cnn_dailymail": 173, "editlens_iclr": 19741, "llmtrace_classification": 11, "llmtrace_detection": 53, "xsum": 233} candidate records are known overlaps with EditLens train/val; datasets marked possible_overlap: ['cnn_dailymail', 'llmtrace_classification', 'llmtrace_detection', 'xsum', 'editlens_iclr (test splits: same generation pipeline)']. `no_detected_overlap` is not proof of no pretraining exposure.
- Optional registry for future models: `configs/training_registry.yaml` (text_sha256 / source_group_id / example_id lists) marks `known_overlap` and enables an independent view.

## Sources (all 16 catalogue entries)

| source | acquisition | measured rows | role |
|---|---|---|---|
| llmtrace_detection | complete | 79342 | primary: controlled_replacement (fill_gaps), fully_ai, source-matched human references |
| llmtrace_classification | complete | 589086 | primary: fully_ai candidates and matched human references (linked to detection via topic_id/text) |
| raid | complete | 8326920 | primary: fully_ai (clean attack=='none' labelled training release); human rows as benchmark-provided references; attacks reserve; test.csv unlabelled (never scored) |
| damasha | complete | 676844 | diagnostic panel only (mechanism/domain/generator unknown at row level); attacked files reserve |
| aitdna | complete | 3258 | primary: aitdna_real_cowriting (all 362 unique texts) |
| beemo | complete | 2187 | primary: beemo_matched_editing (triplets) |
| imdb | complete | 100000 | primary: audited_human_core reviews + calibration |
| xsum | complete | 226711 | primary: audited_human_core news + calibration |
| scientific_papers | complete | 349128 | primary: audited_human_core research (abstracts + body passages) + calibration; @xmath/@xcite passages excluded |
| wikitext | complete | 1809468 | primary: audited_human_core expository + calibration |
| pg19 | complete | 28752 | primary: audited_human_core creative (historical literature) + calibration |
| persuade | complete | 285383 | primary: student_human + calibration |
| no_robots | complete | 10000 | primary: modern_human_instructions + calibration (Coding category excluded as non-prose) |
| mage | complete | 436606 | reserve (in-distribution regression source; hashes kept for overlap audits) |
| editlens_iclr | complete | 80610 | in-family reference panel (test split only); train/val define Pangram training exposure |
| coauthor | complete | 1447 | reserve (dataset licence not stated; legacy GPT-3; replay not independently validated) |
| storyscope | complete | 1484 | fully_ai_creative (2026 frontier generators; no human text released) |
| hap_e2 | complete | 74610 | fully_ai_creative continuations + matched human continuations |
| lamp | complete | 1282 | creative human_edited_ai (MFA writers editing LLM paragraphs) |
| cnn_dailymail | complete | 311971 | historical_human_news (replaces XSum, which the user will train on) |
| generate_heterogeneous | complete | 1188 | block_replacement mixtures (Claude generators) + matched human controls; test split only (group-isolated from train) |

Reserve sources: mage: reserve: in-distribution regression source; training exposure of future models unknown; coauthor: reserve: dataset licence not stated; legacy GPT-3; replay not independently validated

## Known limitations

- v2 changes vs v1: DAMASHA panel dropped; XSum removed from evaluation and calibration (user will train on it); news human controls now CNN/DailyMail; WikiText and PubMed human texts losslessly detokenised (PubMed stays lowercase); creative fully-AI (StoryScope, HAP-E-2), creative matched human continuations (HAP-E-2), block replacement (generate-heterogeneous v1.3.0 test split) and LAMP human-edited AI added; near-duplicates verified with exact shingle Jaccard.
- CNN/DailyMail also overlaps EditLens training data (173 candidate records known_overlap); an excluding-known-overlap view is provided.
- Beige Book (Federal Reserve report) human controls are flagged far more often than other human text by both baselines; expository block-replacement recall is dominated by this genre effect.
- generate-heterogeneous: test split only (group-isolated from its train split, which is user training data); 300 mixtures from 91 books/reports, so excerpts are clustered (cluster-bootstrap intervals). Generators are Claude-family only.
- StoryScope human source stories are not released; its AI stories are paired only by prompt. HAP-E-2 AI texts are style continuations seeded by human text.
- LAMP and Beemo edited texts carry process labels only (no span gold).
- Sentence-level replacement (LLMTrace fill_gaps) is still a single generator (gemini-2.5-flash), and has no creative cell.
- PubMed human research text remains lowercase after detokenisation; RAID material is the labelled training release; IMDb/PERSUADE grouping limits as in v1.
- Development/regression benchmark; not a confirmation release. Several sources are non-commercial; do not redistribute texts.

