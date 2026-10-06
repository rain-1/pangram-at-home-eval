# Execution report — benchmark-v1 with the two Open Pangram baselines (2026-10-06)

Structured, timestamped, secret-redacted log: `artifacts/logs/execution_log.jsonl`. Benchmark card: `docs/BENCHMARK_CARD.md`.

## Status

| deliverable | status |
|---|---|
| Acquisition + inventory of all 16 catalogue sources | **complete** (all 16 fully downloaded and checksum-verified; 38 GB) |
| Normalisation adapters, provenance audit, overlap/leakage groups | complete (`data/normalised`, `data/inventory`, `artifacts/benchmark-v1/overlap`) |
| Frozen, source-disjoint benchmark manifest | complete — `artifacts/benchmark-v1` (hash `e552327a…`), frozen before any evaluation score existed |
| Inference adapters for both baselines + extension interface | complete; loader parity vs pinned upstream passes for both |
| Baseline runs | **complete**: 2 models × 2 protocols × 14,552 documents (98,148 inputs, 0 errors, 0 abstentions) |
| Frozen calibration thresholds (1% primary, 5%) | complete: 8 operating points, all `ok`, n = 3,500 |
| `results.full.json` / `results.summary.json` + schemas | complete, run status `complete`, all 14 semantic invariants pass; summary re-derives byte-identically |
| Local viewer | complete (5 screens; verified in headless Chromium, no JS errors) |
| Tests | 43 passed (no corpora/GPU needed) |
| Smoke profile | complete (`artifacts/benchmark-smoke-v1`, run status `partial` by design) |

Nothing was blocked. Gated access (both Pangram checkpoints, Meta Llama 3.2, `pangram/editlens_iclr`) already existed on the
account; no new agreements were accepted. Per your instruction this session, model weights were downloaded locally
(an exception to the sibling repo's AGENTS.md storage preference).

## Acquisition (all 16)

| source | status | measured rows | role |
|---|---|---|---|
| LLMTrace detection | complete | 79,342 | primary (replacement, fully-AI, matched refs) |
| LLMTrace classification | complete | 589,086 | primary (fully-AI, matched refs) |
| RAID | complete | 8,326,920 (train 5,615,820; extra 2,039,100; unlabelled test 672,000) | primary (clean labelled train release); attacks reserve |
| DAMASHA | complete | 676,844 | diagnostic (mechanism unknown); attacks reserve |
| AITDNA | complete | 3,258 view rows = 362 texts | primary |
| Beemo | complete | 2,187 | primary |
| IMDb | complete | 100,000 | primary human reviews + calibration |
| XSum | complete | 226,711 | primary human news + calibration |
| Scientific Papers (archive.org mirror) | complete | 349,128 | primary human research + calibration |
| WikiText-103 raw | complete | 1,809,468 lines / 28,600 articles | primary human expository + calibration |
| PG-19 (GCS original) | complete | 28,752 books | primary human creative + calibration |
| PERSUADE 2.0 (original) | complete | 25,995 essays | primary student + calibration |
| No Robots | complete | 10,000 | primary modern instructions + calibration |
| MAGE | complete | 436,606 | reserve (label convention 1 = human verified) |
| EditLens ICLR | complete | 80,610 | in-family reference (test only); defines training exposure |
| CoAuthor | complete | 1,447 sessions | reserve (licence not stated) |

Catalogue corrections are in `data/inventory/source_audit.json`. The original catalogue in `spec/` is unchanged.

## What was evaluated

11,052 unique evaluation documents in 10,003 leakage groups and 9,740 independence groups, plus 3,500 leakage-disjoint
human calibration documents. Each of the four model × protocol combinations scored all 14,552 documents.

## Headline results (shared_windows_v1, 1% calibration FPR; full tables in `report.md` and the viewer)

| | RoBERTa | Llama |
|---|---|---|
| Audited human FPR (5 core domains, n = 5,000) | 0.04% (2) | 0.04% (2) |
| Student essays FPR (n = 1,000) | 0.20% | 0.10% |
| **Modern human instruction responses FPR (n = 500)** | **7.8%** | **7.8%** |
| Fully-AI recall, core domains (n = 1,200) | 56.6% | 62.4% |
| Sentence-replacement recall (n = 800) | 14.4% | 10.4% |
| AITDNA co-written recall (n = 267) / human FPR (n = 95) | 74.2% / 0% | 82.4% / 0% |
| Beemo: model output / expert-edited AI recall; human FPR | 27.0% / 19.0%; 5.5% | 46.5% / 23.0%; 4.0% |

Findings:

* The pooled 1% threshold is set almost entirely by No Robots instruction responses: 32–34 of the 35 calibration false
  positives are instruction texts. As a result, historical audited humans are flagged ~0.04% of the time, while recall on
  fully-AI text is only ~55–65%.
* Detection depends strongly on the generator. Base (non-chat) models continuing from a title are almost never caught
  (gpt2 0–2%, mpt 0–2%, mistral 2–5%, cohere 5–16%). Chat models are mostly caught (chatgpt 89–100%, llama-chat 86–96%).
* Inserted AI sentences (LLMTrace fill_gaps, a single generator) are rarely detected at 1% (5–18% per domain). At 5%,
  recall reaches 19–47% by domain.
* Paired Llama − RoBERTa differences: fully-AI recall +5.8 pp [3.4, 8.2]; AITDNA +8.2 pp; replacement −4.0 pp
  (RoBERTa better). The audited-human FPR difference is ~0.

## Shortfalls and limitations

* **Creative domain:** no catalogue source has evidence-supported creative AI or mixed text, so these cells are empty.
  Plot summaries are reported separately as `narrative_summary`.
* **Generators:** fill_gaps replacement was generated by gemini-2.5-flash only.
* **DAMASHA:** the release has no row-level domain, generator, or mechanism metadata.
* **Tokenised human corpora:** the research and expository trusted human texts are pre-tokenised.
* **Training exposure:** 233 XSum documents appear in EditLens training data (marked; an "excluding known overlap" view
  is provided). A post-freeze exact-Jaccard audit found that 2 EditLens test items should have been marked
  `known_overlap`; v1 was kept frozen and later versions should use `verification='exact_jaccard'`.
* **Grouping:** IMDb has no movie IDs and PERSUADE has no writer IDs. AITDNA's 95 human controls give only coarse FPR
  bounds (the degenerate CI is flagged).
* **Release role:** this is a development/regression benchmark, not a confirmation release.
* **Numerics:** bf16/NF4 make scores batch-composition sensitive at ~0.01 (documented in `artifacts/parity`).

---

# Follow-up (same day): calibration sensitivity, benchmark-v2

## Calibration sensitivity (exploratory; official thresholds unchanged)

The 1% order-statistic threshold was re-derived on each calibration dataset alone (n = 500, k = 5), with 2,000-replicate
bootstrap intervals (`calibration_sensitivity.json` in each run; viewer tab "Calibration sensitivity").

- **No Robots is the only dataset whose interval overlaps no other dataset's**, for both models, both protocols, and both
  targets. Its 1% thresholds are 0.97–1.00; every other dataset's are 0.06–0.53.
- In v2 (shared windows, 1%), the alternative calibration choices would do the following:

| Llama | threshold | fully-AI recall | sentence-replacement recall | block-replacement recall | instruction FPR | student FPR |
|---|---|---|---|---|---|---|
| pooled (official) | 0.660 | 62.5% | 10.5% | 35.3% | 7.8% | 0.1% |
| leave No Robots out | 0.379 | 82.2% | 30.8% | 64.0% | 24.6% | 2.0% |
| max over datasets | 0.969 | 34.9% | 0.8% | 9.0% | 2.0% | 0.0% |

- Equal-weight pooling equals the pooled rule here, because all calibration datasets have n = 500.
- Excluding an outlying dataset redefines what "1% FPR" refers to; its FPR (~25% here) must still be reported.
- Interpretation is open. The models may treat assistant-register human writing as AI-like, or some 2023 vendor-written
  No Robots responses may have been LLM-assisted. The data cannot distinguish these.

## benchmark-v2 (`artifacts/benchmark-v2`, run `ai-detector-panels-v2__0393c8cb50`, status complete)

Changes vs v1:
- **Removed:** the DAMASHA panel; XSum (from evaluation and calibration).
- **News human controls:** now CNN/DailyMail. Note that 173 CNN/DailyMail candidate records are also in EditLens training data.
- **Detokenisation:** WikiText and PubMed were losslessly detokenised, with every edit logged.
- **Added:**
  - creative fully-AI: StoryScope (2026 frontier models) and HAP-E-2;
  - creative matched human continuations: HAP-E-2;
  - block replacement: 300 mixtures plus 300 matched controls from the generate-heterogeneous v1.3.0 test split;
  - LAMP: 100 pairs of AI draft plus MFA-edited version.
- **Near-duplicates:** verified with exact Jaccard.

Size and coverage: 12,022 evaluation and 3,500 calibration documents. All 4 model × protocol combinations scored
everything. 31,381 inputs per model were reused from v1 (identical model, parameters, text, and offset).

Headline (shared windows, 1%, RoBERTa | Llama):

| | RoBERTa | Llama |
|---|---|---|
| Creative fully-AI recall (now measurable) | 78.0% | 61.7% |
| Block replacement, all 300 | 28.7% | 35.3% |
| Block replacement: Claude Haiku 4.5 | 57% | 72% |
| Block replacement: Claude Sonnet 5.5 / Opus 5.5 | ~10% | ~11% |
| Heterogeneous matched-control FPR | 5.7% | 4.3% |
| LAMP: AI drafts | 99% | 100% |
| LAMP: MFA-edited | 84% | 82% |

- **Block-replacement controls:** the matched-control FPR comes entirely from Beige Book (17/48 and 12/48 flagged). The
  Gutenberg, Standard Ebooks, and JMLR controls are 0–1 flagged out of 252, so expository block-replacement recall is
  confounded by genre.
- **Frontier generators:** RoBERTa holds up much better than Llama on StoryScope frontier stories (gpt-5.4 96% vs 26%;
  gemini-3-flash 87% vs 22%). Both miss gpt-5-mini and Claude Sonnet 4.6.
- **Detokenisation:** on identical documents, detokenising raised mean scores slightly (e.g. WikiText RoBERTa
  0.010 → 0.075), with 0% above threshold either way. The tokenised format was making human controls look *more* human,
  not less, so v2's controls are the fairer test.

---

# benchmark-v3 (current): No Robots excluded as poison

User decision: No Robots is **POISON**. `configs/benchmark_plan_v3.yaml` adds `poisoned_sources`:

- **No Robots** is removed from evaluation and calibration.
- **Beemo `human_output`** (which is No Robots text) is removed too. Beemo keeps its AI outputs and expert-edited AI texts.
- Calibration is 6 domains × 500 = 3,000.
- Run: `ai-detector-panels-v3__577af7a44f`, status complete.

A label bug was fixed before v3 was used: 111 generate-heterogeneous mixtures whose generator identity was never
confirmed by the API had a null generator. They are now labelled `gpt-6-luna` / `gpt-6.1-sol (requested; identity
unconfirmed)`, and v3 was rebuilt.

Results (shared windows, 1%, RoBERTa | Llama):

| | RoBERTa | Llama |
|---|---|---|
| Threshold | 0.399 | 0.379 (was ~0.66–0.75) |
| Audited human FPR | 0.8% | 0.6% |
| Student essays FPR | 1.0% | 2.0% |
| Fully-AI recall | 73.7% | 82.2% |
| Sentence insertion | 29.5% | 30.8% |
| Block insertion | 47.7% | 64.0% |
| AITDNA co-writing | 85.4% | 91.0% |
| LAMP edited | 91% | 93% |
| Beemo edited | 36% | 50% |

- Block insertion by generator: Haiku 4.5 90% | 97%; Opus 5.5 17% | 53%; Sonnet 5.5 18% | 32%; GPT-6 requested
  (unconfirmed identity) 42–57% | 57–72%.
- Heterogeneous controls FPR is 14% | 20%, driven by the Beige Book genre.

The viewer scorecard was redesigned: headline tiles, a plain-language domain table, and a threshold explorer showing the
share of each class flagged at every threshold (this replaces the histogram).
