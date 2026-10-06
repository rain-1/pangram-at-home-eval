# Build and run a reproducible AI-text-detector evaluation suite

## Assignment

Implement this project in the current repository, run it on the two specified Open Pangram baselines, and leave working code, reproducible benchmark manifests, actual results, and a local results viewer. This is an implementation-and-execution task, not a request for another research plan or a scaffold with mock results.

Read this document, `RESULTS_CONTRACT.md`, `ai_detector_dataset_catalogue.json`, `verified_baselines.json`, and `benchmark_plan.yaml` before implementing. The catalogue is a source-discovery starting point, not an audited dataset release. Verify its schemas, counts, access conditions, licences, and provenance against the actual downloaded revisions. The YAML is a starting experiment specification; freeze the resolved specification before inspecting evaluation scores.

The suite should tell me where a detector succeeds and where it fails. The main questions are whether it detects fully AI-generated documents and AI insertions, while rarely flagging human writing. Preserve two independent axes: **domain/category** and **human / AI / mixed authorship**. Preserve mixture mechanism as an additional axis. Do not collapse everything into a single leaderboard score.

Target roughly 9,500–10,500 evaluation inputs, with additional disjoint calibration material. Download and inventory **every source in the supplied catalogue**, including sources that ultimately remain reserve datasets. Not every downloaded source needs to enter the primary scorecard. The first release should be useful and maintainable, not a sprawling training platform.

## 1. Execution rules and deliverables

Inspect the current repository and reuse its packaging, configuration, frontend, and logging conventions. If there is no suitable existing project, create a small Python package called `aidet_eval`. A Python/Hugging Face/Parquet implementation with a lightweight local web viewer is suitable. Pin a working dependency environment; do not assume older dataset-loading scripts work with the installed `datasets` version. Use configuration for new datasets and models, not a separate evaluation script per model.

Do not train or fine-tune the baselines. Do not generate replacement evaluation data, rent compute, call paid APIs, upload source texts, or publish datasets/results externally. Use available local hardware and existing credentials. Do not expose tokens in logs. Do not accept new gated contact-sharing or licence agreements on the user's behalf. Record the exact necessary action when access is blocked, then continue independent work.

Complete these deliverables:

1. Resumable acquisition and inventory for all 16 catalogue entries, with an explicit status for each.
2. Dataset-specific normalisation adapters, provenance auditing, overlap checks, and a frozen, source-disjoint benchmark manifest.
3. Correct inference adapters for both specified Open Pangram models, plus a small documented interface for adding my future models.
4. Actual baseline runs where access and hardware permit; frozen calibration thresholds; reusable per-example predictions.
5. `results.full.json` and `results.summary.json`, with versioned schemas and the semantics in `RESULTS_CONTRACT.md`.
6. A working local viewer for summaries, comparisons, errors, and full-text/span inspection.
7. Tests, a benchmark card, exact reproduction commands, an execution log, and a final status report distinguishing completed, partial, and blocked work.

A gated model or inaccessible dataset is not a reason to stop the whole project. It is also not permission to substitute another model silently or invent results. If a baseline cannot run, its output status must say so. A synthetic fixture proves software operation, not baseline performance.

Implement a small end-to-end vertical slice first: one real human source, one real AI/mixed source, one available baseline, export, viewer. Then expand to the full catalogue and both baselines. Do not spend the entire task designing abstractions before trying real inference.

## 2. Acquire and inventory all sources

Treat the attached catalogue as an executable acquisition registry. Resolve current canonical upstream locations, and pin immutable revisions or raw-file hashes. Record any repository migration rather than silently changing identities. Use author-maintained releases, not arbitrary mirrors or detector-curated reposts.

For every source, save a machine-readable inventory entry with its upstream URL/repository, resolved revision, retrieval time, file list, hashes, download completeness, configurations/splits, schema, language counts, label counts, domains, generators, mechanism counts where available, grouping keys, span encoding, provenance evidence, applicable terms, and eligibility decisions. Separate reported counts from measured counts. A field that does not exist is unknown, not something to guess from row order.

Download the relevant original release files for all catalogue entries when access and disk space permit. Inventory expected sizes and free space first. Do not clone model histories or download several equivalent mirrors/conversions unnecessarily. Downloads should be cached, resumable, and checksum-checked. Preserve raw data read-only; derive normalised Parquet separately. Do not fetch whole scholarly PDFs when the official text corpus suffices.

If a source exceeds available storage, is gated, disappears, or has unclear acquisition rights, record `partial` or `blocked` with the exact issue, completed files, remaining files, and a reproducible continuation command. Do not describe a streamed sample as a full download. Use full-population deterministic/reservoir sampling rather than taking the first N rows. Never silently omit a difficult source from the inventory.

### Source-specific ingestion requirements

| Source | Required treatment |
|---|---|
| `iitolstykh/LLMTrace_detection` | English-first; retain `data_type`, `model`, `prompt_type`, `topic_id`, `label`, and `ai_char_intervals`. Prioritise `fill_gaps` for the primary replacement panel. Verify interval units/conventions. Other mechanisms remain separate. Audit human provenance. |
| `iitolstykh/LLMTrace_classification` | Domain/generator-stratified pure-human and pure-AI candidates. Cross-link related topics and duplicates with the detection release. Do not count the releases as automatically independent. |
| `liamdugan/raid` | `generation` is text; preserve `source_id`, `domain`, `model`, and `attack`. Verify the human-label rule against the release. Primary material is clean English prose (`attack == 'none'`). Publicly unlabelled test rows cannot be scored as if labels were available. Labelled training-release rows must be described honestly as such. Attacks are reserve/robustness material. |
| `saiteja33/DAMASHA` | Inspect individual CSV/file schemas rather than assuming one homogeneous default loader. Reconstruct offsets while removing annotation tags such as `<AI_Start>` and `<AI_End>`. Establish which files/rows are newly constructed sentence replacements versus continuation, reused corpora, or perturbations. Do not infer these categories without evidence. |
| `UKPLab/AITDNA` | Reconstruct/choose a single appropriate representation of each distinct writing session/text. The catalogue reports 362 distinct texts, not 3,258 independent examples. Verify the actual inventory. Do not use a default 50%-AI document threshold for an any-AI-involvement task. Preserve logged provenance and its limitations. |
| `toloka/beemo` | `human_output` is the human reference, `model_output` the AI response, and `human_edits` an expert's edited AI response—not a human-only control. Group matched versions by verified prompt/source identifiers. Deduplicate repeated human references and overlap with No Robots. No invented exact AI-origin spans. |
| `stanfordnlp/imdb` | Historical review candidates. Sentiment labels are not authorship labels. Retain source identity and handle HTML deterministically. |
| `EdinburghNLP/xsum` | Use original `document` prose, not generated summaries. Preserve article `id`; audit overlap with detection datasets and baseline training sources. |
| `armanc/scientific_papers` | Acquire the official arXiv and PubMed text releases or a verified faithful conversion. Preserve article identity, distinguish abstracts from body passages, and audit extraction/math-placeholder artefacts. A random unverified mirror is not an acceptable workaround for a legacy loader. |
| `Salesforce/wikitext` | Prefer `wikitext-103-raw-v1`; reconstruct article boundaries. Dataset rows are not automatically complete or independent documents. |
| `deepmind/pg19` | Historical literature panel. Group by book and, where known, author. Extract reproducible contiguous passages; do not evaluate entire books as ordinary short documents. Preserve publication/source evidence and note release transformations. |
| PERSUADE 2.0 | Acquire the author-maintained original release linked in the catalogue, not an AI-augmented competition amalgamation. Preserve writer IDs where available. Do not invent collection dates. Student proficiency metadata may define diagnostic slices; do not infer sensitive traits from writing. |
| `HuggingFaceH4/no_robots` | Extract original English human assistant responses, not prompts, formatting wrappers, or translated derivatives. Preserve its curator-attested provenance category. Inspect the actual identifiers rather than assuming every release has `prompt_id`. Cross-link Beemo. |
| `yaful/MAGE` | Reserve/in-distribution regression source until actual training exposure is established. Verify the known convention `1 = human`, `0 = AI`; do not apply the opposite generic mapping. |
| `pangram/editlens_iclr` | Attempt authorised access; retain as an in-family/reference panel, not independent evidence for EditLens-trained baselines. AI editing is not equivalent to causal block-replacement spans. |
| CoAuthor | Attempt acquisition of the original logged sessions. Replay edits/accepted suggestions if final provenance is needed. Keep sessions/writers grouped and older-generator collaboration separate. Reserve unless reconstruction and rights checks pass. |

These are catalogue leads, not permission to skip verification. Correct discrepancies in a new audit file while retaining the original catalogue unchanged. Do not enlarge the project with additional datasets merely to fill a cell.

## 3. Canonical records, provenance, and contamination controls

Create one canonical schema covering all sources. Keep original source labels and their interpretation separately. At minimum retain:

- Stable `example_id`, source dataset/revision/config/split/row ID, parent document IDs, and canonical text SHA-256.
- `source_group_id`, cross-dataset `leakage_group_id`, and an `independence_group_id` (writer/session/book/source as evidence permits). Keep paired variants linked.
- `domain`, `subdomain`, `language`, generator and generation settings when known, and dataset-native metadata.
- `authorship` in `human|ai|mixed|unknown`, `authorship_basis`, `ai_involvement` in `yes|no|unknown`, and `mix_mechanism`.
- Human provenance category and supporting evidence; verified source/publication/collection dates separately from download dates. Preserve a confidence/eligibility assessment without claiming universal proof of human authorship.
- Canonical `text`; parent/excerpt offsets; recorded deterministic transformations; original and canonical hashes.
- AI and human character intervals, annotation completeness, annotation provenance, and derived AI fraction where justified.
- Panel memberships, calibration/evaluation role, licence/redistribution restrictions, exclusion or quarantine reasons.

`mixed` includes documented replacement, continuation, and collaboration processes, but their mechanisms and label bases must remain distinct. Human-edited AI and AI-edited human prose may be labelled process-level mixed when supported. Do not imply that a process label determines which surviving characters were authored by whom. Keep `unknown` where the evidence cannot support a more specific class.

### Span convention

Use zero-based, half-open Unicode code-point intervals `[start, end)` into the exact stored canonical text. They are not UTF-8 byte offsets or JavaScript UTF-16 indices. The viewer must handle that distinction, including emoji and combining characters. Reject invalid Unicode and invalid/reversed/out-of-range intervals.

`ai_char_spans: null` means unavailable; `[]` means an exhaustively established empty AI span set, not missing labels. Store annotation completeness explicitly. The complement of AI spans is human **only** when the annotation is exhaustive and supports that conclusion. Preserve uncertain/unlabelled regions otherwise.

For complete span annotations, `ai_fraction_char` is the length of the union of AI intervals divided by canonical text length, including whitespace. Leave it null when not identifiable. Distinguish insertion provenance, edit-log lineage, alignment-inferred spans, and edit-distance proxies. Do not promote diff-derived or process-level labels to exact causal span gold.

Do not lowercase, rewrite, grammar-correct, or LLM-clean the canonical benchmark. Any deterministic HTML/line-ending/annotation removal must preserve a documented offset mapping or require revalidation. Pangram-specific preprocessing belongs in the model adapter, not in the source text. Quarantine malformed examples rather than guessing repairs.

### Independence and overlap

Build overlap/leakage groups before selecting splits. Link variants of the same source and exact duplicates across datasets. Add a documented near-duplicate check using normalised text fingerprints or shingles; retain evidence, thresholds, and collision statistics. Matching normalisation must not alter the text actually evaluated.

Do not let one paper's abstract/body, one book's excerpts, one prompt's human/AI variants, or one co-writing session cross calibration and evaluation. Preserve official splits where practical, but enforce stronger cross-source grouping. Use verified writer/author identifiers when feasible; do not group all essays merely because they answer one of a few common prompts. Keep independently meaningful source, leakage, and uncertainty-clustering identifiers distinct.

Primary human sampling should favour one item per source and, where feasible, one per writer. Deduplicate repeated Beemo human references. A text may belong to multiple diagnostic panels, but a pooled denominator must not count it repeatedly.

Accept an optional registry of training-text hashes/source IDs for my future models. Mark known overlaps and exclude them from the independent evaluation view. For Pangram, inspect available EditLens training provenance and record possible XSum/related source overlap. Use exposure statuses such as `known_overlap`, `possible_overlap`, `unknown`, and `no_detected_overlap`; the last is not proof of no pretraining exposure.

Never use any detector's score to decide whether a candidate human example is human or to remove difficult cases. Do not use model scores, model performance, or test labels to choose favourable subsets after the manifest is frozen.

## 4. Choose and freeze informative subsets

Use the following planning budget, subject to actual eligible availability. The YAML supplies the same starting targets. Report requested, eligible, selected, and missing counts for every cell; do not manufacture balance.

| Panel | Planning target | Purpose |
|---|---:|---|
| High-confidence human controls | 5,000: about 1,000 each for research, news, reviews, expository/encyclopaedic prose, creative writing | Per-domain false positives; historical and contemporary provenance remain distinguishable. |
| Student human writing | 1,000 | Dedicated essay/proficiency diagnostic. |
| Fully AI-generated | 1,500: about 300 per core domain | Recall by domain, generator, and generation task. |
| Controlled replacement mixtures | 1,000: about 200 per core domain | Detect inserted AI without conflating continuations or whole-document editing. |
| AITDNA | All eligible unique final texts; catalogue expectation 362 | Real co-writing diagnostic, including its human controls. |
| Beemo matched editing | Up to 200 complete human/AI/expert-edited triplets, about 600 inputs before deduplication | Human editing of AI output. |
| Modern human instruction responses | Up to 500 eligible No Robots responses | Modern curator-attested human writing, separate from historical prose. |

The nominal total is 9,962 inputs, not a verified count of independent texts. Deduplication and eligibility will change it. Permit a modest separate source-matched human-reference panel when required to interpret AI/mixture examples; report its additional count explicitly rather than replacing trusted controls with unverified human references. Keep the initial scored suite around 8,000–12,000 unique inputs if practical, rather than scaling to the size of the downloaded corpora.

Maintain two human-control views: independently audited human controls, and benchmark-provided/source-matched human references. A matched reference may enter the trusted view only if its provenance passes that audit. This avoids either treating every benchmark label as verified or comparing only unrelated historical humans with contemporary AI.

Balance generators and tasks within domains where feasible. Report unsupported/missing cells rather than borrowing unrelated material. Preserve research abstracts versus body prose, news versus reviews, historical versus modern writing, and dataset-native categories. Do not map a broad `article` category to research without source evidence. Keep recipes, poetry, forum posts, and other reserve categories visible in the inventory even when not primary panels.

For replacement mixtures, stratify by source, generator, AI-character fraction, number of inserted blocks, block length, and position. Proposed AI-fraction bins are `(0, .05]`, `(.05, .10]`, `(.10, .25]`, `(.25, .50]`, and `(.50, 1)`, plus `unknown`; pure-AI texts are not the final mixed bin. Report sparsity. Block-level replacement is the primary mixed test; continuation, AI editing, human editing, and interactive co-writing remain separately named panels.

Use reproducible source-group sampling and seed 42 by default. Save chosen IDs, ranks, selection probabilities/weights if used, and exclusions. Do not refill excluded rows with whichever examples a detector happens to classify successfully. Keep dataset/source labels available for every domain comparison so corpus-specific shortcuts remain visible.

For lengthy human documents, create deterministic contiguous excerpts with parent offsets. Do not create thousands of pseudo-independent samples from a few books. For any mixed excerpt, remap spans and rederive its actual label/fraction: a crop that removes all AI text is not still a positive merely because its parent was mixed. Prefer retaining complete bounded-length mixed documents. Define length strata using a documented word-count function, not one model's tokenizer. Keep very short and long-text conditions explicit; do not silently drop texts because the upstream baseline trained on a minimum length.

### Calibration and benchmark versions

Reserve approximately 3,000–3,500 **additional** human calibration examples: around 500 per core domain plus students, and modern instruction responses where available. They must be leakage-group-disjoint from all evaluation panels. Keep the calibration distribution and its domain weights explicit. Low-provenance references must not silently replace trusted calibration controls.

Use the same calibration IDs for every model. Do not tune thresholds on evaluation labels. Missing calibration domains are recorded; thresholds describe the distribution actually calibrated, not a guarantee across all writing.

Freeze the normalised examples, selection manifest, overlap groups, panel definitions, calibration IDs, text/span transformations, and length/window policy before looking at baseline evaluation scores. Hash them into a benchmark version. Subsequent reruns use the frozen manifest rather than resampling each time. Any replacement, new domain, new generator, changed gold label, or changed sampling rule creates a new version.

This is initially a development/regression benchmark because I will run it repeatedly while improving models. Support a separately frozen confirmation release later, but do not secretly count inaccessible or never-scored examples as completed evaluation. Label the role of each release honestly.

## 5. Integrate the exact two Open Pangram baselines

The required checkpoints are:

| Local ID | Checkpoint | Base/tokenizer | Reference maximum input length |
|---|---|---|---:|
| `open_pangram_roberta` | `pangram/editlens_roberta-large` | `FacebookAI/roberta-large` | 512 tokens |
| `open_pangram_llama` | `pangram/editlens_Llama-3.2-3B` | `meta-llama/Llama-3.2-3B` | 1,024 tokens |

Authoritative starting implementation: `https://github.com/pangramlabs/EditLens`, especially `scripts/inference.py`, `scripts/train.py`, `scripts/preprocess.py`, and `configs/{roberta,llama}.yaml`. Source URLs and inspected file hashes are in `verified_baselines.json`. Resolve and pin actual checkpoint, base-model, tokenizer, and code revisions during execution. Public model-card metadata is not proof the gated weights have downloaded.

Both model pages currently require access-condition acceptance and list CC-BY-NC-SA-4.0. The Llama base has its own access/terms to check. Use existing authorised access. Distinguish authentication failure from missing files; do not classify a denied adapter-config request as evidence that a checkpoint is a full model. Do not confuse these open checkpoints with the commercial Pangram service.

### Critical inference semantics

The upstream detector outputs an ordinal bucket distribution, not an ordinary binary probability. Its scalar score is:

`score = sum(probability[k] * k for k in range(K)) / (K - 1)`

where probabilities come from a softmax over the K bucket logits. The inspected training configs specify four buckets, but infer and verify K from each downloaded checkpoint. Retain bucket logits, probabilities, predicted bucket, and scalar score. The scalar is an **AI-intervention/extent signal**, not a calibrated probability that AI is present and not a ground-truth fraction of AI-written characters. Do not label it `p_ai`, compute probability calibration error from it, or equate it to the insertion fraction.

The Llama release is an adapter-based sequence classifier. The reference loader replaces the base `score` layer with **LayerNorm followed by a bias-free Linear layer** (`NormedLinear`) before loading the saved adapter/head. Implement this faithfully. Check that the saved head and adapter parameters really load; a generic causal-language-model pipeline or an accidentally random classification head is not the baseline. The RoBERTa loader should follow the actual full-checkpoint structure rather than assuming the Hugging Face page's automatic PEFT example is correct.

The inspected reference Llama path uses NF4 4-bit loading with bfloat16 compute. Prefer that supported configuration on suitable local hardware; record quantisation, dtype, device, library versions, and any deliberate alternative. A changed precision/loader is a named model variant, not an undocumented fallback. Prefer the user's RTX 4080 if it is present, but inspect actual hardware rather than assuming it is available. Batch adaptively and resume after out-of-memory failures without changing the input protocol. Never truncate more text to conceal an OOM.

Reference preprocessing currently demojizes, removes a `</think>`-preceding section, may remove an opening paragraph beginning with certain boilerplate phrases (including `Abstract` or `Title`), lowercases, and collapses whitespace. Reproduce and version it inside the adapter. Record transformations and effective input text/hashes; deletion can affect scientific-text and span interpretation. Do not silently substitute raw text or clean the canonical benchmark globally. A raw-input ablation is optional and must have a different protocol ID.

Before the full run, compare the adapter against the pinned upstream implementation on a small fixed set including short/long texts, paragraph headers, emoji, and mixed prose. Compare logits/scores under the same dtype and quantisation, document numeric tolerances, and test single-item versus batched inference. This is implementation parity, not threshold tuning. Avoid importing training scripts in a way that starts training, telemetry, or heavyweight unrelated dependencies.

### Extension interface

The detector interface receives only stable IDs and text/windows—not labels, provenance, gold spans, or source-specific hints. Return raw scores with their semantics, optional class probabilities, optional native span predictions, input coverage, and timing/error information. Adding my next model should require a small adapter and configuration, then the same `predict`, `calibrate`, `evaluate`, and viewer paths. Do not impose a generative/chat interface on a sequence classifier or train a new encoder control as part of this task.

## 6. Text coverage and inference protocols

A detector missing AI because it never read the inserted passage is not the same diagnosis as reading and misclassifying it. Make coverage inspectable.

Implement two distinctly named protocols:

**`shared_windows_v1` — primary practical comparison.** Create a deterministic, immutable window manifest on canonical text, shared by both baselines. Start with approximately 192 whitespace-delimited words and 64 words of overlap. Shorten/split windows as necessary until, after the adapter's reference preprocessing and special tokens, every window fits both baselines' supported budgets. Handle single overlong words with an explicit code-point split and ensure forward progress. Save the exact canonical window bounds, tokenizer revisions, preprocessing fingerprint, and window IDs. Never split by gold labels or position windows around known AI blocks. Each model sees identical canonical windows; no hidden tokenizer truncation is allowed in this protocol.

Use `max(window_score)` as the predeclared primary document score for detecting any AI involvement. This aggregation is a harness wrapper, not a claim about upstream Pangram's native document behaviour. Preserve all window scores. Document length and number of windows can affect the maximum, so report FPR/recall by length and window count. Calibrate the aggregated document scores—not the individual window scores. Do not choose max/mean/top-k after seeing which wins on evaluation data.

The frozen window manifest remains fixed for future models. A new model unable to consume it is protocol-incompatible until a separately versioned policy is defined; do not change old baselines' inputs silently. Larger-context models can still run this common-window comparison. Keep the benchmark selection hash distinct from the inference-protocol/window hash.

**`native_truncate_v1` — secondary reference behaviour.** Apply the official preprocessing to the full canonical input and use the appropriate 512/1,024-token truncation limit. Reproduce upstream inference rather than calling this complete-document coverage. Record retained input ranges or a trustworthy coverage status, truncation counts, and whether annotated AI was outside the observed part when this can be established. Thresholds are separately calibrated for this protocol.

Preserve mapping between canonical text, cleaned text, and tokenizer offsets when possible. Distinguish canonical window coverage from effective model-visible coverage after deletions. If exact mapping cannot be reconstructed, coverage is null/unknown with a reason, not an invented percentage. Documents that become empty after preprocessing must have an explicit failure/abstention status rather than a default human score.

Both baselines are document/window scorers. Do **not** present their window scores as native token-level classifications. In the primary report, native localisation metrics are unavailable for them. The viewer may show scored windows as contextual diagnostics, clearly labelled as such. A separately named sliding-window localisation approximation is optional, but must never be merged with native span metrics.

## 7. Calibration and metrics

For each model and each inference protocol, freeze two operating points using only the selected human calibration documents: target empirical FPR **1%** and **5%**. The 1% point is primary. These targets are not guaranteed test FPRs.

Make threshold ties deterministic. Let `n` be the number of eligible, successfully scored human calibration examples and `k = floor(alpha * n)`. Sort scores descending and set `threshold` to the score at zero-based index `k`, with decision rule **`score > threshold`**. This gives at most k calibration false positives, including tied scores. Do not round thresholds before applying them or interpolate using test data. Record target FPR, actual calibration FPR, exact threshold, strict comparator, tied-score count, IDs/hash, exclusions, and uncertainty. No eligible calibration data means a blocked operating point, not an automatic threshold of 0.5.

Require calibration coverage to be complete for an official run, or explicitly mark any result provisional and explain the failures. Do not optimise F1 or choose per-domain thresholds on the evaluation set. A test-set ROC curve is a descriptive diagnostic, not permission to relabel a test-optimised threshold as calibrated deployment performance.

### Required scorecard

At both frozen operating points, report:

- Human false-positive rate and counts by domain, dataset, provenance category, and length. Keep independently audited humans separate from benchmark-provided references.
- Fully-AI recall by domain, source, generator, generation task, and length.
- Mixed-document recall by mechanism, domain, generator, AI-fraction bin, inserted-block length/count/position, and length where metadata exists.
- AITDNA co-writing and Beemo edited-AI results as distinct panels. Preserve human controls and matched comparisons.
- Prediction coverage, errors/abstentions, and runtime/throughput for each model/protocol/panel.

Also provide binary confusion matrices with an explicit class mapping: (a) human versus fully AI, excluding mixed, and (b) human versus documented AI involvement, with mixed mechanisms visible. Do not force the bucket distribution into three semantic human/AI/mixed probabilities. Three-way classification is optional only with an explicit, separately calibrated mapping and suitable calibration labels.

Compute AUROC and average precision on properly specified binary slices containing both classes. Include class counts and prevalence; AP and precision depend on the sampled class balance. No Brier/ECE for the raw ordinal extent score. A separately trained probability calibrator would need additional disjoint calibration data and is outside the required first release.

For a future detector with native spans and sufficiently complete gold, support character-level precision/recall/F1 and human-text false highlighting, micro and document-macro. Use known, non-whitespace canonical code-point positions; do not count unknown annotations as human. Partial-gold regions and unsupported predictions are excluded with explicit coverage. Retain counts needed to reproduce every value. Gold AI spans must not be used to select model input windows.

### Denominators, uncertainty, and failures

Every rate must include its numerator, denominator, eligible/selected/successfully scored counts, number of independent groups, and applicability/status. Absent data or unsupported metrics produce null plus a reason, not zero or NaN. Do not let inference errors disappear from the scorecard. Report conditional performance on successful predictions and simple best/worst-case bounds over eligible failed predictions where applicable.

Use paired source/independence-group bootstrap intervals for model differences and clustered samples; resample matched versions together, with a fixed seed and a documented number of replicates (e.g. 2,000). For single, plausibly independent source observations, provide Wilson intervals with their independence assumption recorded. Zero-error bootstrap intervals can be misleadingly degenerate: flag this, and do not advertise 0±0 as proof of perfect performance. A separately labelled group-level any-false-positive rate can use an independent-group binomial interval; a zero-event one-sided 95% upper bound is `1 - 0.05**(1/n_groups)`. Do not mislabel that group-level bound as an exact document-level bound for unequal clusters.

Do not produce a single official overall quality number. Optional macro summaries must specify the included cells and equal-domain or other weights, show coverage, and refuse silent renormalisation over missing cells. Micro results must deduplicate overlapping panel memberships. The main comparison is domain-specific detection versus domain-specific human false positives at the frozen operating points.

## 8. Results, viewer, and completion

Implement the detailed/summarised JSON contract in `RESULTS_CONTRACT.md`. Use one strict set of typed models and generated/versioned JSON Schemas. The summary must be a deterministic reduction of the detailed result, not a second independent evaluation implementation.

Use append-only JSONL or Parquet prediction shards internally for resumability. Still produce the requested **one large standalone `results.full.json`**, containing examples once and predictions linked by ID. An additional `.json.gz` copy is useful. A directory of prediction fragments alone does not satisfy this deliverable.

### Viewer

Build a local viewer that opens one or multiple run directories or result files. Prefer a small existing frontend; otherwise a local Streamlit app or a minimal local web application is sufficient. No hosted account, external database, analytics, or external text upload is needed.

Provide these screens:

1. **Coverage and scorecard:** model/protocol selectors, 1%/5% calibration points, human FPR and pure/mixed recall by domain/panel, percentages with counts and intervals, and prominent unavailable/provisional warnings.
2. **Comparison:** side-by-side runs/models, absolute and percentage-point differences, paired common-example comparisons, and incompatibility warnings for changed benchmark/calibration/protocol definitions. Do not compare unlike aggregates without showing that mismatch.
3. **Error browser:** filter by false positive, false negative, disagreement, dataset/domain/provenance/generator/mechanism/fraction/length; show canonical text, source/group IDs, raw scores, decisions, and gold spans. Highlight human false accusations prominently. Do not leak labels into inference.
4. **Coverage/span inspection:** toggle gold annotations, native predicted spans if available, and separately labelled scored windows; show unread/unknown regions. Use correct code-point indexing in browser code. Treat dataset text as untrusted text, never executable HTML.
5. **Run provenance:** source revisions, model/loader revisions, calibration membership and thresholds, overlap warnings, acquisition status, exclusions, and limitations.

Optional score histograms/ROC curves and a threshold slider are useful. Slider changes must be labelled **exploratory**, must not overwrite official thresholds or saved summary results, and must show the changed FPR. Do not present a displayed editing score of 0.8 as “80% probability of AI”.

The summary file alone should render the scorecard. Full-text inspection becomes available when the full file is loaded. Load large detail files lazily or cache/index them; do not freeze the browser by rendering every document. Support summary tables as CSV and a simple Markdown report generated from the same summary JSON. Provide a local export mode with texts redacted and restrictive redistribution warnings; do not publish copyrighted/student texts by default.

### CLI and tests

Provide commands equivalent to:

```bash
python -m aidet_eval doctor
python -m aidet_eval download --catalogue ai_detector_dataset_catalogue.json --all --resume
python -m aidet_eval inventory
python -m aidet_eval build --config benchmark_plan.yaml
python -m aidet_eval validate --benchmark artifacts/benchmark-v1
python -m aidet_eval predict --benchmark artifacts/benchmark-v1 --models open_pangram_roberta,open_pangram_llama --protocols shared_windows_v1,native_truncate_v1 --resume
python -m aidet_eval calibrate --benchmark artifacts/benchmark-v1
python -m aidet_eval evaluate --benchmark artifacts/benchmark-v1
python -m aidet_eval export --run artifacts/runs/<run_id>
python -m aidet_eval view --runs artifacts/runs
```

Adapt names to the repository, but document real runnable commands, including environment setup. Include a smoke profile separate from the full profile. A resumed run must reuse the same manifest and predictions; never regenerate a changed sample under an old run ID.

Test at least label mappings (especially MAGE), AI-tag removal/offset reconstruction, Unicode spans, cropped labels, grouping/deduplication, no calibration/test overlap, deterministic sampling/windowing, preprocessing coverage, threshold ties, all-equal scores, zero/one-class/empty slices, undefined metric denominators, failure accounting, native-span unavailability, full-to-summary reproducibility, cross-run compatibility, and viewer loading/escaping. Test checkpoint/loader parity when authorised weights are available. Tests must run without downloading the full corpora or requiring GPU access; small synthetic fixtures must be unmistakably separate from actual results.

### Final execution report

Return the code changes and exact locations of cached data, frozen benchmark, model snapshots, full/summary JSON, viewer, and tests. State how many unique documents and independence groups were actually evaluated for each model and protocol. List all 16 acquisition statuses. Summarise actual per-domain results only if inference was run. Describe shortfalls, provenance ambiguities, overlap limitations, and any necessary gated-access actions explicitly.

Completion means functioning code and genuinely executed available evaluations—not copied paper results, seeded scores, placeholder percentages, or a promise to run later. If something is blocked, the completed independent parts and machine-readable blocked status must still be delivered.
