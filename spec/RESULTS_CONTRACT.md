# Unified evaluation result contract — version 1.0

This is a design contract for the coding agent to implement. It is **not a file of measured evaluation results**. Implement strict typed models, a semantic validator, and JSON Schemas for the full and summary documents. Preserve these semantics even if existing repository conventions justify different internal classes.

## 1. Files and invariants

Each completed or partial multi-model evaluation run exports:

- `results.full.json`: one standalone JSON object, with complete canonical examples, predictions, calibration records, aggregates, and provenance. Store example text once and reference its ID elsewhere.
- `results.summary.json`: compact report-generation data, deterministically derived from the full file. No source texts or per-window prediction arrays.
- `results.full.json.gz`: optional compressed equivalent, in addition to the uncompressed JSON.
- `results.full.schema.json`, `results.summary.schema.json`: versioned machine-readable schemas; shared `$defs` are acceptable.

Internally, resumable JSONL/Parquet shards are encouraged. They are not a replacement for the requested full JSON. Never populate result files with synthetic fixture scores or published-paper scores. Fixture/demo files must have clearly different names, IDs, and `data_origin: synthetic_fixture`.

UTF-8 JSON must contain finite numbers only. Unknown/undefined/not-applicable is `null` with a machine-readable reason; never NaN, Infinity, an empty string pretending to be a number, or a numerical zero standing for missing. Count fields are integers. All timestamps include a timezone. Secrets and authentication query strings must not appear anywhere.

Every document has `schema_version`, `artifact_kind`, `data_origin`, `run_id`, and a `status` such as `complete`, `partial`, `blocked`, or `failed`. A smoke run is a distinct benchmark/run profile, not a full evaluation with mostly missing rows. A partial run does not acquire `complete` status merely because all available predictions have been exported.

Percentages are display representations of mathematical rates. A human FPR of 0.013 is **1.3%**, not 0.013%. Keep raw precision; round for display only. Any rounded serialized display field must follow one documented rule and never feed subsequent calculations.

## 2. Full result: top-level structure

Required top-level fields:

```text
schema_version, artifact_kind, data_origin, run_id, status
benchmark
run
sources[]
models[]
protocols[]
calibrations[]
examples[]
predictions[]
aggregates[]
comparisons[]
errors[]
validation
warnings[]
extensions
```

Use an explicit `extensions` object rather than silently adding ambiguous top-level keys. Version incompatible semantic changes, not just syntactic changes.

### `benchmark`

Include benchmark ID/version/profile, purpose (`development_regression` or separately identified confirmation), seed, manifest SHA-256, text/gold/selection fingerprint, calibration-membership hash, panel definitions, task definitions, selection targets and realised counts, domain mappings, provenance/eligibility rules, split/grouping policy, excerpt policy, and canonical-offset convention.

Panel definitions specify example memberships or reproducible filters, target labels, denominators, and whether controls are independently audited or benchmark-provided. An example can have several memberships; the membership union must be deduplicated when used as a pooled population. Include shortfalls and exclusions by reason. Source inventories can be compact summaries with hashes/paths to detailed acquisition audits; do not embed the entire raw downloaded corpora.

### `run`

Include creation/start/finish times, command and resolved configuration, code commit and dirty-tree indicator/hash, dependency-lock hash, relevant package versions, device/hardware, operating system, reproducibility settings, bootstrap seed/replicates, requested models/protocols/panels, and execution counts. Log whether inference and export really completed. Store test-suite and loader-parity outcomes with their scope, not a blanket success claim.

### `sources[]`

One entry per catalogue source, including reserve and blocked sources. Include source ID/name, authoritative location, pinned revision or file hashes, acquisition status, download completeness, observed schema/counts, eligibility status, provenance assessment, rights restrictions, and warning/error references. Separate `reported_rows` from `observed_rows`.

### `models[]`

Include model ID/display name, checkpoint/base/tokenizer IDs and revisions, file hashes when available, adapter version, loading/head configuration, precision and quantisation, device, reference implementation revision, preprocessing fingerprint, score semantics/range/orientation, output capabilities, training-exposure audit, access/run status, and a model-spec fingerprint.

Both specified Pangram models have an ordinal extent score; `score_semantics` must not say calibrated probability. Preserve that fact in summary metadata and the viewer. Missing model access is a model-level status plus properly accounted-for unavailable predictions, not a silently omitted model.

### `protocols[]`

Include protocol ID/version/fingerprint, upstream-native versus harness-wrapper status, canonical window-manifest hash if applicable, window construction parameters, special-token budgeting, preprocessing policy, maximum lengths, aggregation, truncation policy, coverage accounting, and localisation mode (`native`, `approximate_windows`, or `unsupported`).

Native and shared-window protocols must not be merged into one unlabeled result. Changing aggregation/preprocessing requires a different protocol fingerprint. Precision changes belong in model metadata/fingerprint, not an invisible cache implementation detail.

## 3. Example records

Each `examples[]` entry contains:

```text
example_id
source: dataset_id, revision, config, original_split, original_row_id,
        parent_ids, source_urls
source_group_id, leakage_group_id, independence_group_id
role: calibration | evaluation
panel_ids[]
domain, subdomain, language
original_labels
truth: authorship, authorship_basis, ai_involvement, mix_mechanism,
       generator, generation_parameters,
       ai_char_spans, human_char_spans,
       annotation_completeness, annotation_provenance,
       ai_fraction_char, span_count, span_lengths
human_provenance: category, evidence, verified_dates, eligibility, caveats
text, text_sha256, word_count, char_count
parent_excerpt, canonicalisation
rights
extensions
```

Use null for missing metadata. Preserve dataset-native values in `original_labels` or `extensions`; don't destroy evidence in a many-to-one mapping.

The canonical offset convention is Unicode code points, zero-based, half-open `[start,end)`. `text[a:b]` in Python must recover exactly the annotated substring. The JavaScript viewer must not assume string indices are equivalent. `null` spans are unknown; empty spans are genuinely empty only under a complete annotation. Record uncertain regions/partial coverage rather than assuming all unmarked text is human. Fraction and span counts are derived from validated interval unions, not from a detector's output.

Authorship/process labels and exact surface-origin annotations are distinct. For Beemo expert-edited AI, process-level AI involvement can be known while exact surviving AI spans are unknown. Neither a text diff nor an editing-strength score supplies causal span labels by itself.

Store calibration text and its predictions in the full file as well as evaluation text, so threshold reconstruction does not require an external dataset download. The benchmark remains local by default; redacted exports must record which detail capabilities are unavailable.

## 4. Prediction records and execution completeness

A prediction is identified by `(example_id, model_id, protocol_id)`, with an optional explicit variant ID where needed. No duplicate keys. Include:

```text
prediction_id, example_id, model_id, protocol_id
status: ok | error | skipped | blocked
score                       # finite oriented score, or null
score_semantics
raw_outputs                 # logits, bucket probabilities, bucket prediction, etc.
windows[]                   # required for windowed inference
native_span_predictions     # null if unsupported
coverage
operating_point_decisions[]
timing
cache_key, input_fingerprint
error_id, warnings[]
```

A successful Pangram prediction preserves the scalar, K logits/probabilities, and bucket semantics. For a windowed aggregate, preserve these raw outputs per window; do not invent a document-level probability distribution by pretending the maximum is a softmax.

Each window record includes stable window ID, canonical bounds, exact model-visible input or a reproducible transform plus hash, tokens/length before and after processing, raw outputs, status, and coverage/mapping information. Avoid duplicating large repeated strings unnecessarily: reusable input records referenced by ID are acceptable if contained in the same full JSON. A mapping can be null with a reason; coverage must not be fabricated.

`coverage` distinguishes canonical windows covering the document from the actual cleaned/truncated regions consumed. Record truncation and preprocessing deletion separately. Gold-positive coverage can be derived only when mappings/annotations justify it. Keeping unknown coverage explicit is preferable to an inaccurate fraction.

`operating_point_decisions` reference calibration/operating-point IDs and hold `flag_ai: true|false|null`. An unavailable score or threshold yields null, not false. Apply the recorded raw threshold with its recorded comparator. No rounded display value is authoritative.

Represent every expected example/model/protocol combination, including blocked/error records, or provide a compact explicit unavailable-range representation that the validator expands exactly. Do not lose failure denominators. Do not report an aggregation as complete when one of its required windows failed; store it as partial/error and optionally expose a separately labelled lower-bound score. Cache reuse must depend on the text/input fingerprint, model revision/precision/adapter, and protocol; threshold changes may reuse raw predictions but must regenerate decisions and metrics.

## 5. Calibration records

A calibration record contains model/protocol/operating-point IDs, task, target FPR as a fraction, comparator, threshold, calibration example IDs/hash, eligible/scored counts, actual false positives, actual FPR, tied-score statistics, calibration distribution/weights, selection algorithm/version, status, and reasons for exclusions or provisional status.

The default algorithm is the descending-order statistic described in `HANDOFF.md`, with strict `score > threshold`. Store enough information to reproduce that threshold from the prediction records. No test-derived threshold is allowed to use an official calibration operating-point ID. Keep any exploratory/native upstream threshold under a separate ID.

## 6. Aggregates and metric values

Use stable, flat slice records instead of a deeply nested, model-specific dictionary. Each aggregate/cell has:

```text
cell_id
model_id, protocol_id, operating_point_id or null
task_id, panel_id, split
slice: domain, subdomain, dataset_id, provenance_category,
       authorship, mix_mechanism, generator, length_bin,
       ai_fraction_bin, block_length_bin, ...
population_definition, example_ids_hash
support
metrics
confusion_matrix or null
warnings[]
```

Only materialise useful predeclared slices, not the full Cartesian product of every metadata field. Distinguish an explicitly requested but empty cell from a slice that was never requested. `all` versus `unknown` versus null filter values must have documented meanings.

`support` includes selected/eligible/scored/error/skipped/blocked counts, human/AI/mixed/unknown counts, number of source/leakage/independence groups, and the class prevalence applicable to a metric. Also expose per-metric support when it differs from the cell's population.

A metric object has these fields:

```json
{
  "status": "not_computed",
  "value": null,
  "unit": "proportion",
  "percent": null,
  "numerator": null,
  "denominator": null,
  "denominator_population": "valid_predictions",
  "n_examples": 0,
  "n_independence_groups": 0,
  "ci": null,
  "failure_bounds": null,
  "reason": "illustrative_contract_only"
}
```

This example is deliberately unmeasured; never copy it into a claimed completed result. Real statuses include `ok`, `undefined`, `not_applicable`, `insufficient_data`, `not_computed`, and `provisional`.

For a defined FPR, `numerator = false_positives`, `denominator = scored_humans`, `value = numerator/denominator`, and `percent = 100*value`. Recall uses detected positives over scored positives under the specified task. AUROC/F1 do not have a simple rate numerator/denominator; those fields can be null, but support/class counts must be present. Undefined precision with zero predicted positives is null with a reason, not silently zero. A 0% measured FPR with a nonzero denominator is valid and distinct from null.

Confidence intervals state method, level, lower/upper, unit, clustering variable, number of groups, seed/replicates if bootstrapped, assumptions, and status. Percentage interval bounds can be derived for display. Never conflate a document-level interval, a group-level any-error interval, and a calibration-FPR target. Flag degenerate all-zero bootstrap intervals. Missing replicates/undefined one-class bootstrap samples must be counted rather than silently treated as zeros.

Failure bounds use all eligible predictions for the relevant label: for example, recall lower bound `detected/eligible_positives` and upper bound `(detected + unscored_positives)/eligible_positives`. Mark these as bounds, not observed conditional recall. Similarly provide FPR bounds when human predictions are missing.

Confusion matrices name class ordering and the treatment of mixed/unknown examples. Keep native localisation, approximate localisation, and document classification in separate tasks.

## 7. Summary and comparisons

The summary retains:

```text
schema_version, artifact_kind, data_origin, run_id, status
full_result_sha256
benchmark: identity/fingerprints, profile, purpose, realised counts/shortfalls
run: compact execution metadata
models[]: compact identity, score semantics, capabilities, exposure/status
protocols[]: fingerprints and readable descriptions
operating_points[]: exact thresholds, targets, achieved calibration rates, counts
source_status[]
cells[]: the aggregate records, metric values/percentages/counts/intervals
comparisons[]
coverage
warnings[]
```

Do not duplicate source texts, token logits, per-window arrays, or individual examples into the summary. It must be sufficient for report tables, charts, and compatibility checks without loading the full result. Use stable cell IDs and machine-readable metric names such as `human_fpr`, `fully_ai_recall`, `mixed_recall`, `auroc`, `average_precision`, `prediction_coverage`, and `human_character_false_highlight_rate`.

Generate the summary from full results with one deterministic reducer. Serialize full results first, hash their bytes, then put that hash in the summary; avoid a circular full/summary hash dependency. Deterministic re-export should not inject a new wall-clock timestamp or recalculate decisions from rounded percentages.

A comparison specifies left/right run and model IDs, task/panel/slice, protocol and calibration policy, eligible/intersection counts and common-example hash, metric, both estimates, difference, and uncertainty/status. Rate differences are **percentage points**, not relative percentages. Relative change is optional and undefined when its denominator is zero.

Before calling two results directly comparable, check benchmark/text/gold membership, task definitions, calibration membership and selection policy, and protocol/window/preprocessing compatibility. Model-specific threshold values naturally differ; identical thresholds are not a compatibility requirement. Model-specific dtype/quantisation differences must remain visible. Report model performance on their full supported populations and on the paired common-ID population, with coverage, rather than quietly intersecting away failures.

The viewer may join compatible summary files, but must request matching full files for per-example pairing or new paired bootstrap estimates. It must not infer paired significance from two separate marginal confidence intervals.

## 8. Semantic validation

Validate structural schemas and then these invariants:

1. IDs/references are unique and resolvable; text hashes, lengths, intervals, and parent mappings agree.
2. Calibration and evaluation are leakage-group-disjoint; all selected examples have explicit expected prediction coverage.
3. Successful raw scores are finite; bucket probabilities are normalised within a documented numerical tolerance; reported extent scores match the declared formula.
4. Shared-window inference never silently truncates; unsuccessful windows cannot yield a claimed complete max aggregate.
5. Every threshold and decision is reproducible; official calibration uses no evaluation IDs.
6. Metric populations/counts/confusions reconcile; summary values/percentages and coverage agree with full details; null and zero retain distinct meanings.
7. Unsupported native span metrics stay unavailable; unknown gold is not treated as human; code-point offsets survive viewer round trips.
8. Claimed run completion, source acquisition completeness, and model execution status match the recorded evidence.

Write validation outcomes into the result and fail export as `complete` when mandatory invariants fail. A readable, accurately labelled partial result is preferable to a polished but invalid scorecard.
