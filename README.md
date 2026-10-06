# aidet_eval — reproducible AI-text-detector evaluation suite

Evaluates detectors on where they succeed and fail, keeping **domain/category** and **human / AI / mixed authorship**
(plus mixture mechanism) as independent axes. Human false-positive rates and AI/mixed recall are reported per domain at
operating points frozen on a disjoint human calibration set. There is deliberately no single overall score.

Specification inputs (unchanged, read-only): `spec/` (`HANDOFF.md`, `RESULTS_CONTRACT.md`, `benchmark_plan.yaml`,
`ai_detector_dataset_catalogue.json`, `verified_baselines.json`; checksums in `spec/SHA256SUMS`).

## Environment

```bash
uv sync --extra gpu --group dev        # Python 3.12, pinned by uv.lock (torch 2.14 / transformers 5.18 / peft 0.21 / bitsandbytes 0.50)
uv run python -m aidet_eval doctor     # packages, GPU, disk, HF access (token never printed), snapshots, source status
```

Baseline inference needs a CUDA GPU (reference precision: RoBERTa fp32 weights + bf16 autocast; Llama NF4 4-bit + bf16
compute) and existing Hugging Face access to `pangram/editlens_roberta-large`, `pangram/editlens_Llama-3.2-3B` and
`meta-llama/Llama-3.2-3B`. The tool never accepts access conditions for you; if `doctor` reports no access, accept them on
the model pages yourself and log in with `! huggingface-cli login`. Tests need neither GPU nor corpora.

## Reproduction (exact commands used for benchmark-v1)

```bash
uv run python -m aidet_eval download --catalogue spec/ai_detector_dataset_catalogue.json --all --resume   # 16 sources, ~38 GB, checksum-verified
uv run python -m aidet_eval status                                   # per-source acquisition status
uv run python -m aidet_eval.model_snapshots                          # pinned model/tokenizer snapshots -> models/hf (~7.4 GB)
uv run python -m aidet_eval normalise                                # raw -> data/normalised/<source>.parquet + measured inventories
uv run python -m aidet_eval inventory                                # data/inventory/inventory.json + source_audit.json
uv run python -m aidet_eval build --config spec/benchmark_plan.yaml --profile full    # -> artifacts/benchmark-v1 (frozen)
uv run python -m aidet_eval validate --benchmark artifacts/benchmark-v1
HF_HUB_OFFLINE=1 uv run python -m aidet_eval windows --benchmark artifacts/benchmark-v1   # frozen shared_windows_v1 manifest
HF_HUB_OFFLINE=1 uv run python -m aidet_eval parity                  # adapter vs pinned upstream EditLens inference
HF_HUB_OFFLINE=1 uv run python -m aidet_eval predict --benchmark artifacts/benchmark-v1 \
    --models open_pangram_roberta,open_pangram_llama --protocols shared_windows_v1,native_truncate_v1 --resume
uv run python -m aidet_eval calibrate --benchmark artifacts/benchmark-v1
uv run python -m aidet_eval evaluate  --benchmark artifacts/benchmark-v1      # writes results.full/summary JSON, schemas, CSV, report.md
uv run python -m aidet_eval export --run artifacts/runs/<run_id>               # re-derive + verify summary from the full file
uv run python -m aidet_eval export --run artifacts/runs/<run_id> --redact      # local text-redacted copy
uv run python -m aidet_eval view --runs artifacts/runs                         # http://127.0.0.1:8765/
uv run pytest -q
```

The default run ID is `<benchmark_id>__<benchmark_hash[:10]>`; `predict --resume` reuses existing shards whose cache keys
(model fingerprint, protocol fingerprint, input text hash, truncation flag) match, and refuses to mix benchmarks under one run ID.

**benchmark-v2** (current; v1 stays frozen for comparison). Adds sources requested on 2026-10-06 (StoryScope, HAP-E-2,
LAMP, CNN/DailyMail, generate-heterogeneous v1.3.0 test split), lossless detokenisation of WikiText/PubMed, drops DAMASHA and
XSum (see `configs/benchmark_plan_v2.yaml` and `docs/BENCHMARK_CARD_v2.md`):

```bash
uv run python -m aidet_eval download --sources storyscope,hap_e2,lamp,cnn_dailymail,generate_heterogeneous
uv run python -m aidet_eval normalise --sources storyscope,hap_e2,lamp,cnn_dailymail,generate_heterogeneous,wikitext_detok,scientific_papers_detok
uv run python -m aidet_eval build --config configs/benchmark_plan_v2.yaml --profile full        # -> artifacts/benchmark-v2
HF_HUB_OFFLINE=1 uv run python -m aidet_eval windows --benchmark artifacts/benchmark-v2
HF_HUB_OFFLINE=1 uv run python -m aidet_eval predict --benchmark artifacts/benchmark-v2 --resume   # reuses identical v1 inputs
uv run python -m aidet_eval calibrate --benchmark artifacts/benchmark-v2
uv run python -m aidet_eval evaluate  --benchmark artifacts/benchmark-v2
uv run python -m aidet_eval calibration-sensitivity --benchmark artifacts/benchmark-v2 --run artifacts/runs/ai-detector-panels-v2__0393c8cb50
```

**Thresholds calibrated elsewhere** (for your own models): put them in a YAML file and pass it to `evaluate`. They become
operating points `external:<name>` with the same strict `score > threshold` rule, alongside (never replacing) the
benchmark's own calibration:

```yaml
source: "calibrated on my held-out calibration set, 2026-10-10"
thresholds:
  my_model:
    shared_windows_v1: {my_cal_1pct: 0.83, my_cal_5pct: 0.61}
```

```bash
uv run python -m aidet_eval evaluate --benchmark artifacts/benchmark-v2 --external-thresholds my_thresholds.yaml
```

**Calibration sensitivity** (`calibration-sensitivity`, viewer tab "Calibration sensitivity"): thresholds re-derived on each
calibration dataset alone with bootstrap intervals, leave-one-dataset-out / equal-weight / max-over-datasets variants, and
what each would do on the evaluation panels. Exploratory only.

**Smoke profile** (separate benchmark ID, ~1% scale, for checking the software end to end):

```bash
uv run python -m aidet_eval build --config spec/benchmark_plan.yaml --profile smoke     # -> artifacts/benchmark-smoke-v1
HF_HUB_OFFLINE=1 uv run python -m aidet_eval windows  --benchmark artifacts/benchmark-smoke-v1
HF_HUB_OFFLINE=1 uv run python -m aidet_eval predict  --benchmark artifacts/benchmark-smoke-v1
uv run python -m aidet_eval evaluate --benchmark artifacts/benchmark-smoke-v1
```

## Layout

| path | content |
|---|---|
| `configs/sources.yaml` | resolved acquisition registry (pinned revisions / upstream hashes) for all 16 catalogue entries |
| `configs/models.yaml` | detector registry (adapter class, checkpoint/base revisions, precision, batch size) |
| `data/raw/<source>/` | read-only original release files + `_acquisition.json` (status, hashes, continuation command) |
| `data/normalised/<source>.parquet` | canonical candidate records (one schema for all sources) |
| `data/inventory/` | `inventory.json` (all sources), `source_audit.json` (catalogue corrections), per-source measured inventories |
| `models/hf/` | pinned model snapshots (HF cache layout); `models/<id>.snapshot.json` file hashes |
| `artifacts/benchmark-v1/` | frozen benchmark: `examples.parquet`, `manifest.json`, `selection_log.parquet`, `overlap/`, `windows/`, `BENCHMARK_CARD.md`, `FROZEN` |
| `artifacts/runs/<run_id>/` | `predictions/<model>/<protocol>.jsonl` shards, `calibration.json`, `results.full.json(.gz)`, `results.summary.json`, schemas, `results.summary.cells.csv`, `report.md` |
| `artifacts/parity/parity_report.json` | loader parity vs pinned upstream |
| `artifacts/logs/execution_log.jsonl` | structured, secret-redacted execution log |
| `docs/` | benchmark card, execution report |

## Semantics you should not misread

* Open Pangram scores are an **ordinal AI-intervention extent signal**: `sum_k softmax(logits)_k * k / (K-1)` with K=4 buckets
  verified from each checkpoint. They are not calibrated probabilities of AI presence and not AI-character fractions;
  no Brier/ECE is computed.
* `shared_windows_v1` (primary): frozen ~192-word windows with 64-word overlap, shrunk until every window fits both
  baselines' budgets after reference preprocessing; document score = `max(window score)` (a predeclared harness wrapper).
  `native_truncate_v1` (secondary): upstream behaviour, full-document preprocessing + 512/1024-token truncation, with
  recorded model-visible coverage. Each protocol is calibrated separately.
* Thresholds: `k = floor(alpha * n)` on human calibration scores sorted descending; `threshold = scores[k]`; flag iff
  `score > threshold`. 1% (primary) and 5% targets; they are calibration properties, not guaranteed test FPRs.
* Intervals: Wilson for one-per-group observations, independence-group percentile bootstrap (2000 replicates, seed 42)
  otherwise; paired group bootstrap for model differences (percentage points).

## Adding a detector

1. Implement `aidet_eval.models.base.Detector` (`load`, `capabilities`, `describe`, `predict(inputs, truncate)`, optionally
   `count_tokens`). Inputs carry only an ID, canonical text and its canonical offset — never labels or provenance.
2. Add an entry to `configs/models.yaml` with `adapter: your.module:YourDetector` (and a pinned revision).
3. Run `predict --models <id>`, then `calibrate`, `evaluate`, `view`. If your model cannot consume the frozen
   `shared_windows_v1` windows, it is protocol-incompatible until a new, separately versioned protocol is defined.
4. Optional: list its training texts/sources in `configs/training_registry.yaml` (`models: {<id>: [{path, kind}]}` with kind
   `text_sha256 | source_group_id | example_id`) and rebuild a new benchmark version to get its exposure view.

## Licensing / sharing

Benchmark texts come from sources with non-commercial or uncleared terms (see `data/inventory/inventory.json`). Results
files contain texts; keep them local or use `export --redact`. Nothing is uploaded by this tool.
