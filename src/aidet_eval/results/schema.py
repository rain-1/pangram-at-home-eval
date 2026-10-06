"""Typed models for the unified evaluation result contract v1.0 (RESULTS_CONTRACT.md).

JSON Schemas for results.full.json and results.summary.json are generated from these models
(`python -m aidet_eval.results.schema` or automatically on export). Nested dataset-native
metadata lives in explicit `extensions` objects.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
Status = Literal["complete", "partial", "blocked", "failed"]
DataOrigin = Literal["measured", "synthetic_fixture"]
MetricStatus = Literal["ok", "undefined", "not_applicable", "insufficient_data", "not_computed", "provisional"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class CI(Open):
    method: str
    level: float
    lower: float
    upper: float
    unit: str
    clustering_variable: Optional[str] = None
    n_groups: int
    seed: Optional[int] = None
    replicates: Optional[int] = None
    assumptions: str
    status: str
    degenerate_all_zero: bool = False
    undefined_replicates: int = 0


class FailureBounds(Strict):
    lower: float
    upper: float
    n_eligible: int
    n_unscored: int
    note: str


class Metric(Open):
    status: MetricStatus
    value: Optional[float]
    unit: str
    percent: Optional[float]
    numerator: Optional[int]
    denominator: Optional[int]
    denominator_population: Optional[str]
    n_examples: int
    n_independence_groups: int
    ci: Optional[CI]
    failure_bounds: Optional[FailureBounds]
    reason: Optional[str]


class Support(Strict):
    selected: int
    eligible: int
    scored: int
    error: int
    abstained: int
    skipped: int
    blocked: int
    human: int
    ai: int
    mixed: int
    unknown: int
    n_source_groups: int
    n_leakage_groups: int
    n_independence_groups: int


class Cell(Strict):
    cell_id: str
    model_id: str
    protocol_id: str
    operating_point_id: Optional[str]
    task_id: Optional[str]
    panel_id: str
    split: str
    slice: dict[str, Any]
    population_definition: str
    example_ids_hash: str
    support: Support
    metrics: dict[str, Metric]
    confusion_matrix: Optional[dict[str, Any]]
    warnings: list[str]


class Calibration(Open):
    calibration_id: str
    model_id: str
    protocol_id: str
    operating_point_id: str
    primary: bool
    task: str
    target_fpr: Optional[float]
    comparator: Literal[">"]
    threshold: Optional[float]
    status: Literal["ok", "provisional", "blocked"]
    reasons: list[str]
    calibration_example_ids_sha256: str
    n_selected: int
    n_eligible: int
    n_scored: int
    n_failed: int
    false_positives: Optional[int]
    actual_fpr: Optional[float]
    tied_scores_at_threshold: Optional[int]
    selection_algorithm: str


class Window(Strict):
    window_id: str
    canonical_start: int
    canonical_end: int
    n_words: int
    construction: str
    status: str
    score: Optional[float]
    raw_outputs: Optional[dict[str, Any]]
    n_tokens_model_visible: Optional[int]
    model_input_transform: str
    model_input_sha256: Optional[str]
    coverage: dict[str, Any]
    error: Optional[str]


class Decision(Strict):
    operating_point_id: str
    calibration_id: str
    flag_ai: Optional[bool]


class Prediction(Strict):
    prediction_id: str
    example_id: str
    model_id: str
    protocol_id: str
    role: Literal["calibration", "evaluation"]
    status: Literal["ok", "error", "skipped", "blocked"]
    status_detail: Optional[str]
    score: Optional[float]
    score_semantics: str
    partial_max_lower_bound: Optional[float]
    raw_outputs: Optional[dict[str, Any]]
    windows: Optional[list[Window]]
    native_span_predictions: Optional[list[list[int]]]
    coverage: dict[str, Any]
    operating_point_decisions: list[Decision]
    timing: dict[str, Any]
    cache_key: Optional[str]
    input_fingerprint: Optional[str]
    error_id: Optional[str]
    warnings: list[str]


class ExampleSource(Strict):
    dataset_id: str
    dataset_key: str
    revision: str
    config: Optional[str]
    original_split: Optional[str]
    original_row_id: str
    parent_ids: list[str]
    source_urls: list[str]


class Truth(Strict):
    authorship: Literal["human", "ai", "mixed", "unknown"]
    authorship_basis: Optional[str]
    ai_involvement: Literal["yes", "no", "unknown"]
    mix_mechanism: Optional[str]
    generator: Optional[str]
    generation_parameters: Optional[dict[str, Any]]
    ai_char_spans: Optional[list[list[int]]]
    human_char_spans: Optional[list[list[int]]]
    unknown_char_spans: Optional[list[list[int]]]
    annotation_completeness: str
    annotation_provenance: Optional[str]
    ai_fraction_char: Optional[float]
    span_count: Optional[int]
    span_lengths: Optional[list[int]]


class HumanProvenance(Strict):
    category: Optional[str]
    tier: Optional[str]
    evidence: Optional[dict[str, Any]]
    verified_dates: Optional[dict[str, Any]]
    eligibility: str
    caveats: list[str]


class Example(Strict):
    example_id: str
    source: ExampleSource
    source_group_id: str
    leakage_group_id: str
    independence_group_id: str
    role: Literal["calibration", "evaluation"]
    panel_ids: list[str]
    domain: Optional[str]
    subdomain: Optional[str]
    language: Optional[str]
    original_labels: dict[str, Any]
    truth: Truth
    human_provenance: HumanProvenance
    text: Optional[str]
    text_sha256: str
    word_count: int
    char_count: int
    length_bin: str
    parent_excerpt: Optional[dict[str, Any]]
    canonicalisation: dict[str, Any]
    rights: dict[str, Any]
    training_exposure: dict[str, str]
    artefact_flags: list[str]
    extensions: dict[str, Any]


class Source(Open):
    source_id: str
    name: str
    authoritative_location: str
    pinned_revision: Optional[str]
    acquisition_status: Literal["complete", "partial", "blocked", "failed", "not_started"]
    download_completeness: dict[str, Any]
    reported_rows: Optional[Any]
    observed_rows: Optional[int]
    eligibility_status: str
    provenance_assessment: str
    rights_restrictions: str


class Model(Open):
    model_id: str
    display_name: str
    status: str
    score_semantics: str
    model_spec_fingerprint: Optional[str]


class Protocol(Open):
    protocol_id: str
    version: str
    implementation_kind: str
    fingerprint: str
    localisation_mode: str


class Comparison(Open):
    comparison_id: str
    left: dict[str, Any]
    right: dict[str, Any]
    protocol_id: str
    operating_point_id: str
    metric: str
    n_common_scored: int
    difference_pp: Optional[float]
    status: str


class Validation(Strict):
    structural_ok: bool
    semantic_checks: list[dict[str, Any]]
    ok: bool


class FullResult(Strict):
    schema_version: Literal["1.0.0"]
    artifact_kind: Literal["aidet_eval.results.full"]
    data_origin: DataOrigin
    run_id: str
    status: Status
    benchmark: dict[str, Any]
    run: dict[str, Any]
    sources: list[Source]
    models: list[Model]
    protocols: list[Protocol]
    calibrations: list[Calibration]
    examples: list[Example]
    predictions: list[Prediction]
    aggregates: list[Cell]
    comparisons: list[Comparison]
    errors: list[dict[str, Any]]
    validation: Validation
    warnings: list[dict[str, Any]]
    extensions: dict[str, Any]


class SummaryResult(Strict):
    schema_version: Literal["1.0.0"]
    artifact_kind: Literal["aidet_eval.results.summary"]
    data_origin: DataOrigin
    run_id: str
    status: Status
    full_result_sha256: str
    benchmark: dict[str, Any]
    run: dict[str, Any]
    models: list[dict[str, Any]]
    protocols: list[dict[str, Any]]
    operating_points: list[dict[str, Any]]
    source_status: list[dict[str, Any]]
    cells: list[Cell]
    comparisons: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    extensions: dict[str, Any] = Field(default_factory=dict)


def write_schemas(out_dir) -> dict[str, str]:
    import json
    from pathlib import Path

    out = {}
    for name, model in (("results.full.schema.json", FullResult), ("results.summary.schema.json", SummaryResult)):
        s = model.model_json_schema()
        s["$id"] = f"urn:aidet_eval:{name}:{SCHEMA_VERSION}"
        s["title"] = f"{name} (contract 1.0, schema {SCHEMA_VERSION})"
        p = Path(out_dir) / name
        p.write_text(json.dumps(s, indent=1))
        out[name] = str(p)
    return out


if __name__ == "__main__":
    import sys

    print(write_schemas(sys.argv[1] if len(sys.argv) > 1 else "."))
