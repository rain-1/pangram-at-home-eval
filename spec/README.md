# AI-detector evaluation handoff bundle

Give this directory to a coding agent with access to the intended repository and a Python/GPU environment. Tell it to read `HANDOFF.md` and execute the assignment.

## Contents

- `HANDOFF.md`: the main implementation prompt, including acquisition of all 16 catalogue sources, subset selection, both exact Open Pangram baseline integrations, calibration, metrics, viewer, and completion criteria.
- `RESULTS_CONTRACT.md`: full and summary JSON semantics, required records, denominators, compatibility rules, and validation. The agent must implement typed schemas from this contract.
- `benchmark_plan.yaml`: machine-readable starting targets and evaluation policies. Counts are proposals, not measured availability.
- `ai_detector_dataset_catalogue.json`: the original research catalogue, retained unchanged.
- `verified_baselines.json`: model identities and primary-source integration findings checked on 6 October 2026, including source-file hashes and unresolved weight-access checks.
- `SHA256SUMS`: bundle-file checksums.

## Launcher prompt

Read `HANDOFF.md`, `RESULTS_CONTRACT.md`, `benchmark_plan.yaml`, `verified_baselines.json`, and `ai_detector_dataset_catalogue.json`. Implement and execute the described evaluation project in the current repository. Download and inventory every catalogue source, freeze justified source-disjoint subsets, run the two specified Open Pangram baselines through validated adapters, and produce the full and summary JSON exports plus a working local viewer. Reuse repository conventions. Proceed through independent work when a source or model is blocked, recording accurate statuses; do not substitute models or fabricate results. Finish with actual artifact paths, reproduction commands, tests, and the execution report required by the handoff.

## Status of this bundle

This bundle contains an implementation specification, catalogue, and source verification—not downloaded datasets, trained models, an implemented viewer, or completed baseline evaluations. Both baseline model pages advertise gated file access; existing authorised Hugging Face access is needed for execution. No access conditions were accepted and no gated weights were downloaded to prepare this bundle.
