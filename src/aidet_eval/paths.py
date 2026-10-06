"""Repository-relative locations. Override the root with AIDET_ROOT."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AIDET_ROOT", Path(__file__).resolve().parents[2]))
SPEC = ROOT / "spec"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"  # read-only original release files, one directory per source
NORMALISED = DATA / "normalised"  # derived canonical parquet per source
INVENTORY = DATA / "inventory"  # measured per-source inventory JSON
ARTIFACTS = ROOT / "artifacts"
RUNS = ARTIFACTS / "runs"
LOGS = ARTIFACTS / "logs"
MODELS = ROOT / "models"  # local pinned model snapshots (HF cache layout)
CATALOGUE = SPEC / "ai_detector_dataset_catalogue.json"
PLAN = SPEC / "benchmark_plan.yaml"
BASELINES = SPEC / "verified_baselines.json"
