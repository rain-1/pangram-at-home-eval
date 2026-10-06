"""Detector extension interface.

A detector receives only stable input IDs and text (a canonical window or a full
canonical document). It never sees labels, provenance, gold spans or source hints.

To add a model:
  1. implement a subclass of `Detector` (usually ~100 lines) returning `DetectorOutput`s;
  2. add an entry to configs/models.yaml with `adapter: package.module:ClassName`;
  3. run `python -m aidet_eval predict --models <id>` then calibrate/evaluate/view as usual.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorInput:
    input_id: str  # stable, label-free ID (e.g. "<example_id>#w003" or "<example_id>#native")
    text: str  # exact canonical substring (window) or full canonical text
    canonical_offset: int = 0  # start of `text` within the canonical document (for coverage mapping)


@dataclass
class DetectorOutput:
    input_id: str
    status: str  # ok | error | abstain
    score: float | None = None  # oriented: higher = more AI signal
    raw_outputs: dict[str, Any] | None = None  # logits, probabilities, predicted bucket, ...
    native_spans: list[list[int]] | None = None  # None = unsupported
    coverage: dict[str, Any] = field(default_factory=dict)
    input_fingerprint: str | None = None
    n_tokens: int | None = None
    error: str | None = None
    seconds: float | None = None


@dataclass
class Capabilities:
    score_semantics: str
    score_range: tuple[float, float]
    score_orientation: str = "higher_is_more_ai"
    class_probabilities: str | None = None  # e.g. "ordinal_bucket_distribution"
    native_spans: bool = False
    supports_truncation_protocol: bool = True
    max_input_tokens: int | None = None


class Detector(ABC):
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.model_id: str = spec["id"]

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Model metadata for results (revisions, hashes, precision, preprocessing fingerprint)."""

    @abstractmethod
    def predict(self, inputs: list[DetectorInput], truncate: bool) -> list[DetectorOutput]:
        """Score inputs. With truncate=False an over-budget input must be an error, never truncated."""

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Model-visible token counts (after the model's own preprocessing, with special tokens)."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        from aidet_eval.util import fingerprint

        return fingerprint(self.describe())


def load_detector(spec: dict[str, Any]) -> Detector:
    mod, cls = spec["adapter"].split(":")
    return getattr(importlib.import_module(mod), cls)(spec)
