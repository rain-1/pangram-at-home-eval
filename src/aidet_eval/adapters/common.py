"""Canonical record construction and measured-inventory helpers shared by source adapters."""

from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from aidet_eval import spans as S
from aidet_eval.util import sha256_text

SEED = 42

# Canonical record columns (nested values are JSON strings in parquet).
JSON_COLS = [
    "parent_ids", "source_urls", "leakage_keys", "original_labels", "generation_parameters", "ai_char_spans",
    "human_char_spans", "unknown_char_spans", "verified_dates", "eligibility_reasons", "canonicalisation", "parent_excerpt",
    "rights", "artefact_flags", "extensions", "human_provenance_evidence",
]


def hash_rank(*parts: Any, seed: int = SEED) -> str:
    """Deterministic population-wide sampling key (lower = selected first)."""
    return hashlib.sha256(("|".join(map(str, (seed, *parts)))).encode()).hexdigest()


@dataclass
class Rec:
    dataset_key: str
    dataset_id: str
    revision: str
    original_row_id: str
    text: str
    config: str | None = None
    original_split: str | None = None
    variant: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    source_group_id: str | None = None
    leakage_keys: list[str] = field(default_factory=list)
    independence_key: str | None = None
    domain: str | None = None
    subdomain: str | None = None
    language: str | None = "en"
    original_labels: dict = field(default_factory=dict)
    authorship: str = "unknown"  # human | ai | mixed | unknown
    authorship_basis: str | None = None
    ai_involvement: str = "unknown"  # yes | no | unknown
    mix_mechanism: str | None = None
    generator: str | None = None
    generation_parameters: dict | None = None
    ai_char_spans: list | None = None
    human_char_spans: list | None = None
    unknown_char_spans: list | None = None
    annotation_completeness: str = "none"  # complete | partial | process_only | none
    annotation_provenance: str | None = None
    human_provenance_category: str | None = None
    human_provenance_evidence: dict | None = None
    provenance_tier: str | None = None  # independently_audited | benchmark_provided | curator_attested | logged_session | in_family_reference
    verified_dates: dict | None = None
    eligibility: str = "eligible"  # eligible | reserve | excluded | quarantined
    eligibility_reasons: list[str] = field(default_factory=list)
    original_text_sha256: str | None = None
    canonicalisation: dict = field(default_factory=lambda: {"transformations": [], "offset_mapping": "identity"})
    parent_excerpt: dict | None = None
    rights: dict = field(default_factory=dict)
    artefact_flags: list[str] = field(default_factory=list)
    extensions: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        try:
            S.validate_text(self.text)
            n = len(self.text)
            ai = S.validate_intervals(self.ai_char_spans, n)
            hu = S.validate_intervals(self.human_char_spans, n)
            S.validate_intervals(self.unknown_char_spans, n)
        except S.SpanError as e:
            self.eligibility = "quarantined"
            self.eligibility_reasons = self.eligibility_reasons + [f"invalid:{e}"]
            ai = hu = None
            self.ai_char_spans = self.human_char_spans = None
        complete = self.annotation_completeness == "complete"
        if ai is not None:
            ai = S.union(ai)
        if complete and ai is not None and hu is None:
            hu = S.complement(ai, len(self.text))  # only justified under exhaustive annotation
        if not self.text.strip():
            self.eligibility = "quarantined"
            self.eligibility_reasons = self.eligibility_reasons + ["empty_text"]
        frac = S.ai_fraction(ai, len(self.text), complete)
        vid = f":{self.variant}" if self.variant else ""
        row = {
            "example_id": f"{self.dataset_key}:{self.original_split or '-'}:{self.original_row_id}{vid}",
            "dataset_key": self.dataset_key, "dataset_id": self.dataset_id, "revision": self.revision,
            "config": self.config, "original_split": self.original_split, "original_row_id": str(self.original_row_id),
            "variant": self.variant, "parent_ids": self.parent_ids, "source_urls": self.source_urls,
            "source_group_id": self.source_group_id or f"{self.dataset_key}:{self.original_row_id}",
            "leakage_keys": self.leakage_keys, "independence_key": self.independence_key,
            "domain": self.domain, "subdomain": self.subdomain, "language": self.language,
            "original_labels": self.original_labels, "authorship": self.authorship,
            "authorship_basis": self.authorship_basis, "ai_involvement": self.ai_involvement,
            "mix_mechanism": self.mix_mechanism, "generator": self.generator,
            "generation_parameters": self.generation_parameters,
            "ai_char_spans": [list(x) for x in ai] if ai is not None else None,
            "human_char_spans": [list(x) for x in hu] if hu is not None else None,
            "unknown_char_spans": self.unknown_char_spans,
            "annotation_completeness": self.annotation_completeness,
            "annotation_provenance": self.annotation_provenance,
            "ai_fraction_char": frac,
            "span_count": len(ai) if ai is not None else None,
            "human_provenance_category": self.human_provenance_category,
            "human_provenance_evidence": self.human_provenance_evidence,
            "provenance_tier": self.provenance_tier,
            "verified_dates": self.verified_dates,
            "eligibility": self.eligibility, "eligibility_reasons": self.eligibility_reasons,
            "text": self.text, "text_sha256": sha256_text(self.text),
            "original_text_sha256": self.original_text_sha256 or sha256_text(self.text),
            "canonicalisation": self.canonicalisation, "parent_excerpt": self.parent_excerpt,
            "word_count": S.word_count(self.text), "char_count": len(self.text),
            "rights": self.rights, "artefact_flags": self.artefact_flags, "extensions": self.extensions,
        }
        return row


class Inventory:
    """Measured inventory accumulated while streaming a full source."""

    def __init__(self, key: str):
        self.key = key
        self.counters: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.rows_by_file: dict[str, int] = collections.Counter()
        self.schemas: dict[str, Any] = {}
        self.notes: list[str] = []
        self.facts: dict[str, Any] = {}

    def count(self, field_: str, value: Any, n: int = 1) -> None:
        self.counters[field_][str(value)] += n

    def to_dict(self) -> dict:
        return {
            "measured_rows_by_file": dict(self.rows_by_file),
            "measured_rows_total": int(sum(self.rows_by_file.values())),
            "schemas": self.schemas,
            "measured_counts": {k: dict(v.most_common()) for k, v in self.counters.items()},
            "facts": self.facts,
            "notes": self.notes,
        }


def contiguous_excerpt(paragraph_spans: list[tuple[int, int]], text: str, rank_key: str,
                       min_words: int = 150, max_words: int = 500) -> tuple[int, int] | None:
    """Deterministic paragraph-aligned contiguous excerpt.

    Starts at a hash-chosen paragraph (gold-independent) and extends forward while staying
    <= max_words; returns None if no window reaches min_words.
    """
    if not paragraph_spans:
        return None
    wc = [S.word_count(text[a:b]) for a, b in paragraph_spans]
    n = len(paragraph_spans)
    start_idx = int(hashlib.sha256(rank_key.encode()).hexdigest(), 16) % n
    for offset in range(n):
        i = (start_idx + offset) % n
        total, j = 0, i
        while j < n and total + wc[j] <= max_words:
            total += wc[j]
            j += 1
        if total >= min_words:
            return paragraph_spans[i][0], paragraph_spans[j - 1][1]
    return None


def paragraph_spans(text: str, sep: str = "\n") -> list[tuple[int, int]]:
    out, pos = [], 0
    for part in text.split(sep):
        a, b = pos, pos + len(part)
        if part.strip():
            # trim surrounding whitespace but keep exact substring offsets
            la = len(part) - len(part.lstrip())
            rb = len(part.rstrip())
            out.append((a + la, a + rb))
        pos = b + len(sep)
    return out


def dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)
