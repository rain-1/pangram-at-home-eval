"""Character-span utilities.

Convention: zero-based, half-open [start, end) intervals in Unicode code points into the
exact stored canonical text (Python `text[a:b]`). Not UTF-8 bytes, not UTF-16 units.
`None` = unavailable; `[]` = exhaustively established empty set (only with complete annotation).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

Interval = tuple[int, int]


class SpanError(ValueError):
    pass


def validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise SpanError("text is not a str")
    try:
        text.encode("utf-8", "strict")
    except UnicodeEncodeError as e:  # lone surrogates etc.
        raise SpanError(f"invalid unicode: {e.reason}") from e
    if "\x00" in text:
        raise SpanError("NUL character in text")


def validate_intervals(spans: Iterable[Sequence[int]] | None, n: int) -> list[Interval] | None:
    if spans is None:
        return None
    out = []
    for s in spans:
        if len(s) != 2:
            raise SpanError(f"interval must have 2 ints: {s!r}")
        a, b = s
        if not (isinstance(a, int) and isinstance(b, int)) or isinstance(a, bool) or isinstance(b, bool):
            raise SpanError(f"non-integer interval {s!r}")
        if a < 0 or b > n:
            raise SpanError(f"interval {s!r} out of range for length {n}")
        if a >= b:
            raise SpanError(f"empty or reversed interval {s!r}")
        out.append((a, b))
    return out


def union(spans: Iterable[Interval]) -> list[Interval]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def complement(spans: Iterable[Interval], n: int) -> list[Interval]:
    out, pos = [], 0
    for a, b in union(spans):
        if a > pos:
            out.append((pos, a))
        pos = b
    if pos < n:
        out.append((pos, n))
    return out


def total_length(spans: Iterable[Interval]) -> int:
    return sum(b - a for a, b in union(spans))


def ai_fraction(spans: list[Interval] | None, n: int, complete: bool) -> float | None:
    """Union of AI intervals / canonical length (whitespace included). Null if not identifiable."""
    if spans is None or not complete or n == 0:
        return None
    return total_length(spans) / n


def crop(spans: list[Interval] | None, start: int, end: int) -> list[Interval] | None:
    """Remap intervals into the excerpt text[start:end] (offsets relative to the excerpt)."""
    if spans is None:
        return None
    out = []
    for a, b in spans:
        a2, b2 = max(a, start), min(b, end)
        if a2 < b2:
            out.append((a2 - start, b2 - start))
    return union(out)


def strip_tags_with_offsets(tagged: str, open_tag: str, close_tag: str) -> tuple[str, list[Interval]]:
    """Remove paired annotation tags, returning clean text and code-point AI intervals.

    Raises SpanError on unbalanced or nested tags (quarantine rather than repair).
    """
    pat = re.compile(re.escape(open_tag) + "|" + re.escape(close_tag))
    out_parts, spans, pos, clean_len, open_at = [], [], 0, 0, None
    for m in pat.finditer(tagged):
        seg = tagged[pos:m.start()]
        out_parts.append(seg)
        clean_len += len(seg)
        if m.group(0) == open_tag:
            if open_at is not None:
                raise SpanError("nested open tag")
            open_at = clean_len
        else:
            if open_at is None:
                raise SpanError("close tag without open tag")
            if clean_len > open_at:
                spans.append((open_at, clean_len))
            open_at = None
        pos = m.end()
    if open_at is not None:
        raise SpanError("unclosed open tag")
    out_parts.append(tagged[pos:])
    return "".join(out_parts), union(spans)


WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    """Documented length function: number of maximal non-whitespace runs (model-independent)."""
    return sum(1 for _ in WORD_RE.finditer(text))


def block_position(spans: list[Interval], n: int) -> str | None:
    """Coarse location of AI blocks: start/middle/end/multiple-regions, from interval midpoints."""
    if not spans or n == 0:
        return None
    zones = set()
    for a, b in spans:
        mid = (a + b) / 2 / n
        zones.add("start" if mid < 1 / 3 else "end" if mid > 2 / 3 else "middle")
    return next(iter(zones)) if len(zones) == 1 else "multiple"


def rederive_label(spans: list[Interval] | None, n: int, complete: bool) -> tuple[str, float | None]:
    """Authorship of a (possibly cropped) excerpt from its remapped AI spans.

    A crop that removes all AI text is human only if the parent annotation was exhaustive;
    otherwise the label is unknown. Returns (authorship, ai_fraction_char).
    """
    if spans is None:
        return "unknown", None
    tot = total_length(spans)
    if tot == 0:
        return ("human", 0.0) if complete else ("unknown", None)
    if tot >= n:
        return "ai", 1.0
    return "mixed", (tot / n if complete else None)
