"""EditLens reference preprocessing, versioned inside the adapter.

`clean_text` reproduces pangramlabs/EditLens scripts/preprocess.py (blob
7d615e32612a916675b8345e6888294ffad60031, commit 05a588f15d792330ccaf91be8ee4fdb54ce26835)
step for step. It is applied only to model inputs, never to the canonical benchmark text.

`clean_text_with_mapping` additionally tracks, for every output character, the
canonical code-point index it came from. The mapping is trusted only when the
tracked output equals `clean_text(text)` exactly; otherwise it is reported as
unavailable with a reason rather than approximated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import emoji

UPSTREAM = {
    "repo": "https://github.com/pangramlabs/EditLens",
    "commit": "05a588f15d792330ccaf91be8ee4fdb54ce26835",
    "path": "scripts/preprocess.py",
    "blob_sha": "7d615e32612a916675b8345e6888294ffad60031",
    "steps": ["normalize_emoji", "remove_think_tag", "remove_ai_header", "lowercase", "normalize_whitespace"],
}
PREPROCESS_VERSION = "editlens_clean_text@7d615e3"

BOILERPLATE_STARTS = [
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
]


# ---- exact upstream semantics -------------------------------------------------

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_emoji(text):
    return emoji.demojize(text)


def remove_think_tag(text):
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    return text


def _header_matches(first_paragraph: str) -> bool:
    first_paragraph = re.sub(r"^[^a-zA-Z0-9]*", "", first_paragraph)
    first_paragraph = emoji.replace_emoji(first_paragraph, "")
    return any(first_paragraph.startswith(phrase) for phrase in BOILERPLATE_STARTS)


def remove_ai_header(text):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) == 0:
        return text
    if _header_matches(paragraphs[0]):
        if len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return text


def clean_text(text):
    text = normalize_emoji(text)
    text = remove_think_tag(text)
    text = remove_ai_header(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


# ---- offset-tracking re-implementation -----------------------------------------
# A tracked string is (chars: list[str], src: list[int]) where src[i] is the canonical
# code-point index that produced chars[i] (each element is exactly one code point).


@dataclass
class CleanResult:
    text: str
    src: list[int] | None  # canonical index per output code point, or None
    mapping_status: str  # "exact" | "unavailable"
    mapping_reason: str | None
    steps_applied: dict  # which deleting steps fired
    deleted_intervals: list[tuple[int, int]] | None  # canonical intervals removed (non-whitespace evidence)


def _t_demojize(chars, src):
    s = "".join(chars)
    out_c, out_s, pos = [], [], 0
    for m in emoji.emoji_list(s):
        a, b = m["match_start"], m["match_end"]
        out_c += chars[pos:a]
        out_s += src[pos:a]
        rep = emoji.demojize(s[a:b])
        out_c += list(rep)
        # spread the alias over the whole emoji sequence so no source code point looks deleted
        n = b - a
        out_s += [src[a + min(k * n // max(len(rep), 1), n - 1)] for k in range(len(rep))]
        pos = b
    out_c += chars[pos:]
    out_s += src[pos:]
    return out_c, out_s


def _t_strip(chars, src):
    s = "".join(chars)
    left = len(s) - len(s.lstrip())
    right = len(s.rstrip())
    return chars[left:right], src[left:right]


def _t_think(chars, src, fired):
    s = "".join(chars)
    if "</think>" in s:
        fired["remove_think_tag"] = True
        parts = s.split("</think>")
        a = len(parts[0]) + len("</think>")
        b = a + len(parts[1])
        return _t_strip(chars[a:b], src[a:b])
    return chars, src


def _t_header(chars, src, fired):
    s = "".join(chars)
    lines, starts, pos = s.split("\n"), [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    keep = [(ln, st) for ln, st in zip(lines, starts) if ln.strip()]
    if not keep:
        return chars, src
    if _header_matches(keep[0][0]) and len(keep) > 1:
        fired["remove_ai_header"] = True
        out_c, out_s = [], []
        for j, (ln, st) in enumerate(keep[1:]):
            if j:
                prev_ln, prev_st = keep[1:][j - 1]
                nl = prev_st + len(prev_ln)  # the newline that ended the previous kept line
                out_c.append("\n")
                out_s.append(src[nl] if nl < len(src) else src[-1])
            out_c += chars[st:st + len(ln)]
            out_s += src[st:st + len(ln)]
        return out_c, out_s
    return chars, src


def _t_lower(chars, src):
    whole = "".join(chars).lower()
    per = [c.lower() for c in chars]
    out_c, out_s = [], []
    for c, s_ in zip(per, src):
        out_c += list(c)
        out_s += [s_] * len(c)
    if len(out_c) == len(whole):
        out_c = list(whole)  # context-sensitive casing (e.g. final sigma) is length-preserving here
    return out_c, out_s


def _t_ws(chars, src):
    s = "".join(chars)
    out_c, out_s, pos = [], [], 0
    for m in re.finditer(r"\s+", s):
        out_c += chars[pos:m.start()]
        out_s += src[pos:m.start()]
        out_c.append(" ")
        out_s.append(src[m.start()])
        pos = m.end()
    out_c += chars[pos:]
    out_s += src[pos:]
    return _t_strip(out_c, out_s)


def clean_text_with_mapping(text: str, base_offset: int = 0) -> CleanResult:
    expected = clean_text(text)
    fired = {"remove_think_tag": False, "remove_ai_header": False}
    try:
        chars, src = list(text), list(range(base_offset, base_offset + len(text)))
        chars, src = _t_demojize(chars, src)
        chars, src = _t_think(chars, src, fired)
        chars, src = _t_header(chars, src, fired)
        chars, src = _t_lower(chars, src)
        chars, src = _t_ws(chars, src)
        got = "".join(chars)
    except Exception as e:  # noqa: BLE001
        return CleanResult(expected, None, "unavailable", f"mapping_error:{type(e).__name__}", fired, None)
    if got != expected:
        return CleanResult(expected, None, "unavailable", "mapping_reconstruction_mismatch", fired, None)
    deleted = _deleted_intervals(text, base_offset, src)
    return CleanResult(expected, src, "exact", None, fired, deleted)


def _deleted_intervals(text: str, base: int, src: list[int]) -> list[tuple[int, int]]:
    """Canonical intervals whose non-whitespace characters do not reach the model input."""
    present = set(src)
    out: list[list[int]] = []
    for i, ch in enumerate(text):
        g = base + i
        if g in present or ch.isspace():
            continue
        if out and out[-1][1] == g:
            out[-1][1] = g + 1
        else:
            # merge across pure-whitespace gaps
            if out and all(text[k - base].isspace() for k in range(out[-1][1], g)):
                out[-1][1] = g + 1
            else:
                out.append([g, g + 1])
    return [(a, b) for a, b in out]
