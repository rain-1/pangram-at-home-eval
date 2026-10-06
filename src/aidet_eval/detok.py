"""Deterministic, reversible detokenisation of pre-tokenised human corpora (benchmark-v2).

WikiText-103 and the arXiv/PubMed corpus separate punctuation with spaces (WikiText also uses
@-@ / @,@ / @.@ for intra-token hyphens/commas/periods; the scientific corpus is lowercased).
`detok(text, kind)` walks the single-space gaps between tokens and deletes a gap only by a fixed
rule (closing punctuation, clitics, opening brackets, paired straight double quotes on a line,
@x@ markers). Every edit is recorded with its original offset; `restore()` rebuilds the original
from the output + edits and is asserted equal, so the transformation is lossless by construction.
Lines with an odd number of standalone double quotes keep their quote spacing (flagged).
Casing is never changed: the scientific corpus stays lowercase (artefact flag retained).
"""

from __future__ import annotations

import re

DETOK_VERSION = "detok_v2_gap_rules"
CLOSE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'s", "'re", "'ve", "'ll", "'d", "'m", "n't", "'"}
OPEN = {"(", "[", "{", "$", "#"}
MARKERS = {"@-@": "-", "@,@": ",", "@.@": "."}


def detok(text: str, kind: str) -> tuple[str, list[tuple[int, int, str]], list[str]]:
    """Return (detokenised text, edits[(orig_start, orig_end, replacement)], flags)."""
    out: list[str] = []
    edits: list[tuple[int, int, str]] = []
    flags: set[str] = set()
    pos = 0
    for line in text.split("\n"):
        lead = 1 if (kind == "wikitext" and line.startswith(" ")) else 0
        trail = 1 if (kind == "wikitext" and line.endswith(" ") and len(line) > lead) else 0
        if lead:
            edits.append((pos, pos + 1, ""))  # WikiText line-edge formatting space
        core = line[lead:len(line) - trail]
        out.append(_line(core, pos + lead, kind, edits, flags))
        if trail:
            edits.append((pos + len(line) - 1, pos + len(line), ""))
        pos += len(line) + 1
    return "\n".join(out), edits, sorted(flags)


def _line(line: str, base: int, kind: str, edits: list, flags: set) -> str:
    # tokens separated by single spaces; other whitespace runs are left untouched
    toks = [(m.start(), m.end()) for m in re.finditer(r"[^ ]+", line)]
    if not toks:
        return line
    quotes = [i for i, (a, b) in enumerate(toks) if line[a:b] == '"']
    balanced = len(quotes) % 2 == 0
    if quotes and not balanced:
        flags.add("unbalanced_quotes_left_spaced")
    opening_q = {q for j, q in enumerate(quotes) if j % 2 == 0} if balanced else set()
    closing_q = {q for j, q in enumerate(quotes) if j % 2 == 1} if balanced else set()
    res = [line[:toks[0][0]]]
    for i, (a, b) in enumerate(toks):
        tok = line[a:b]
        if tok in MARKERS and kind == "wikitext":
            res.append(MARKERS[tok])
            edits.append((base + a, base + b, MARKERS[tok]))
        else:
            res.append(tok)
        if i + 1 == len(toks):
            res.append(line[b:])
            break
        na, nb = toks[i + 1]
        gap = line[b:na]
        nxt = line[na:nb]
        delete = False
        if gap == " ":
            if kind == "wikitext" and (tok in MARKERS or nxt in MARKERS):
                delete = True
            elif nxt in CLOSE or (i + 1) in closing_q:
                delete = True
            elif tok in OPEN or i in opening_q:
                delete = True
            elif kind == "sci" and nxt == "-" and i + 2 < len(toks) and re.match(r"\w", line[toks[i + 2][0]]) and re.search(r"\w$", tok):
                delete = True  # word - word -> word-word
            elif kind == "sci" and tok == "-" and i > 0 and re.search(r"\w$", line[toks[i - 1][0]:toks[i - 1][1]]) and re.match(r"\w", nxt):
                delete = True
        if delete:
            edits.append((base + b, base + na, ""))
        else:
            res.append(gap)
    return "".join(res)


def restore(detok_text: str, original_len: int, edits: list[tuple[int, int, str]], original: str) -> bool:
    """Rebuild the original from the edit list (applied to original offsets) and compare."""
    rebuilt, pos = [], 0
    for a, b, rep in sorted(edits):
        rebuilt.append(original[pos:a])
        rebuilt.append(rep)
        pos = b
    rebuilt.append(original[pos:])
    return "".join(rebuilt) == detok_text and len(original) == original_len


def detok_checked(text: str, kind: str) -> tuple[str, dict]:
    out, edits, flags = detok(text, kind)
    ok = restore(out, len(text), edits, text)
    if not ok:
        raise AssertionError("detokenisation edit log does not reproduce output")
    return out, {"version": DETOK_VERSION, "n_edits": len(edits), "flags": flags,
                 "edits": [[a, b, r] for a, b, r in edits], "lossless": True}
