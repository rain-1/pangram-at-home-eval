"""Unit tests for label conventions, spans, preprocessing, grouping, sampling, windows, calibration and metrics."""

from __future__ import annotations

import importlib.util
import json
import random

import numpy as np
import pandas as pd
import pytest

from aidet_eval import metrics as M
from aidet_eval import overlap
from aidet_eval import spans as S
from aidet_eval.adapters.common import hash_rank
from aidet_eval.adapters.hf_sources import _llmtrace_record, mage_authorship, raid_authorship
from aidet_eval.calibrate import threshold_for
from aidet_eval.models.editlens_preprocess import clean_text, clean_text_with_mapping


# ------------------------------------------------------------------ label mappings
def test_mage_label_convention_is_one_human():
    assert mage_authorship(1) == "human"
    assert mage_authorship(0) == "ai"
    with pytest.raises(KeyError):
        mage_authorship(2)


def test_raid_human_rule():
    assert raid_authorship("human") == "human"
    for m in ("gpt4", "chatgpt", "mistral-chat", "Human"):
        assert raid_authorship(m) == "ai"


def test_llmtrace_fill_gaps_maps_to_replacement_with_exact_spans():
    row = {"data_type": "news", "text": "Human start. AI middle. Human end.", "label": "mixed", "prompt_type": "fill_gaps",
           "topic_id": "t1", "model": "gemini-2.5-flash", "ai_char_intervals": "[[13, 23]]"}
    r = _llmtrace_record("llmtrace_detection", "x", "rev", "train", 0, row, detection=True).to_row()
    assert r["authorship"] == "mixed" and r["mix_mechanism"] == "sentence_replacement"
    assert r["domain"] == "news" and r["annotation_completeness"] == "complete"
    assert r["ai_char_spans"] == [[13, 23]] and row["text"][13:23] == "AI middle."
    assert r["human_char_spans"] == [[0, 13], [23, len(row["text"])]]


def test_llmtrace_story_is_not_creative():
    row = {"data_type": "story", "text": "Plot summary.", "label": "human", "prompt_type": None, "topic_id": "t", "model": "human"}
    r = _llmtrace_record("llmtrace_detection", "x", "rev", "train", 0, row, detection=False).to_row()
    assert r["domain"] == "narrative_summary"


# ------------------------------------------------------------------ spans & unicode
def test_tag_removal_reconstructs_offsets():
    tagged = "Hi there. <AI_Start>Generated 👩‍👧 text.</AI_End> Bye.<AI_Start>X</AI_End>"
    text, spans = S.strip_tags_with_offsets(tagged, "<AI_Start>", "</AI_End>")
    assert "<AI" not in text
    assert [text[a:b] for a, b in spans] == ["Generated 👩‍👧 text.", "X"]


@pytest.mark.parametrize("bad", ["<AI_Start>a<AI_Start>b</AI_End>", "a</AI_End>", "<AI_Start>unclosed"])
def test_tag_removal_rejects_malformed(bad):
    with pytest.raises(S.SpanError):
        S.strip_tags_with_offsets(bad, "<AI_Start>", "</AI_End>")


def test_codepoint_offsets_differ_from_utf16_and_bytes():
    text = "a😀éb"
    assert len(text) == 5  # code points
    assert len(text.encode("utf-16-le")) // 2 == 6  # UTF-16 units
    assert text[1:2] == "😀" and text[2:4] == "é"
    S.validate_intervals([[1, 2], [2, 4]], len(text))


@pytest.mark.parametrize("iv", [[[3, 2]], [[0, 0]], [[-1, 2]], [[0, 99]], [[0.5, 2]]])
def test_invalid_intervals_rejected(iv):
    with pytest.raises(S.SpanError):
        S.validate_intervals(iv, 10)


def test_invalid_unicode_rejected():
    with pytest.raises(S.SpanError):
        S.validate_text("bad \ud800 surrogate")


def test_null_vs_empty_spans_and_fraction():
    assert S.ai_fraction(None, 10, True) is None
    assert S.ai_fraction([], 10, True) == 0.0
    assert S.ai_fraction([(0, 5)], 10, False) is None
    assert S.ai_fraction([(0, 5), (3, 8)], 10, True) == pytest.approx(0.8)


def test_crop_removing_all_ai_is_not_positive():
    spans = [(10, 20)]
    cropped = S.crop(spans, 30, 50)
    assert cropped == []
    assert S.rederive_label(cropped, 20, complete=True) == ("human", 0.0)
    assert S.rederive_label(cropped, 20, complete=False) == ("unknown", None)
    part = S.crop(spans, 15, 40)
    assert part == [(0, 5)]
    assert S.rederive_label(part, 25, complete=True)[0] == "mixed"
    assert S.rederive_label(S.crop(spans, 10, 20), 10, complete=True) == ("ai", 1.0)


# ------------------------------------------------------------------ preprocessing
def _upstream():
    spec = importlib.util.spec_from_file_location("up", "third_party/EditLens/scripts/preprocess.py")
    if spec is None or not spec.loader or not __import__("os").path.exists("third_party/EditLens/scripts/preprocess.py"):
        return None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_preprocessing_matches_upstream_and_maps_exactly():
    up = _upstream()
    rnd = random.Random(1)
    alphabet = list("abc XYZ\n\t😀é́İΣς") + ["</think>", "Here", "Abstract", "Title:"]
    for _ in range(2000):
        t = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
        r = clean_text_with_mapping(t, base_offset=100)
        if up is not None:
            assert clean_text(t) == up.clean_text(t)
        assert r.text == clean_text(t)
        assert r.mapping_status == "exact"
        assert all(100 <= s < 100 + len(t) for s in r.src)


def test_preprocessing_header_deletion_is_recorded():
    t = "Abstract\nWe measure things."
    r = clean_text_with_mapping(t)
    assert r.text == "we measure things."
    assert r.steps_applied["remove_ai_header"] is True
    assert r.deleted_intervals == [(0, 8)]


# ------------------------------------------------------------------ grouping / dedup
def test_fingerprint_and_near_duplicates():
    base = " ".join(f"w{i}" for i in range(60))
    near = base.replace("w30", "x30")
    ids, texts = ["a", "b", "c"], [base, near, " ".join(f"z{i}" for i in range(60))]
    # true 5-gram Jaccard = 51/61 = 0.836: exact verification must find it; the v1 MinHash estimate is noisy near 0.8
    pairs, stats = overlap.near_duplicate_pairs(ids, texts, verification="exact_jaccard")
    assert {(a, b) for a, b, _ in pairs} == {("a", "b")}
    clear = base.replace("w59", "x59")  # J = 55/57 -> also found by the estimate
    pairs2, _ = overlap.near_duplicate_pairs(["a", "b"], [base, clear])
    assert {(a, b) for a, b, _ in pairs2} == {("a", "b")}
    assert overlap.fingerprint("Hello, World!") == overlap.fingerprint("hello world")
    uf = overlap.UnionFind()
    uf.union("a", "k1"); uf.union("b", "k1"); uf.union("c", "k2")  # noqa: E702
    assert uf.find("a") == uf.find("b") != uf.find("c")


def test_short_texts_are_not_fingerprint_linked():
    assert overlap.POLICY["min_tokens_for_exact_fingerprint_link"] >= 8
    assert overlap.minhash(overlap.match_tokens("Yes.")) is None


# ------------------------------------------------------------------ sampling & calibration disjointness
def test_hash_rank_is_deterministic_and_seeded():
    assert hash_rank("p", "e1") == hash_rank("p", "e1")
    assert hash_rank("p", "e1") != hash_rank("p", "e1", seed=7)


def test_selector_keeps_calibration_and_evaluation_groups_disjoint():
    from aidet_eval.build import Selector

    rows = [{"example_id": f"e{i}", "leakage_group_id": f"g{i // 2}", "independence_group_id": f"i{i // 2}", "eligibility": "eligible",
             "stratum": i % 3, "eligibility_reasons": None} for i in range(40)]
    pool = pd.DataFrame(rows)
    sel = Selector(pool, 42)
    ev = sel.take(pool, 10, "evaluation", "p_eval", {}, strata=["stratum"])
    ca = sel.take(pool, 10, "calibration", "p_cal", {})
    assert set(ev.leakage_group_id).isdisjoint(set(ca.leakage_group_id))
    sel2 = Selector(pool, 42)
    assert list(sel2.take(pool, 10, "evaluation", "p_eval", {}, strata=["stratum"]).example_id) == list(ev.example_id)


# ------------------------------------------------------------------ windows
class _CharBudget:
    """Fake tokenizer budget: each code point after preprocessing counts as a token (+2 specials)."""

    budgets = {"a": 60, "b": 80}

    def counts(self, texts):
        return {m: [len(clean_text(t)) + 2 for t in texts] for m in self.budgets}

    def fits(self, texts):
        c = self.counts(texts)
        return [all(c[m][i] <= self.budgets[m] for m in c) for i in range(len(texts))]


def test_windows_cover_all_words_fit_budget_and_progress():
    from aidet_eval.windows import make_windows

    text = "short words here " * 30 + "x" * 200 + " tail"
    w1 = make_windows(text, _CharBudget())
    w2 = make_windows(text, _CharBudget())
    assert w1 == w2
    covered = set()
    for w in w1:
        assert _CharBudget().fits([text[w["start"]:w["end"]]])[0]
        covered.update(range(w["start"], w["end"]))
    nonspace = {i for i, ch in enumerate(text) if not ch.isspace()}
    assert nonspace <= covered
    assert any(w["kind"] == "codepoint_split" for w in w1)
    starts = [w["start"] for w in w1]
    assert starts == sorted(starts)


# ------------------------------------------------------------------ thresholds
def test_threshold_ties_and_strict_comparator():
    scores = [0.9, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    t = threshold_for(scores, 0.1)  # k = 1 -> threshold = sorted_desc[1] = 0.8
    assert t["threshold"] == 0.8 and t["false_positives"] == 1 and t["tied_scores_at_threshold"] == 3
    assert t["false_positives"] <= t["k"]


def test_threshold_all_equal_scores_and_empty():
    t = threshold_for([0.5] * 50, 0.01)
    assert t["threshold"] == 0.5 and t["false_positives"] == 0
    assert threshold_for([], 0.01)["status"] == "blocked"
    assert threshold_for([], 0.01)["threshold"] is None


# ------------------------------------------------------------------ metrics: empty / one-class / null-vs-zero
def test_rate_empty_is_null_not_zero():
    m = M.rate(np.array([], bool), np.array([]), denominator_population="x", n_eligible=0, n_unscored=0, higher_is_error=True)
    assert m["value"] is None and m["status"] == "not_applicable" and m["percent"] is None
    m2 = M.rate(np.array([], bool), np.array([]), denominator_population="x", n_eligible=5, n_unscored=5, higher_is_error=True)
    assert m2["value"] is None and m2["status"] == "undefined"
    assert m2["failure_bounds"] == {"lower": 0.0, "upper": 1.0, "n_eligible": 5, "n_unscored": 5, "note": m2["failure_bounds"]["note"]}


def test_zero_rate_is_valid_and_flags_group_bound():
    m = M.rate(np.zeros(100, bool), np.arange(100), denominator_population="x", n_eligible=100, n_unscored=0, higher_is_error=True)
    assert m["value"] == 0.0 and m["percent"] == 0.0 and m["status"] == "ok"
    assert m["ci"]["method"] == "wilson_score" and m["ci"]["upper"] > 0
    assert m["group_level_any_error_upper_95"]["value"] == pytest.approx(1 - 0.05 ** (1 / 100))


def test_clustered_rate_uses_bootstrap_and_flags_degenerate():
    m = M.rate(np.zeros(20, bool), np.repeat(np.arange(5), 4), denominator_population="x", n_eligible=20, n_unscored=0, higher_is_error=True)
    assert m["ci"]["method"] == "cluster_percentile_bootstrap" and m["ci"]["degenerate_all_zero"] is True


def test_failure_bounds():
    m = M.rate(np.array([1, 0, 0, 0], bool), np.arange(4), denominator_population="x", n_eligible=6, n_unscored=2, higher_is_error=False)
    assert m["value"] == 0.25
    assert m["failure_bounds"]["lower"] == pytest.approx(1 / 6) and m["failure_bounds"]["upper"] == pytest.approx(3 / 6)


def test_one_class_ranking_metrics_undefined():
    r = M.ranking_metrics(np.array([0.1, 0.2]), np.array([0, 0]), np.array([1, 2]))
    assert r["auroc"]["value"] is None and r["auroc"]["reason"] == "one_class_slice"
    r2 = M.ranking_metrics(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]), np.arange(4), bootstrap=False)
    assert r2["auroc"]["value"] == 1.0


def test_paired_difference_in_percentage_points():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 0, 0], bool)
    d = M.paired_difference(a, b, np.arange(4))
    assert d["difference_pp"] == pytest.approx(25.0)


def test_metric_rejects_nonfinite():
    with pytest.raises(ValueError):
        M.metric("ok", float("nan"))


# ------------------------------------------------------------------ detector interface
def test_editlens_declares_no_native_spans_and_extent_semantics():
    import yaml

    from aidet_eval.models.base import load_detector

    for spec in yaml.safe_load(open("configs/models.yaml"))["models"]:
        if not spec["adapter"].endswith(":EditLensDetector"):
            continue
        cap = load_detector(spec).capabilities()
        assert cap.native_spans is False
        assert cap.score_semantics == "ordinal_ai_intervention_extent"


def test_vectorised_auroc_bootstrap_equals_naive_group_resampling():
    rng = np.random.default_rng(1)
    n = 200
    y = rng.integers(0, 2, n)
    sc = np.round(rng.normal(y * 0.8, 1, n), 1)  # rounding creates ties
    g = rng.integers(0, 80, n)
    lohi, _, _ = M._auroc_cluster_bootstrap(sc, y, g, reps=200, seed=7)
    order = np.argsort(sc, kind="mergesort")
    s2, y2, g2 = sc[order], y[order], g[order]
    ug, inv = np.unique(g2, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    r = np.random.default_rng(7)
    vals = []
    for _ in range(200):
        ix = np.concatenate([members[p] for p in r.integers(0, len(ug), len(ug))])
        vals.append(M.auroc(s2[ix], y2[ix]))
    assert lohi == pytest.approx((np.percentile(vals, 2.5), np.percentile(vals, 97.5)))


# ------------------------------------------------------------------ detokenisation (benchmark-v2)
def test_detokenisation_is_lossless_and_natural():
    from aidet_eval.detok import detok_checked

    wt = " The song ( 1998 ) was a hit , selling 1 @,@ 000 copies in the U.S. ; it 's well @-@ known . \n"
    out, info = detok_checked(wt, "wikitext")
    assert out == "The song (1998) was a hit, selling 1,000 copies in the U.S.; it's well-known.\n"
    assert info["lossless"] and info["n_edits"] > 0
    sci = "patients ( n = 12 ) had one - third lower risk , i.e. 0.05 ."
    out2, _ = detok_checked(sci, "sci")
    assert out2 == "patients (n = 12) had one-third lower risk, i.e. 0.05."
    odd = 'he said " hello there .'
    out3, info3 = detok_checked(odd, "wikitext")
    assert "unbalanced_quotes_left_spaced" in info3["flags"] and ' " hello' in out3



def test_precomputed_scores_require_matching_text_hash(tmp_path):
    import hashlib

    from aidet_eval.models.base import DetectorInput
    from aidet_eval.models.precomputed import PrecomputedDetector

    (tmp_path / "t--ck.meta.json").write_text(json.dumps({"checkpoint_sha256": "x", "scorer": "fixture"}))
    good = "fixture text one"
    recs = [{"input_id": "a", "status": "ok", "mean_token": 0.7, "text_sha256": hashlib.sha256(good.encode()).hexdigest()},
            {"input_id": "b", "status": "ok", "mean_token": 0.2, "text_sha256": "0" * 64}]
    (tmp_path / "t--ck--00of01.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    d = PrecomputedDetector({"id": "fixture_pre", "adapter": "x", "scores_dir": str(tmp_path), "tag": "t", "checkpoint": "ck", "score_field": "mean_token"})
    d.load()
    out = d.predict([DetectorInput("a", good), DetectorInput("b", "changed text"), DetectorInput("c", "missing")], truncate=False)
    assert out[0].status == "ok" and out[0].score == 0.7
    assert out[1].status == "error" and "mismatch" in out[1].error
    assert out[2].status == "error" and out[2].score is None
