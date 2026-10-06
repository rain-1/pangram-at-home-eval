"""Detector whose scores were computed elsewhere (e.g. the 35B MoE on the training Space, too large for a local GPU).

Scores come from the workbench's benchmark/score_benchmark.py, which reads inputs exported by export_inputs.py with exactly
the ids and texts that `predict` builds. Each record carries the sha256 of the text it scored; an input whose current text
hash differs, or that has no record, is returned as an error, never silently filled in.

Spec fields: scores_dir (files `<tag>--<checkpoint>--NNofMM.jsonl` plus `<tag>--<checkpoint>.meta.json`), tag, checkpoint
(file stem, e.g. stage2-epoch2-adapters), score_field (mean_token | max_sentence | max_token).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aidet_eval.models.base import Capabilities, Detector, DetectorInput, DetectorOutput

SEMANTICS = {
    "mean_token": "mean AI probability over non-whitespace tokens (workbench document score)",
    "max_sentence": "maximum sentence score; sentence = mean AI probability of its tokens",
    "max_token": "maximum token AI probability",
}


class PrecomputedDetector(Detector):
    def load(self) -> None:
        d = Path(self.spec["scores_dir"]); stem = f'{self.spec["tag"]}--{self.spec["checkpoint"]}'
        meta = d / f"{stem}.meta.json"
        files = sorted(d.glob(f"{stem}--*of*.jsonl"))
        if not meta.exists() or not files:
            raise FileNotFoundError(f"no precomputed scores for {stem} in {d}")
        self.meta = json.loads(meta.read_text()); self.records: dict[str, dict] = {}
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line); self.records[r["input_id"]] = r

    def capabilities(self) -> Capabilities:
        return Capabilities(score_semantics=SEMANTICS[self.spec["score_field"]], score_range=(0.0, 1.0), native_spans=False, max_input_tokens=510)

    def describe(self) -> dict[str, Any]:
        return {"adapter": "precomputed", "tag": self.spec["tag"], "checkpoint": self.spec["checkpoint"],
                "checkpoint_sha256": getattr(self, "meta", {}).get("checkpoint_sha256"), "scorer": getattr(self, "meta", {}).get("scorer"),
                "score_field": self.spec["score_field"], "score_semantics": SEMANTICS[self.spec["score_field"]]}

    def runtime_info(self) -> dict[str, Any]:
        return {"records": len(getattr(self, "records", {})), "inputs_sha256": getattr(self, "meta", {}).get("inputs_sha256")}

    def predict(self, inputs: list[DetectorInput], truncate: bool) -> list[DetectorOutput]:
        out = []
        for i in inputs:
            r = self.records.get(i.input_id); sha = hashlib.sha256(i.text.encode()).hexdigest()
            if r is None or r.get("text_sha256") != sha:
                out.append(DetectorOutput(i.input_id, "error", error="no precomputed score" if r is None else "text hash mismatch", input_fingerprint=sha))
                continue
            score = r.get(self.spec["score_field"])
            # raw_outputs stays None: the results validator treats raw_outputs as EditLens bucket distributions.
            out.append(DetectorOutput(i.input_id, "ok" if r["status"] == "ok" and score is not None else "error", score=score,
                                      coverage={"truncated": r.get("truncated"), "tokens_scored": r.get("tokens_scored"),
                                                "aux_scores": {k: r.get(k) for k in ("mean_token", "max_sentence", "max_token")}},
                                      input_fingerprint=sha, n_tokens=r.get("n_tokens")))
        return out
