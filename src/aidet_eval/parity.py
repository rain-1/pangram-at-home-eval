"""Implementation parity: aidet_eval adapter vs pinned upstream EditLens inference.

Runs upstream scripts/inference.py::run_inference (commit 05a588f) in-process on a fixed
set of edge-case texts, with `wandb`, `hydra`, `omegaconf` and `torchmetrics` replaced by
inert stubs so importing train.py (for NormedLinear) cannot start training or telemetry.
Compares bucket logits/scores against our adapter under the same precision/quantisation,
at batch size 1 and batched. This is implementation parity only; no thresholds are tuned.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml

from aidet_eval import paths
from aidet_eval.util import log_event, utcnow, write_json

LONG = " ".join(f"Sentence number {i} describes the harbour, the weather and a ship named Aurora." for i in range(120))
PARITY_TEXTS = [
    ("short", "Thanks, see you tomorrow."),
    ("plain", "The committee met on Tuesday to discuss the budget. Several members raised concerns about the cost of the new library, which had grown by forty percent since the original estimate."),
    ("abstract_header", "Abstract\nWe propose a method for estimating the thermal conductivity of thin films using pump-probe spectroscopy. Results agree with simulations to within 5%."),
    ("title_header", "Title: A Night in Lisbon\n\nThe tram rattled down the hill while the fado singer's voice drifted out of an open window."),
    ("here_header", "Here is the revised paragraph you asked for:\n\nOur findings suggest that early intervention reduces long-term costs, although the effect size varies by region."),
    ("sure_single_para", "Sure, the meeting is at noon and I will bring the slides."),
    ("emoji", "Loved this place 😍🍕!! The crust was perfect 👌 and the staff were lovely. Would come back 💯"),
    ("zwj_emoji", "Our family 👨‍👩‍👧 went hiking; the view from the summit 🏔️ was unbelievable."),
    ("think", "<think>The user wants a summary.</think>\nThe report finds that rainfall decreased by 12% over the decade."),
    ("mixed_prose", "I grew up near the coast and spent summers fixing boats with my uncle. In today's rapidly evolving landscape, maritime craftsmanship represents a vital intersection of tradition and innovation, fostering resilience across communities."),
    ("unicode", "Café owners in São Paulo — and in Zürich — reported ΣΟΦΟΣ-style signage; naïve coöperation İstanbul ﬁnal."),
    ("whitespace", "  Line one.\r\n\r\n\tLine   two with   spaces.\n\n\nLine three.  "),
    ("long_512_to_1024_tokens", " ".join(LONG.split(". ")[:45]) + "."),
    ("long_over_1024_tokens", LONG),
]


def _install_stubs() -> None:
    """Inert stand-ins for training-only imports of train.py. Import transformers first."""
    import importlib.machinery

    import transformers  # noqa: F401  (resolve Trainer & integrations before stubbing)
    from transformers import Trainer, TrainingArguments  # noqa: F401

    def _fail(*a, **k):
        raise RuntimeError("stubbed training/telemetry dependency invoked during parity run")

    for name in ("wandb", "hydra", "omegaconf", "torchmetrics"):
        if name in sys.modules and not getattr(sys.modules[name], "_aidet_stub", False):
            continue
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        m._aidet_stub = True
        m.init = m.log = m.finish = _fail
        sys.modules[name] = m
    sys.modules["hydra"].main = lambda *a, **k: (lambda f: f)
    sys.modules["omegaconf"].DictConfig = object
    sys.modules["omegaconf"].OmegaConf = types.SimpleNamespace(to_container=_fail)
    sys.modules["torchmetrics"].F1Score = _fail
    sys.modules["torchmetrics"].Accuracy = _fail


def _upstream_run(ckpt: str, base: str, max_length: int, texts: list[str], tmp: Path) -> tuple[np.ndarray, np.ndarray]:
    _install_stubs()
    sys.path.insert(0, str(paths.ROOT / "third_party" / "EditLens" / "scripts"))
    try:
        import importlib

        inference = importlib.import_module("inference")
        from datasets import Dataset

        ds_path = tmp / "parity_ds"
        Dataset.from_dict({"text": texts}).save_to_disk(str(ds_path))
        # capture raw logits as well as upstream's derived columns
        captured = {}
        orig_predict = inference.Trainer.predict

        def spy(self, *a, **k):
            out = orig_predict(self, *a, **k)
            captured["logits"] = np.asarray(out.predictions, dtype=np.float64)
            return out

        inference.Trainer.predict = spy
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ds = inference.run_inference(ckpt, base, str(ds_path), text_col="text", max_length=max_length)
        finally:
            os.chdir(cwd)
            inference.Trainer.predict = orig_predict
        tag = Path(base).name if "/" not in base else base.split("/")[-1]
        score_col = [c for c in ds.column_names if c.endswith("_score")][0]
        return captured["logits"], np.asarray(ds[score_col], dtype=np.float64)
    finally:
        sys.path.pop(0)


def run_parity(model_ids: list[str] | None = None) -> int:
    import torch

    from aidet_eval.models.base import DetectorInput, load_detector

    specs = yaml.safe_load(open(paths.CONFIGS / "models.yaml"))["models"]
    report = {"created_at": utcnow(), "scope": "implementation parity on a fixed 14-text edge-case set; not a performance test",
              "texts": [{"id": i, "chars": len(t)} for i, t in PARITY_TEXTS], "models": []}
    rc = 0
    for spec in specs:
        if model_ids and spec["id"] not in model_ids:
            continue
        det = load_detector(spec)
        det.load()
        texts = [t for _, t in PARITY_TEXTS]
        inputs = [DetectorInput(i, t) for i, t in PARITY_TEXTS]
        det.batch_size = 1
        single = det.predict(inputs, truncate=True)
        det.batch_size = len(inputs)
        batched = det.predict(inputs, truncate=True)
        ckpt, base = str(det.ckpt_path), str(det.base_path)
        del det
        torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as td:
            up_logits, up_scores = _upstream_run(ckpt, base, spec["max_length"], texts, Path(td))
        torch.cuda.empty_cache()
        ours1 = np.array([o.raw_outputs["logits"] for o in single])
        oursb = np.array([o.raw_outputs["logits"] for o in batched])
        s1 = np.array([o.score for o in single])
        sb = np.array([o.score for o in batched])
        tol_score = 0.02
        rec = {
            "model_id": spec["id"],
            "precision": spec["precision"],
            "upstream_batch_size": 4 if spec["loader"] == "qlora_normed_linear_head" else 24,
            "max_abs_logit_diff_single_vs_upstream": float(np.max(np.abs(ours1 - up_logits))),
            "max_abs_logit_diff_batched_vs_upstream": float(np.max(np.abs(oursb - up_logits))),
            "max_abs_score_diff_single_vs_upstream": float(np.max(np.abs(s1 - up_scores))),
            "max_abs_score_diff_batched_vs_upstream": float(np.max(np.abs(sb - up_scores))),
            "max_abs_score_diff_single_vs_batched": float(np.max(np.abs(s1 - sb))),
            "bucket_argmax_agreement_single_vs_upstream": int(np.sum(ours1.argmax(1) == up_logits.argmax(1))),
            "n": len(texts),
            "score_tolerance": tol_score,
            "tolerance_rationale": "bf16 autocast (and NF4 for Llama) makes logits depend on padding/batch composition "
                                   "(observed: RoBERTa batched == upstream exactly, single-item differs by ~0.01); "
                                   "the tolerance bounds score differences, not bit equality",
            "per_text": [{"id": i, "ours_single": float(a), "ours_batched": float(b), "upstream": float(c)}
                         for (i, _), a, b, c in zip(PARITY_TEXTS, s1, sb, up_scores)],
        }
        rec["pass"] = bool(rec["max_abs_score_diff_single_vs_upstream"] <= tol_score
                           and rec["max_abs_score_diff_batched_vs_upstream"] <= tol_score)
        rc |= 0 if rec["pass"] else 1
        report["models"].append(rec)
        log_event("parity", model=spec["id"], passed=rec["pass"],
                  max_score_diff=rec["max_abs_score_diff_single_vs_upstream"])
    write_json(paths.ARTIFACTS / "parity" / "parity_report.json", report)
    return rc
