"""Adapter for the Open Pangram EditLens checkpoints (RoBERTa-large and Llama-3.2-3B QLoRA).

Faithful to pangramlabs/EditLens scripts/inference.py @05a588f:
  * K buckets inferred from the checkpoint (config id2label/out_proj for the full RoBERTa
    checkpoint; saved `score.linear.weight` rows for the Llama adapter);
  * tokenizer from the *base model* repo; if it has no pad token, pad = eos, left padding;
  * Llama: NF4 4-bit base with bf16 compute, base `score` replaced by NormedLinear
    (LayerNorm -> bias-free Linear) before PeftModel.from_pretrained; every saved adapter/head
    tensor is checked against the loaded module after loading;
  * evaluation under bf16 autocast (upstream Trainer(bf16=True));
  * score = sum_k softmax(logits)_k * k / (K-1): an ordinal AI-intervention extent signal,
    NOT a calibrated probability of AI presence and NOT an AI-character fraction.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from aidet_eval.models.base import Capabilities, Detector, DetectorInput, DetectorOutput
from aidet_eval.models.editlens_preprocess import PREPROCESS_VERSION, UPSTREAM, clean_text_with_mapping
from aidet_eval.util import fingerprint, sha256_text

SCORE_SEMANTICS = "ordinal_ai_intervention_extent"
REFERENCE = {
    "repo": "https://github.com/pangramlabs/EditLens",
    "commit": "05a588f15d792330ccaf91be8ee4fdb54ce26835",
    "inference_blob_sha": "53b784cd06906260051d3c0c0abf63aa68e7f4fb",
    "train_blob_sha": "16457842d9cd6cb157a525daba2b48230cda38bc",
}


def _merge(idx: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for i in sorted(set(idx)):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


class NormedLinear:  # defined lazily with torch below; placeholder for type checkers
    pass


def _normed_linear_cls():
    import torch

    class _NormedLinear(torch.nn.Module):
        """Identical to EditLens scripts/train.py::NormedLinear."""

        def __init__(self, hidden_size, num_labels, device=None, dtype=None):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
            self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

        def forward(self, x):
            return self.linear(self.norm(x))

    return _NormedLinear


class EditLensDetector(Detector):
    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.model = None
        self.tok = None
        self.n_buckets: int | None = None
        self.device = None
        self.load_report: dict[str, Any] = {}
        self.max_length = int(spec["max_length"])
        self.batch_size = int(spec.get("batch_size", 8))

    # ------------------------------------------------------------------ paths
    def _snap(self, repo: str, rev: str, allow: list[str]) -> Path:
        from aidet_eval.model_snapshots import snapshot

        return snapshot(repo, rev, allow, download=False)

    @property
    def ckpt_path(self) -> Path:
        return self._snap(self.spec["checkpoint"], self.spec["checkpoint_revision"], self.spec["checkpoint_allow"])

    @property
    def base_path(self) -> Path:
        return self._snap(self.spec["base_model"], self.spec["base_revision"], self.spec["base_allow"])

    # -------------------------------------------------------------- tokenizer
    def load_tokenizer(self):
        if self.tok is not None:
            return self.tok
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(self.base_path))
        self.load_report["tokenizer_pad_token_was_none"] = tok.pad_token is None
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.padding_side = "left"
        self.load_report["tokenizer"] = {"class": type(tok).__name__, "pad_token": tok.pad_token,
                                         "padding_side": tok.padding_side, "is_fast": tok.is_fast}
        self.tok = tok
        return tok

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token counts after reference preprocessing, including special tokens, no truncation."""
        from aidet_eval.models.editlens_preprocess import clean_text

        tok = self.load_tokenizer()
        enc = tok([clean_text(t) for t in texts], add_special_tokens=True, truncation=False)
        return [len(x) for x in enc["input_ids"]]

    # ------------------------------------------------------------------ model
    def _infer_buckets(self) -> int:
        if self.spec["loader"] == "full_sequence_classification":
            cfg = json.loads((self.ckpt_path / "config.json").read_text())
            k = cfg.get("num_labels") or len(cfg.get("id2label") or {})
            from safetensors import safe_open

            with safe_open(str(self.ckpt_path / "model.safetensors"), "pt") as f:
                k_w = f.get_slice("classifier.out_proj.weight").get_shape()[0]
            if k != k_w:
                raise RuntimeError(f"config labels {k} != out_proj rows {k_w}")
            return int(k)
        from safetensors import safe_open

        with safe_open(str(self.ckpt_path / "adapter_model.safetensors"), "pt") as f:
            for key in f.keys():
                if "score" in key and "linear.weight" in key:
                    return int(f.get_slice(key).get_shape()[0])
        raise RuntimeError("could not infer n_buckets from adapter score head")

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for the reference precision configuration")
        self.device = torch.device("cuda:0")
        self.load_tokenizer()
        k = self._infer_buckets()
        self.n_buckets = k
        t0 = time.time()
        if self.spec["loader"] == "full_sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(str(self.ckpt_path), dtype=torch.float32)
            missing = self._verify_full_checkpoint(model)
            model.to(self.device)
        elif self.spec["loader"] == "qlora_normed_linear_head":
            from peft import PeftModel
            from transformers import BitsAndBytesConfig

            q = self.spec["precision"]["quantisation"]
            bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                                     bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]))
            base = AutoModelForSequenceClassification.from_pretrained(
                str(self.base_path), num_labels=k, quantization_config=bnb, device_map={"": 0})
            base.config.pad_token_id = self.tok.pad_token_id
            if not (hasattr(base, "score") and isinstance(base.score, torch.nn.Linear)):
                raise RuntimeError("base model has no Linear `score` head to replace")
            hidden = base.config.hidden_size
            dev = next(base.parameters()).device
            base.score = _normed_linear_cls()(hidden, k, device=dev)
            model = PeftModel.from_pretrained(base, str(self.ckpt_path))
            missing = self._verify_adapter(model)
        else:
            raise ValueError(self.spec["loader"])
        model.eval()
        self.model = model
        self.load_report.update(n_buckets=k, load_seconds=round(time.time() - t0, 1), verification=missing)

    def _verify_full_checkpoint(self, model) -> dict:
        import torch
        from safetensors import safe_open

        sd = model.state_dict()
        n, mism, absent = 0, [], []
        with safe_open(str(self.ckpt_path / "model.safetensors"), "pt") as f:
            for key in f.keys():
                n += 1
                # legacy LayerNorm names in the saved checkpoint are renamed by transformers at load
                lk = key[:-6] + ".weight" if key.endswith(".gamma") else key[:-5] + ".bias" if key.endswith(".beta") else key
                if lk not in sd:
                    absent.append(key)
                    continue
                if not torch.equal(sd[lk].float().cpu(), f.get_tensor(key).float()):
                    mism.append(key)
        if mism or [a for a in absent if "position_ids" not in a]:
            raise RuntimeError(f"checkpoint tensors not loaded: mismatched={mism[:5]} absent={absent[:5]}")
        return {"saved_tensors": n, "matched": n - len(absent), "absent_buffers": absent, "head": "classifier.{dense,out_proj}"}

    def _verify_adapter(self, model) -> dict:
        import torch
        from safetensors import safe_open

        loaded = {k: v for k, v in model.state_dict().items()}
        n, matched, mism = 0, 0, []
        with safe_open(str(self.ckpt_path / "adapter_model.safetensors"), "pt") as f:
            for key in f.keys():
                n += 1
                # saved: base_model.model.<path>.lora_A.weight / base_model.model.score.norm.weight
                cands = [k for k in loaded if _adapter_key_match(key, k)]
                if not cands:
                    mism.append((key, "absent"))
                    continue
                t = f.get_tensor(key).float()
                if not any(torch.equal(loaded[c].float().cpu(), t) for c in cands):
                    mism.append((key, "value_mismatch"))
                else:
                    matched += 1
        if mism:
            raise RuntimeError(f"adapter/head tensors not loaded correctly: {mism[:5]} ({len(mism)} total)")
        score_mod = model.base_model.model.score
        mts = getattr(score_mod, "modules_to_save", None)
        head_type = type(mts["default"]).__name__ if mts is not None and "default" in mts else type(score_mod).__name__
        return {"saved_tensors": n, "matched": matched, "head_module": head_type}

    # -------------------------------------------------------------- metadata
    def capabilities(self) -> Capabilities:
        return Capabilities(score_semantics=SCORE_SEMANTICS, score_range=(0.0, 1.0),
                            class_probabilities="ordinal_bucket_distribution", native_spans=False,
                            max_input_tokens=self.max_length)

    def describe(self) -> dict[str, Any]:
        import importlib.metadata as md

        snap = {}
        p = Path(__file__).resolve().parents[3] / "models" / f"{self.model_id}.snapshot.json"
        if p.exists():
            s = json.loads(p.read_text())
            snap = {r["repo"]: {"revision": r["revision"], "files": {f["name"]: f["sha256"] for f in r["files"]}} for r in s["repos"]}
        return {
            "model_id": self.model_id,
            "display_name": self.spec["display_name"],
            "adapter": "aidet_eval.models.editlens:EditLensDetector",
            "adapter_version": "1.0.0",
            "checkpoint": self.spec["checkpoint"],
            "checkpoint_revision": self.spec["checkpoint_revision"],
            "base_model": self.spec["base_model"],
            "base_revision": self.spec["base_revision"],
            "tokenizer": self.spec["tokenizer"],
            "tokenizer_revision": self.spec["base_revision"],
            "file_sha256": snap,
            "loader": self.spec["loader"],
            "head": "RobertaClassificationHead" if self.spec["loader"] == "full_sequence_classification" else "NormedLinear(LayerNorm -> Linear(bias=False))",
            "precision": self.spec["precision"],
            "max_length_including_special_tokens": self.max_length,
            "n_buckets": self.n_buckets,
            "reference_implementation": REFERENCE,
            "preprocessing": {"version": PREPROCESS_VERSION, "upstream": UPSTREAM, "fingerprint": fingerprint(UPSTREAM)},
            "score_semantics": SCORE_SEMANTICS,
            "score_formula": "sum(softmax(logits)[k]*k for k in range(K))/(K-1)",
            "score_range": [0.0, 1.0],
            "score_orientation": "higher_is_more_ai",
            "not_a_calibrated_probability": True,
            "not_an_ai_character_fraction": True,
            "native_spans": False,
            "libraries": {k: _ver(md, k) for k in ("torch", "transformers", "peft", "bitsandbytes", "tokenizers", "emoji", "accelerate")},
            "licence": self.spec.get("licence"),
        }

    def runtime_info(self) -> dict[str, Any]:
        import torch

        return {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "cuda": torch.version.cuda, "python": platform.python_version(), "load_report": self.load_report}

    # --------------------------------------------------------------- predict
    def _forward(self, enc_list: list[dict]) -> np.ndarray:
        import torch

        batch = self.tok.pad({"input_ids": [e["input_ids"] for e in enc_list],
                              "attention_mask": [e["attention_mask"] for e in enc_list]}, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch).logits
        return logits.float().cpu().numpy()

    def predict(self, inputs: list[DetectorInput], truncate: bool) -> list[DetectorOutput]:
        import torch
        from scipy.special import softmax

        assert self.model is not None, "call load() first"
        prepared = []
        outs: dict[str, DetectorOutput] = {}
        for inp in inputs:
            cr = clean_text_with_mapping(inp.text, inp.canonical_offset)
            cov: dict[str, Any] = {
                "mapping_status": cr.mapping_status, "mapping_reason": cr.mapping_reason,
                "preprocessing_steps_fired": cr.steps_applied,
                "deleted_by_preprocessing": [list(x) for x in cr.deleted_intervals] if cr.deleted_intervals is not None else None,
            }
            fp = sha256_text(cr.text)
            if not cr.text.strip():
                outs[inp.input_id] = DetectorOutput(inp.input_id, "abstain", coverage=cov, input_fingerprint=fp,
                                                    error="empty_after_preprocessing")
                continue
            full = self.tok(cr.text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
            n_full = len(full["input_ids"])
            if n_full > self.max_length and not truncate:
                outs[inp.input_id] = DetectorOutput(inp.input_id, "error", coverage=cov, input_fingerprint=fp,
                                                    n_tokens=n_full, error="input_exceeds_token_budget_no_truncation_allowed")
                continue
            enc = self.tok(cr.text, add_special_tokens=True, truncation=True, max_length=self.max_length,
                           return_offsets_mapping=True)
            n_used = len(enc["input_ids"])
            ends = [b for (a, b) in enc["offset_mapping"] if b > a]
            consumed_end = max(ends) if ends else 0
            truncated = n_full > n_used
            cov.update(n_tokens_full=n_full, n_tokens_used=n_used, truncated=truncated,
                       cleaned_chars_total=len(cr.text), cleaned_chars_consumed=consumed_end)
            if cr.src is not None:
                vis = _merge(cr.src[:consumed_end])
                cov["model_visible_canonical"] = vis
                cov["first_unread_canonical"] = (cr.src[consumed_end] if truncated and consumed_end < len(cr.src) else None)
            else:
                cov["model_visible_canonical"] = None
            prepared.append((inp, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, cov, fp, n_used))

        # adaptive batching: sort by length; halve batch on OOM; never alter inputs
        prepared.sort(key=lambda x: x[4])
        i, bs = 0, self.batch_size
        while i < len(prepared):
            chunk = prepared[i:i + bs]
            t0 = time.time()
            try:
                logits = self._forward([c[1] for c in chunk])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    inp, _, cov, fp, n = chunk[0]
                    outs[inp.input_id] = DetectorOutput(inp.input_id, "error", coverage=cov, input_fingerprint=fp,
                                                        n_tokens=n, error="cuda_oom_at_batch_size_1")
                    i += 1
                    bs = self.batch_size
                else:
                    bs = max(1, bs // 2)
                continue
            dt = (time.time() - t0) / len(chunk)
            probs = softmax(logits.astype(np.float64), axis=1)
            k = probs.shape[1]
            scores = probs @ np.arange(k) / (k - 1)
            for (inp, _, cov, fp, n), lg, pr, sc in zip(chunk, logits, probs, scores):
                outs[inp.input_id] = DetectorOutput(
                    inp.input_id, "ok", score=float(sc),
                    raw_outputs={"logits": [float(x) for x in lg], "bucket_probabilities": [float(x) for x in pr],
                                 "predicted_bucket": int(np.argmax(pr)), "n_buckets": int(k)},
                    native_spans=None, coverage=cov, input_fingerprint=fp, n_tokens=n, seconds=dt)
            i += len(chunk)
        return [outs[x.input_id] for x in inputs]


def _ver(md, name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def _adapter_key_match(saved: str, loaded: str) -> bool:
    """Map a saved PEFT key to loaded state-dict keys (adapter name 'default', modules_to_save copy)."""
    s = saved.replace("base_model.model.", "", 1)
    l_ = loaded.replace("base_model.model.", "", 1)
    if ".lora_A." in s or ".lora_B." in s:
        return l_ == s.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default.")
    # head saved as score.norm.weight -> loaded score.modules_to_save.default.norm.weight
    head, rest = s.split(".", 1)
    return l_ == f"{head}.modules_to_save.default.{rest}"
