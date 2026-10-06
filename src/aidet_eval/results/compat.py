"""Cross-run comparability checks (mirrors the viewer's compatibility warnings)."""

from __future__ import annotations


def compatibility(left: dict, right: dict, protocol_id: str, left_model: str, right_model: str) -> list[str]:
    w = []
    lb, rb = left["benchmark"], right["benchmark"]
    if lb["benchmark_hash"] != rb["benchmark_hash"]:
        w.append("benchmark_version_differs")
    if lb["calibration_membership_sha256"] != rb["calibration_membership_sha256"]:
        w.append("calibration_membership_differs")
    lp = next((p for p in left["protocols"] if p["protocol_id"] == protocol_id), None)
    rp = next((p for p in right["protocols"] if p["protocol_id"] == protocol_id), None)
    if lp is None or rp is None or lp["fingerprint"] != rp["fingerprint"]:
        w.append("protocol_fingerprint_differs_or_missing")
    lm = next((m for m in left["models"] if m["model_id"] == left_model), None)
    rm = next((m for m in right["models"] if m["model_id"] == right_model), None)
    if lm is None or rm is None:
        w.append("model_missing")
    elif lm.get("precision") != rm.get("precision"):
        w.append("precision_or_quantisation_differs")
    if left["data_origin"] != right["data_origin"]:
        w.append("data_origin_differs")
    return w
