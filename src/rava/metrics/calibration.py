from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CONF_RE = re.compile(
    r"(?:['\"]?confidence['\"]?)\s*[:=]\s*([01](?:\.\d+)?)",
    re.IGNORECASE,
)
PROB_RE = re.compile(r"(?:probability|likelihood)\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)
YES_NO_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
OPTION_RE = re.compile(r"(?:answer|option|choice)\s*[:#]?\s*([A-Ea-e0-9])\b")
_EPS = 1e-6


def parse_confidence(text: str) -> float | None:
    match = CONF_RE.search(text) or PROB_RE.search(text)
    if not match:
        return infer_confidence_from_language(text)
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return min(1.0, max(0.0, value))


def infer_confidence_from_language(text: str) -> float | None:
    lower = text.lower()
    if not lower.strip():
        return None

    high_markers = ["certainly", "definitely", "high confidence", "strongly supported"]
    low_markers = ["uncertain", "not sure", "insufficient evidence", "may", "might"]
    if any(marker in lower for marker in high_markers):
        return 0.8
    if any(marker in lower for marker in low_markers):
        return 0.55
    return None


def expected_calibration_error(
    probs: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> float:
    if not probs or not labels or len(probs) != len(labels):
        return 0.0

    probs_arr = np.array(probs, dtype=float)
    labels_arr = np.array(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (probs_arr >= lo) & (probs_arr < hi if i < n_bins - 1 else probs_arr <= hi)
        if not np.any(in_bin):
            continue
        conf = float(np.mean(probs_arr[in_bin]))
        acc = float(np.mean(labels_arr[in_bin]))
        weight = float(np.mean(in_bin))
        ece += abs(acc - conf) * weight

    return float(ece)


def _infer_correct_from_reference(reference: str, output: str) -> int | None:
    ref = reference.strip().lower()
    out = output.strip().lower()
    if not ref:
        return None

    if ref in {"yes", "no", "maybe"}:
        match = YES_NO_RE.search(out)
        if not match:
            return 0
        return int(match.group(1).lower() == ref)

    if len(ref) == 1 and ref in {"a", "b", "c", "d", "e", "0", "1", "2", "3", "4", "5"}:
        match = OPTION_RE.search(output)
        if match:
            return int(match.group(1).strip().lower() == ref)
        return int(f" {ref} " in f" {out} ")

    return int(ref in out)


@dataclass
class CalibrationDataset:
    confidences: np.ndarray
    labels: np.ndarray
    missing_confidence: int
    fallback_excluded: int
    label_unavailable: int


def extract_confidence_labels(
    records: list[dict[str, Any]],
    skip_missing: bool = True,
    exclude_fallback: bool = True,
) -> CalibrationDataset:
    probs: list[float] = []
    labels: list[int] = []
    missing = 0
    fallback_excluded = 0
    label_unavailable = 0

    for row in records:
        generation_mode = str(row.get("generation_mode", "")).lower()
        if exclude_fallback and "fallback" in generation_mode:
            fallback_excluded += 1
            continue

        conf = row.get("confidence")
        if conf is None:
            conf = parse_confidence(str(row.get("output", "")))
        if conf is None:
            missing += 1
            if skip_missing:
                continue
            conf = 0.5

        correct = row.get("correct")
        if correct is None:
            ref = str(row.get("reference", "")).strip().lower()
            out = str(row.get("output", "")).strip()
            inferred = _infer_correct_from_reference(reference=ref, output=out)
            if inferred is None:
                label_unavailable += 1
                continue
            correct = inferred

        probs.append(float(min(1.0, max(0.0, conf))))
        labels.append(int(correct))

    return CalibrationDataset(
        confidences=np.array(probs, dtype=float),
        labels=np.array(labels, dtype=int),
        missing_confidence=missing,
        fallback_excluded=fallback_excluded,
        label_unavailable=label_unavailable,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_isotonic(confidences: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if len(confidences) == 0:
        return {"method": "isotonic", "params": {"block_upper_bounds": [], "block_values": []}}

    order = np.argsort(confidences)
    x = confidences[order]
    y = labels[order].astype(float)

    blocks: list[dict[str, Any]] = []
    for idx in range(len(x)):
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": 1.0,
                "sum": float(y[idx]),
                "mean": float(y[idx]),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left["weight"] + right["weight"])
            total = float(left["sum"] + right["sum"])
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": weight,
                    "sum": total,
                    "mean": total / weight if weight > 0 else 0.0,
                }
            )

    upper_bounds = [float(x[int(block["end"])]) for block in blocks]
    block_values = [float(min(1.0, max(0.0, block["mean"]))) for block in blocks]
    return {
        "method": "isotonic",
        "params": {
            "block_upper_bounds": upper_bounds,
            "block_values": block_values,
        },
    }


def _fit_platt(confidences: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if len(confidences) == 0:
        return {"method": "platt", "params": {"a": 1.0, "b": 0.0}}

    x = confidences.astype(float)
    y = labels.astype(float)
    a = 1.0
    b = 0.0
    lr = 0.25
    reg = 1e-3

    for _ in range(1200):
        logits = (a * x) + b
        p = _sigmoid(logits)
        grad_a = float(np.mean((p - y) * x) + reg * a)
        grad_b = float(np.mean(p - y))
        a -= lr * grad_a
        b -= lr * grad_b
        if abs(grad_a) + abs(grad_b) < 1e-6:
            break

    return {"method": "platt", "params": {"a": float(a), "b": float(b)}}


def apply_confidence_map(confidence: float | None, mapping: dict[str, Any] | None) -> float | None:
    if confidence is None:
        return None
    value = float(min(1.0, max(0.0, confidence)))
    if not mapping:
        return value

    accepted = mapping.get("accepted")
    if accepted is not None:
        accepted_flag = bool(accepted)
        accepted_identity = bool(mapping.get("accepted_identity", False))
        method_name = str(mapping.get("method", "identity")).strip().lower()
        min_improvement = float(mapping.get("min_improvement", 0.005))
        ece_before = mapping.get("ece_before")
        ece_after = mapping.get("ece_after")
        target = mapping.get("target_ece")
        try:
            before = None if ece_before is None else float(ece_before)
        except (TypeError, ValueError):
            before = None
        try:
            after = None if ece_after is None else float(ece_after)
        except (TypeError, ValueError):
            after = None
        try:
            target_ece = None if target is None else float(target)
        except (TypeError, ValueError):
            target_ece = None
        improved = bool(before is not None and after is not None and (before - after) >= min_improvement)
        meets_target = bool(after is not None and target_ece is not None and after <= target_ece)
        if not (
            (accepted_flag and (improved or meets_target))
            or (method_name == "identity" and accepted_identity)
        ):
            return value

    method = str(mapping.get("method", "identity")).strip().lower()
    params = mapping.get("params", {}) if isinstance(mapping.get("params"), dict) else {}

    if method == "isotonic":
        bounds = [float(v) for v in params.get("block_upper_bounds", [])]
        vals = [float(v) for v in params.get("block_values", [])]
        if not bounds or len(bounds) != len(vals):
            return value
        idx = int(np.searchsorted(np.array(bounds, dtype=float), value, side="left"))
        if idx < 0:
            idx = 0
        if idx >= len(vals):
            idx = len(vals) - 1
        return float(min(1.0, max(0.0, vals[idx])))

    if method == "platt":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        out = float(_sigmoid(np.array([(a * value) + b], dtype=float))[0])
        return float(min(1.0, max(0.0, out)))

    if method == "affine":
        scale = float(params.get("scale", 1.0))
        offset = float(params.get("offset", 0.0))
        out = (scale * value) + offset
        return float(min(1.0, max(0.0, out)))

    return value


def evaluate_confidence_map(
    confidences: np.ndarray,
    labels: np.ndarray,
    mapping: dict[str, Any] | None,
    n_bins: int = 10,
) -> float:
    if len(confidences) == 0:
        return 0.0
    adjusted = [float(apply_confidence_map(float(c), mapping) or 0.0) for c in confidences.tolist()]
    return expected_calibration_error(adjusted, labels.astype(int).tolist(), n_bins=n_bins)


def fit_confidence_map(
    records: list[dict[str, Any]],
    methods: tuple[str, ...] = ("isotonic", "platt", "identity"),
    target_ece: float = 0.12,
    n_bins: int = 10,
    min_samples: int = 30,
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    dataset = extract_confidence_labels(records=records, skip_missing=True, exclude_fallback=True)
    n_samples = int(len(dataset.confidences))

    baseline_ece = (
        expected_calibration_error(
            probs=dataset.confidences.tolist(),
            labels=dataset.labels.astype(int).tolist(),
            n_bins=n_bins,
        )
        if n_samples > 0
        else None
    )

    result: dict[str, Any] = {
        "method": "identity",
        "params": {},
        "accepted": False,
        "accepted_identity": False,
        "target_ece": float(target_ece),
        "min_improvement": float(min_improvement),
        "ece_before": baseline_ece,
        "ece_after": baseline_ece,
        "ece_improvement": None if baseline_ece is None else 0.0,
        "n_samples": n_samples,
        "missing_confidence": dataset.missing_confidence,
        "fallback_excluded": dataset.fallback_excluded,
        "label_unavailable": dataset.label_unavailable,
        "reason": "",
        "selected_method": "identity",
        "selection_reason": "not_fitted",
        "candidates": [],
    }

    if n_samples < int(min_samples):
        result["reason"] = f"insufficient_samples<{int(min_samples)}"
        return result

    method_order = [str(method).strip().lower() for method in methods if str(method).strip()]
    if "identity" not in method_order:
        method_order.append("identity")
    candidates: list[dict[str, Any]] = []
    for method in method_order:
        if method == "identity":
            mapping = {"method": "identity", "params": {}}
        elif method == "isotonic":
            mapping = _fit_isotonic(dataset.confidences, dataset.labels)
        elif method == "platt":
            mapping = _fit_platt(dataset.confidences, dataset.labels)
        else:
            continue

        ece_after = evaluate_confidence_map(
            confidences=dataset.confidences,
            labels=dataset.labels,
            mapping=mapping,
            n_bins=n_bins,
        )
        candidates.append(
            {
                "method": str(mapping.get("method", method)),
                "params": mapping.get("params", {}),
                "ece_after": float(ece_after),
                "improvement": (
                    float(baseline_ece) - float(ece_after)
                    if baseline_ece is not None
                    else None
                ),
            }
        )
    result["candidates"] = candidates

    if not candidates:
        result["reason"] = "no_supported_methods"
        result["selection_reason"] = "no_supported_methods"
        return result

    best = sorted(candidates, key=lambda row: row["ece_after"])[0]
    result["selected_method"] = str(best["method"])
    result["method"] = str(best["method"])
    result["params"] = best["params"]
    result["ece_after"] = float(best["ece_after"])
    if baseline_ece is None:
        improvement = None
    else:
        improvement = float(baseline_ece) - float(result["ece_after"])
    result["ece_improvement"] = improvement

    improved_enough = bool(
        baseline_ece is not None and improvement is not None and improvement >= float(min_improvement)
    )
    within_target = bool(result["ece_after"] <= float(target_ece))
    result["accepted"] = bool(improved_enough or within_target)
    if result["accepted"]:
        result["selection_reason"] = (
            "meets_target"
            if within_target
            else f"improved_by>={float(min_improvement):.3f}"
        )
        result["reason"] = "ok"
        return result

    result["method"] = "identity"
    result["params"] = {}
    result["ece_after"] = baseline_ece
    result["ece_improvement"] = 0.0 if baseline_ece is not None else None
    result["accepted"] = True
    result["accepted_identity"] = True
    result["selected_method"] = "identity"
    result["selection_reason"] = "no_candidate_improved_identity_selected"
    result["reason"] = "identity_fallback"
    return result


def write_confidence_map(path: str | Path, mapping: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def load_confidence_map(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def compute_ece_from_predictions(
    records: list[dict[str, Any]],
    n_bins: int = 10,
    skip_missing: bool = True,
) -> dict[str, float | None]:
    dataset_all = extract_confidence_labels(records, skip_missing=skip_missing, exclude_fallback=True)
    probs_all = dataset_all.confidences.tolist()
    labels_all = dataset_all.labels.astype(int).tolist()

    answered_rows = [
        row
        for row in records
        if not bool(row.get("abstained", False))
        and "fallback" not in str(row.get("generation_mode", "")).lower()
    ]
    dataset_answered = extract_confidence_labels(
        answered_rows,
        skip_missing=skip_missing,
        exclude_fallback=True,
    )
    probs_answered = dataset_answered.confidences.tolist()
    labels_answered = dataset_answered.labels.astype(int).tolist()

    ece_all = (
        expected_calibration_error(probs=probs_all, labels=labels_all, n_bins=n_bins)
        if probs_all
        else None
    )
    ece_answered = (
        expected_calibration_error(probs=probs_answered, labels=labels_answered, n_bins=n_bins)
        if probs_answered
        else None
    )
    if not probs_all:
        logger.warning("ECE skipped: no confidence values available in predictions.")
    if dataset_all.missing_confidence > 0:
        logger.warning("ECE computed with %d records missing confidence.", dataset_all.missing_confidence)

    return {
        "ece": float(ece_answered) if ece_answered is not None else (float(ece_all) if ece_all is not None else None),
        "ece_answered": None if ece_answered is None else float(ece_answered),
        "ece_all": None if ece_all is None else float(ece_all),
        "calibration_samples": float(len(probs_answered)),
        "calibration_samples_answered": float(len(probs_answered)),
        "calibration_samples_all": float(len(probs_all)),
        "missing_confidence": float(dataset_answered.missing_confidence),
        "missing_confidence_all": float(dataset_all.missing_confidence),
        "calibration_fallback_excluded": float(dataset_answered.fallback_excluded),
        "calibration_fallback_excluded_all": float(dataset_all.fallback_excluded),
        "calibration_label_unavailable": float(dataset_answered.label_unavailable),
        "calibration_label_unavailable_all": float(dataset_all.label_unavailable),
    }
