"""Small, dependency-light helpers used by API monitoring and tests."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def summarise_predictions(rows: Iterable[dict[str, Any]], low_confidence_threshold: float = 0.5) -> dict[str, Any]:
    """Summarise prediction log rows; this is operational monitoring, not data-drift proof."""
    items = list(rows)
    by_model = Counter(str(row.get("model_type", "unknown")) for row in items)
    detections = sum(int(row.get("detection_count") or 0) for row in items)
    confidence_values = [float(row["avg_confidence"]) for row in items if row.get("avg_confidence") is not None]
    low_confidence = sum(
        1 for row in items
        if row.get("avg_confidence") is not None and float(row["avg_confidence"]) < low_confidence_threshold
    )
    return {
        "window_predictions": len(items),
        "detections_total": detections,
        "predictions_by_model": dict(by_model),
        "mean_prediction_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "low_confidence_predictions": low_confidence,
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_rate": round(low_confidence / len(confidence_values), 4) if confidence_values else None,
        "monitoring_note": "Confidence-based operational indicators only. Statistical data drift requires a reference dataset and feature/image comparison."
    }


def summarise_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate detection-level summary without assuming any domain class meanings."""
    confidence = [float(d["confidence"]) for d in detections]
    class_counts = Counter(str(d["class_name"]) for d in detections)
    return {
        "detection_count": len(detections),
        "avg_confidence": round(sum(confidence) / len(confidence), 4) if confidence else None,
        "max_confidence": round(max(confidence), 4) if confidence else None,
        "class_counts": dict(class_counts),
    }
