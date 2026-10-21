from app.monitoring import summarise_detections, summarise_predictions


def test_summarise_detections_empty():
    result = summarise_detections([])
    assert result["detection_count"] == 0
    assert result["avg_confidence"] is None
    assert result["class_counts"] == {}


def test_summarise_detections_counts_classes_and_confidence():
    result = summarise_detections([
        {"class_name": "trash", "confidence": 0.8},
        {"class_name": "trash", "confidence": 0.6},
        {"class_name": "garbage_bag", "confidence": 0.4},
    ])
    assert result["detection_count"] == 3
    assert result["avg_confidence"] == 0.6
    assert result["max_confidence"] == 0.8
    assert result["class_counts"] == {"trash": 2, "garbage_bag": 1}


def test_summarise_predictions_low_confidence_rate():
    rows = [
        {"model_type": "garbage", "detection_count": 2, "avg_confidence": 0.8},
        {"model_type": "spill", "detection_count": 1, "avg_confidence": 0.3},
    ]
    result = summarise_predictions(rows, 0.5)
    assert result["window_predictions"] == 2
    assert result["detections_total"] == 3
    assert result["predictions_by_model"] == {"garbage": 1, "spill": 1}
    assert result["mean_prediction_confidence"] == 0.55
    assert result["low_confidence_predictions"] == 1
    assert result["low_confidence_rate"] == 0.5


def test_summarise_predictions_empty():
    result = summarise_predictions([])
    assert result["window_predictions"] == 0
    assert result["mean_prediction_confidence"] is None
    assert result["low_confidence_rate"] is None
