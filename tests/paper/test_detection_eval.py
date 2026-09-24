import numpy as np

from scripts.paper.evaluate_qdq_detection import compute_metrics, decode_v4


def test_decode_v4_scales_and_filters_boxes():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [64, 32, 320, 160, 0.9, 1]
    output[0, 1] = [0, 0, 640, 640, 0.0001, 0]
    detections, dropped = decode_v4(output, width=1280, height=640, min_conf=0.001)
    assert dropped == 0
    assert len(detections) == 1
    assert detections[0]["class_id"] == 1
    assert detections[0]["box_xyxy"] == [128.0, 32.0, 640.0, 160.0]


def test_metrics_perfect_and_false_positive():
    ground_truth = [{"classes": [0], "boxes_xyxy": [[0, 0, 10, 10]]}]
    predictions = [
        [
            {"class_id": 0, "confidence": 0.9, "box_xyxy": [0, 0, 10, 10]},
            {"class_id": 0, "confidence": 0.8, "box_xyxy": [20, 20, 30, 30]},
        ]
    ]
    metrics = compute_metrics(ground_truth, predictions, task_conf=0.25)
    assert np.isclose(metrics["mAP50"], 0.995)  # Ultralytics 101-point trapezoid convention
    assert np.isclose(metrics["mAP50_95"], 0.995)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["detection_count"] == 2


def test_metrics_no_detections():
    metrics = compute_metrics(
        [{"classes": [0], "boxes_xyxy": [[0, 0, 10, 10]]}], [[]], task_conf=0.25
    )
    assert metrics["mAP50"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
