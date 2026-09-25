"""Check the opt-in Space route against the paper decoder and WS contract."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.paper.evaluate_qdq_detection import decode_v4
from src.pipeline.paper_v4 import _decode_v4, register_routes

ROOT = Path(__file__).resolve().parents[2]


def test_v4_decoder_matches_paper_evaluator() -> None:
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [64, 96, 320, 400, 0.8, 1]
    output[0, 1] = [100, 120, 210, 270, 0.4, 0]
    output[0, 2] = [0, 0, 50, 50, np.nan, 0]
    output[0, 3] = [30, 40, 80, 90, 0.9, 3]
    expected, _ = decode_v4(output, width=1280, height=480, min_conf=0.1)
    actual = _decode_v4(output, width=1280, height=480)
    assert len(actual) == len(expected) == 2
    for row, paper in zip(actual, expected, strict=True):
        np.testing.assert_allclose(row[:4], paper["box_xyxy"], atol=1e-5)
        assert row[4] == pytest.approx(paper["confidence"])
        assert int(row[5]) == paper["class_id"]


def test_v4_websocket_reset_and_result() -> None:
    required = [
        ROOT / "model_space/yolo_v4_signs_fp32.onnx",
        ROOT / "model_space/yolo_v4_signs_int8_head_excluded.onnx",
        ROOT / "model_space/korean_sign_net_fp32.onnx",
        ROOT / "data/roi_cls/classes.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("local ignored model artifacts unavailable")
    app = FastAPI()
    register_routes(app, ROOT)
    with TestClient(app) as client:
        status = client.get("/api/paper-v4/status").json()
        assert status["status"] == "ready"
        assert status["models"]["head_excluded_qdq"]["size_bytes"] > 0
        ok, jpg = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
        assert ok
        encoded = base64.b64encode(jpg).decode("ascii")
        with client.websocket_connect("/ws/paper-v4") as socket:
            socket.send_json({"type": "reset"})
            assert socket.receive_json() == {"type": "ack", "message": "reset"}
            socket.send_json({"type": "frame", "data": encoded, "variant": "head_excluded_qdq"})
            result = socket.receive_json()
            assert result["type"] == "result"
            assert result["data"]["frame_id"] == 1
            assert result["data"]["variant"] == "head_excluded_qdq"
            assert set(result["data"]["stage_ms"]) == {"detect", "track", "recognize"}
            socket.send_json({"type": "reset"})
            assert socket.receive_json()["type"] == "ack"
            socket.send_json({"type": "frame", "data": encoded, "variant": "fp32"})
            result = socket.receive_json()
            assert result["type"] == "result"
            assert result["data"]["frame_id"] == 1
            assert result["data"]["variant"] == "fp32"
