"""Opt-in YOLO26 paper measurement route; the public v3 demo stays separate.

The detector parser and tracking settings mirror scripts/paper/evaluate_pipeline.py.
This route is enabled only with EDGE_SIGN_PAPER_V4=1 and does not serve a UI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mmap
import os
import platform
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.track.bytetrack import ByteTracker

DETECTORS = {
    "fp32": "yolo_v4_signs_fp32.onnx",
    "head_excluded_qdq": "yolo_v4_signs_int8_head_excluded.onnx",
}
RECOGNIZER = "korean_sign_net_fp32.onnx"
CLASS_MAP = "data/roi_cls/classes.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    resident_pages = int(statm.read_text(encoding="ascii").split()[1])
    return resident_pages * mmap.PAGESIZE


def _decode_v4(output: np.ndarray, width: int, height: int) -> np.ndarray:
    """YOLO26 [1,300,6] xyxy/conf/class; built-in top-300, no extra NMS."""
    if output.shape != (1, 300, 6):
        raise ValueError(f"unexpected YOLO26 output shape: {output.shape}")
    rows = output[0]
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 4] >= 0.1)
    rows = rows[valid].copy()
    if not len(rows):
        return np.empty((0, 6), dtype=np.float32)
    cls = np.rint(rows[:, 5])
    valid_class = np.isin(cls, (0, 1)) & (np.abs(rows[:, 5] - cls) <= 0.01)
    rows = rows[valid_class]
    if not len(rows):
        return np.empty((0, 6), dtype=np.float32)
    rows[:, [0, 2]] *= width / 640
    rows[:, [1, 3]] *= height / 640
    rows[:, :4] = np.clip(rows[:, :4], 0, [width, height, width, height])
    rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
    rows[:, 5] = np.rint(rows[:, 5])
    return cast(np.ndarray, rows.astype(np.float32, copy=False))


class PaperV4Service:
    def __init__(self, root: Path, threads: int):
        self.root = root
        self.threads = threads
        self.paths = {name: root / "model_space" / filename for name, filename in DETECTORS.items()}
        self.recognizer_path = root / "model_space" / RECOGNIZER
        self.classes_path = root / CLASS_MAP
        required = [*self.paths.values(), self.recognizer_path, self.classes_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"paper v4 artifacts missing: {missing}")
        self.classes = json.loads(self.classes_path.read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.detectors = {
            name: ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            for name, path in self.paths.items()
        }
        self.recognizer = ort.InferenceSession(
            str(self.recognizer_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        for name, session in self.detectors.items():
            if session.get_outputs()[0].shape != [1, 300, 6]:
                raise ValueError(f"{name} is not a YOLO26 top-300 export")
        self.models = {
            name: {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in self.paths.items()
        }
        self.models["recognizer"] = {
            "filename": self.recognizer_path.name,
            "size_bytes": self.recognizer_path.stat().st_size,
            "sha256": _sha256(self.recognizer_path),
        }

    def status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "generation": "YOLO26-n v4",
            "onnxruntime_version": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
            "threads": self.threads,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "rss_bytes": _rss_bytes(),
            "models": self.models,
            "input_resolution": [640, 640],
            "decoder": "YOLO26 xyxy/conf/class top-300; no external NMS; min_conf=0.1",
            "tracking": {
                "track_thresh": 0.25,
                "match_thresh": 0.8,
                "low_thresh": 0.1,
                "track_buffer": 30,
            },
            "recognition": "KoreanSignNet 14-class, predicted track ROI, coarse-class candidates",
        }

    def new_session(self) -> PaperV4Session:
        return PaperV4Session(self)


class PaperV4Session:
    def __init__(self, service: PaperV4Service):
        self.service = service
        self.reset()

    def reset(self) -> None:
        self.tracker = ByteTracker(
            track_thresh=0.25,
            match_thresh=0.8,
            low_thresh=0.1,
            track_buffer=30,
            frame_rate=30,
        )
        self.frame_id = 0

    def _recognize(
        self, frame: np.ndarray, box: np.ndarray, coarse_class: int
    ) -> tuple[int, str, float] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        rgb = cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (32, 32)), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        inp = np.transpose(tensor, (2, 0, 1))[None]
        session = self.service.recognizer
        logits = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
        classes = self.service.classes
        candidates = classes["sign_ids"] if coarse_class == 0 else classes["light_ids"]
        chosen = max(candidates, key=lambda idx: float(logits[idx]))
        exp = np.exp(logits - logits.max())
        confidence = float(exp[chosen] / exp.sum())
        return chosen, classes["names"][chosen], confidence

    def process_frame(self, frame: np.ndarray, variant: str) -> dict[str, Any]:
        if variant not in self.service.detectors:
            raise ValueError(f"unknown paper v4 variant: {variant}")
        self.frame_id += 1
        started = time.perf_counter()
        height, width = frame.shape[:2]
        resized = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]
        session = self.service.detectors[variant]
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        detections = _decode_v4(raw, width, height)
        detected = time.perf_counter()
        tracks = self.tracker.update(detections)
        tracked = time.perf_counter()
        outputs = []
        for track in tracks:
            recognized = self._recognize(frame, track.tlbr, int(track.cls))
            if recognized is None:
                continue
            outputs.append(
                {
                    "id": int(track.track_id),
                    "class": int(track.cls),
                    "class_name": "traffic_sign" if int(track.cls) == 0 else "traffic_light",
                    "bbox": [float(value) for value in track.tlbr],
                    "conf": float(track.score),
                    "fine_class": recognized[0],
                    "label": recognized[1],
                    "recognition_confidence": recognized[2],
                }
            )
        completed = time.perf_counter()
        return {
            "frame_id": self.frame_id,
            "variant": variant,
            "generation": "YOLO26-n v4",
            "inference_ms": (completed - started) * 1000,
            "stage_ms": {
                "detect": (detected - started) * 1000,
                "track": (tracked - detected) * 1000,
                "recognize": (completed - tracked) * 1000,
            },
            "detection_count": len(detections),
            "track_count": len(outputs),
            "tracks": outputs,
        }


def register_routes(app: FastAPI, root: Path) -> None:
    service: PaperV4Service | None = None
    load_error: str | None = None

    @app.on_event("startup")
    def _load_paper_v4() -> None:
        nonlocal service, load_error
        try:
            threads = int(os.environ.get("EDGE_SIGN_PAPER_V4_THREADS", "2"))
            if threads < 1:
                raise ValueError("EDGE_SIGN_PAPER_V4_THREADS must be positive")
            service = PaperV4Service(root, threads)
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"

    @app.get("/api/paper-v4/status")
    def paper_v4_status() -> dict[str, Any]:
        return (
            service.status()
            if service is not None
            else {"status": "unavailable", "error": load_error}
        )

    @app.websocket("/ws/paper-v4")
    async def paper_v4_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        if service is None:
            await websocket.send_json(
                {"type": "error", "message": load_error or "paper v4 unavailable"}
            )
            await websocket.close()
            return
        session = service.new_session()
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                if msg.get("type") == "reset":
                    session.reset()
                    await websocket.send_json({"type": "ack", "message": "reset"})
                    continue
                if msg.get("type") != "frame":
                    continue
                variant = msg.get("variant", "head_excluded_qdq")
                if variant not in service.detectors:
                    await websocket.send_json(
                        {"type": "error", "message": f"unknown variant: {variant}"}
                    )
                    continue
                try:
                    encoded = msg.get("data", "").split(",", 1)[-1]
                    frame = cv2.imdecode(
                        np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None:
                        raise ValueError("JPEG decode failed")
                    result = session.process_frame(frame, variant)
                    await websocket.send_json({"type": "result", "data": result})
                except (ValueError, TypeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            return
