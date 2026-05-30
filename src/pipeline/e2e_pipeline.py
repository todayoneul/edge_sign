"""
Edge-Sign v2 E2E Pipeline

검출(YOLOv8n-ONNX) + 추적(ByteTrack) + 인식(OCR/분류) 통합 파이프라인.

재활용:
  - src/track/bytetrack.py → ByteTracker
  - web/korean_ocr_quant.onnx  → KoreanOCRNet (char-level, 2350 classes)
  - model_space/yolov8n_signs_*.onnx → YOLOv8n

사용법:
  python src/pipeline/e2e_pipeline.py \\
    --yolo model_space/yolov8n_signs_fp32.onnx \\
    --ocr  web/korean_ocr_quant.onnx \\
    --input data/aihub_traffic/val/

  # 단일 이미지/프레임 디렉토리 테스트
  python src/pipeline/e2e_pipeline.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.track.bytetrack import ByteTracker, STrack

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

# 검출기 클래스 (v3: 신호등 분리). signboard는 이 데모에선 미사용.
CLASS_NAMES = {0: "traffic_sign", 1: "traffic_light", 2: "signboard"}
TEMPORAL_BUFFER_LEN = 8  # 트랙별 인식 결과 누적 프레임 수

# 한국 표지판/신호등 분류기 14클래스 (data/roi_cls/classes.json)
_KCLS = None  # {"names": [...], "sign_ids": [...], "light_ids": [...]}

def _load_kcls() -> dict:
    global _KCLS
    if _KCLS is None:
        p = ROOT / "data" / "roi_cls" / "classes.json"
        if p.exists():
            _KCLS = json.loads(p.read_text(encoding="utf-8"))
        else:
            _KCLS = {"names": [], "sign_ids": [], "light_ids": []}
    return _KCLS

# OCR 인덱스→문자 매핑 (web/idx_to_char.json)
_IDX_TO_CHAR: dict[int, str] | None = None

def _load_idx_to_char() -> dict[int, str]:
    global _IDX_TO_CHAR
    if _IDX_TO_CHAR is None:
        p = ROOT / "data" / "idx_to_char.json"
        if not p.exists():
            p = ROOT / "web" / "idx_to_char.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            _IDX_TO_CHAR = {int(k): v for k, v in raw.items()}
        else:
            _IDX_TO_CHAR = {}
    return _IDX_TO_CHAR


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8n ONNX 후처리
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """box [4] vs boxes [N, 4] → IoU [N]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> np.ndarray:
    """단순 greedy NMS → keep 인덱스 반환."""
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        iou_vals = _iou(boxes[idx], boxes[order])
        for j, i in enumerate(order):
            if i != idx and iou_vals[j] > iou_thresh:
                suppressed[i] = True
    return np.array(keep, dtype=np.int32)


def postprocess_yolo(
    raw_output: np.ndarray,
    input_w: int = 640,
    input_h: int = 640,
    orig_w: int = 640,
    orig_h: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> np.ndarray:
    """
    YOLOv8 ONNX 출력 → [N, 6] (x1,y1,x2,y2,conf,cls) 픽셀 좌표.

    YOLOv8은 num_classes=2이면 출력 shape = [1, 6, 8400]
      6 = cx(0) cy(1) w(2) h(3) cls0(4) cls1(5)
    또는 [1, 8400, 6] 형태도 허용.
    좌표는 입력 이미지(640×640) 픽셀 단위이므로 원본 해상도로 스케일 조정.
    """
    pred = raw_output[0]  # [6, 8400] 또는 [8400, 6]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T       # → [8400, 6]

    # 박스(cx,cy,w,h) + 클래스 점수
    boxes_cxcywh = pred[:, :4]          # [N, 4]
    class_scores = pred[:, 4:]          # [N, num_classes]

    conf = class_scores.max(axis=1)     # [N]
    cls = class_scores.argmax(axis=1)   # [N]

    # 신뢰도 필터
    mask = conf > conf_thres
    if not mask.any():
        return np.empty((0, 6), dtype=np.float32)

    boxes_cxcywh = boxes_cxcywh[mask]
    conf = conf[mask]
    cls = cls[mask]

    # cx,cy,w,h → x1,y1,x2,y2 (입력 이미지 픽셀)
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 원본 이미지 좌표로 스케일 변환
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    xyxy[:, [0, 2]] *= scale_x
    xyxy[:, [1, 3]] *= scale_y
    xyxy = np.clip(xyxy, 0, [orig_w, orig_h, orig_w, orig_h])

    # NMS (클래스별)
    keeps = []
    for c in np.unique(cls):
        mask_c = cls == c
        keep_c = _nms(xyxy[mask_c], conf[mask_c], iou_thres)
        idx = np.where(mask_c)[0][keep_c]
        keeps.append(idx)
    if not keeps:
        return np.empty((0, 6), dtype=np.float32)
    keep = np.concatenate(keeps)

    return np.column_stack([xyxy[keep], conf[keep], cls[keep]]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 인식: KoreanOCRNet (ONNX, 단일 문자 분류)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ocr_roi(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    검출 bbox ROI → KoreanOCRNet 입력 [1, 1, 64, 64].
    ROI를 64×64 흑백으로 리사이즈 후 정규화 [0, 1].
    """
    x1, y1, x2, y2 = bbox[:4].astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, np.newaxis, :, :]   # [1, 1, 64, 64]


def decode_ocr_output(logits: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """OCR 로짓 → [(문자, 확률)] 리스트 (Top-K)."""
    idx_to_char = _load_idx_to_char()
    probs = softmax(logits[0])
    top_indices = probs.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        char = idx_to_char.get(int(idx), f"[{idx}]")
        results.append((char, float(probs[idx])))
    return results


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
# E2E 파이프라인 클래스
# ─────────────────────────────────────────────────────────────────────────────

GTSDB_CLASSES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way junction",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No trucks", "No entry", "General caution", "Dangerous left curve",
    "Dangerous right curve", "Double curve", "Bumpy road", "Slippery road",
    "Narrow right", "Road works", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles", "Beware ice/snow", "Wild animals",
    "End restrictions", "Turn right", "Turn left", "Go straight",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout", "End no passing", "End no passing >3.5t",
]


class EdgeSignPipeline:
    """
    검출(YOLOv8s) + 추적(ByteTrack) + 인식(OCR/분류) 통합 파이프라인.

    Args:
        yolo_onnx:  YOLOv8s ONNX 모델 경로
        ocr_onnx:   KoreanOCRNet ONNX 모델 경로 (signboard 인식용)
        tsign_onnx: TrafficSignNet ONNX 모델 경로 (traffic_sign 분류용)
        conf_thres: 검출 신뢰도 임계값
        iou_thres:  NMS IoU 임계값
        track_thresh: ByteTrack 고신뢰 임계값
    """

    def __init__(
        self,
        yolo_onnx:  Optional[str] = None,
        ocr_onnx:   Optional[str] = None,
        tsign_onnx: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        track_thresh: float = 0.5,
        det_taxonomy: str = "v3",
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime가 설치되지 않았습니다: pip install onnxruntime")

        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self._frame_id  = 0

        # 검출기 클래스 택소노미 — 로드된 검출기 모델에 맞춰야 라우팅이 정확.
        #   v3: 0=traffic_sign, 1=traffic_light, 2=signboard (신호등 분리 모델)
        #   v2: 0=traffic_sign, 1=signboard            (구 모델, 신호등 미분리)
        # 잘못 지정하면 v2 간판이 신호등 색상으로 오분류되므로 app.py에서 명시 전달.
        self.det_taxonomy = det_taxonomy
        if det_taxonomy == "v2":
            self.class_names = {0: "traffic_sign", 1: "signboard"}
            self._ocr_cls = 1
        else:
            self.class_names = {0: "traffic_sign", 1: "traffic_light", 2: "signboard"}
            self._ocr_cls = 2

        # YOLOv8s 세션
        self.yolo_session = None
        if yolo_onnx and Path(yolo_onnx).exists():
            self.yolo_session = ort.InferenceSession(
                str(yolo_onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            inp = self.yolo_session.get_inputs()[0]
            self._yolo_input_name = inp.name
            self._yolo_h = inp.shape[2] if isinstance(inp.shape[2], int) else 640
            self._yolo_w = inp.shape[3] if isinstance(inp.shape[3], int) else 640
            print(f"[Pipeline] YOLOv8s loaded: {Path(yolo_onnx).name}")
        else:
            print(f"[Pipeline] WARNING: YOLOv8s ONNX not found -- detection disabled")

        # KoreanOCRNet 세션
        self.ocr_session = None
        if ocr_onnx and Path(ocr_onnx).exists():
            self.ocr_session = ort.InferenceSession(
                str(ocr_onnx), providers=["CPUExecutionProvider"]
            )
            self._ocr_input_name = self.ocr_session.get_inputs()[0].name
            print(f"[Pipeline] KoreanOCRNet loaded: {Path(ocr_onnx).name}")
        else:
            print(f"[Pipeline] WARNING: OCR ONNX not found -- signboard OCR disabled")

        # 한국 표지판/신호등 분류기 세션 (korean_sign_net 14클래스)
        # 기존 tsign_onnx 인자 재사용 — 한국 분류기 ONNX 경로를 받음.
        self.tsign_session = None
        self._kcls = _load_kcls()
        if tsign_onnx and Path(tsign_onnx).exists():
            self.tsign_session = ort.InferenceSession(
                str(tsign_onnx), providers=["CPUExecutionProvider"]
            )
            self._tsign_input_name = self.tsign_session.get_inputs()[0].name
            print(f"[Pipeline] Korean classifier loaded: {Path(tsign_onnx).name} "
                  f"({len(self._kcls['names'])} classes)")
        else:
            print(f"[Pipeline] INFO: 분류기 ONNX 없음 -- 라벨 = class_name")

        # ByteTracker
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30,
        )

        # 트랙별 인식 결과 누적 버퍼
        # {track_id: deque[(label, confidence)]}
        self._track_buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TEMPORAL_BUFFER_LEN)
        )

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _preprocess_yolo(self, frame: np.ndarray):
        """BGR 프레임 → NCHW float32 [1, 3, H, W] (YOLOv8n 입력)."""
        resized = cv2.resize(frame, (self._yolo_w, self._yolo_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[np.newaxis]  # [1, 3, H, W]

    def _run_yolo(self, frame: np.ndarray) -> np.ndarray:
        """ONNX 추론 → [N, 6] (x1,y1,x2,y2,conf,cls)."""
        if self.yolo_session is None:
            return np.empty((0, 6), dtype=np.float32)
        h, w = frame.shape[:2]
        inp = self._preprocess_yolo(frame)
        raw = self.yolo_session.run(None, {self._yolo_input_name: inp})
        return postprocess_yolo(
            raw[0],
            input_w=self._yolo_w, input_h=self._yolo_h,
            orig_w=w, orig_h=h,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres,
        )

    def _run_ocr(self, frame: np.ndarray, bbox: np.ndarray) -> list[tuple[str, float]]:
        """signboard ROI → Top-3 OCR 문자 리스트."""
        if self.ocr_session is None:
            return []
        inp = preprocess_ocr_roi(frame, bbox)
        if inp is None:
            return []
        out = self.ocr_session.run(None, {self._ocr_input_name: inp})
        return decode_ocr_output(out[0])

    def _run_tsign(self, frame: np.ndarray, bbox: np.ndarray,
                   det_cls: int = 0) -> list[tuple[str, float]]:
        """ROI → 한국 분류기 Top-3 (det_cls로 표지판/신호등 후보 제한).

        det_cls 0(traffic_sign) → sign_ids 서브셋, 1(traffic_light) → light_ids.
        학습과 동일 전처리: (rgb/255 - 0.5)/0.5, 3×32×32.
        """
        # fallback 신뢰도는 낮게 — temporal buffer 점수를 왜곡하지 않도록 (미인식 표시)
        fallback = self.class_names.get(det_cls, "traffic_sign")
        if self.tsign_session is None or not self._kcls["names"]:
            return [(fallback, 0.0)]
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return [(fallback, 0.0)]
        roi = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(cv2.resize(roi, (32, 32)), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        inp = np.transpose(tensor, (2, 0, 1))[np.newaxis]   # [1, 3, 32, 32]
        out = self.tsign_session.run(None, {self._tsign_input_name: inp})
        probs = softmax(out[0][0])

        # 검출 클래스에 맞는 후보만 남김 (신호등↔표지판 혼동 방지)
        names = self._kcls["names"]
        cand = self._kcls["light_ids"] if det_cls == 1 else self._kcls["sign_ids"]
        cand = [i for i in cand if i < len(probs)]
        if not cand:
            cand = list(range(len(probs)))
        cand.sort(key=lambda i: probs[i], reverse=True)
        return [(names[i], float(probs[i])) for i in cand[:3]]

    def _stable_recognition(self, track_id: int) -> str:
        """temporal buffer에서 최빈 레이블(신뢰도 누적 기준) 반환."""
        buf = self._track_buffers[track_id]
        if not buf:
            return ""
        # 레이블별 신뢰도 합산
        score_map: dict[str, float] = defaultdict(float)
        for label, conf in buf:
            score_map[label] += conf
        best = max(score_map, key=lambda k: score_map[k])
        return best

    # ── 메인 인터페이스 ───────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        한 프레임을 처리하여 구조화된 인식 결과를 반환.

        Returns:
            {
              "frame_id": int,
              "ts_ms": float,         # 처리 시작 타임스탬프 (ms)
              "inference_ms": float,  # 추론 소요 시간 (ms)
              "tracks": [
                {
                  "id": int,
                  "class": int,       # v3: 0=traffic_sign 1=traffic_light 2=signboard
                  "class_name": str,
                  "bbox": [x1,y1,x2,y2],
                  "conf": float,
                  "label": str,       # 인식 결과 (temporal 안정화)
                  "top_labels": [(label, conf), ...],  # 현 프레임 Top-3
                }
              ]
            }
        """
        self._frame_id += 1
        t0 = time.perf_counter()

        # 1. YOLOv8n 검출
        dets = self._run_yolo(frame)

        # 2. ByteTrack 추적
        tracks: list[STrack] = self.tracker.update(dets)

        # 3. 인식 + 버퍼 갱신
        result_tracks = []
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr
            cls = int(track.cls)
            bbox = np.array([x1, y1, x2, y2])

            # 라우팅 (택소노미별): signboard(=self._ocr_cls) → OCR,
            #   그 외(표지판/신호등) → 한국 분류기(검출 클래스로 후보 서브셋 제한).
            if cls == self._ocr_cls:
                top_labels = self._run_ocr(frame, bbox)
                label_cur = top_labels[0][0] if top_labels else ""
                conf_cur  = top_labels[0][1] if top_labels else 0.0
            else:
                top_labels = self._run_tsign(frame, bbox, det_cls=cls)
                label_cur  = top_labels[0][0] if top_labels else self.class_names.get(cls, "")
                conf_cur   = top_labels[0][1] if top_labels else float(track.score)

            self._track_buffers[track.track_id].append((label_cur, conf_cur))
            stable = self._stable_recognition(track.track_id)

            result_tracks.append({
                "id":         track.track_id,
                "class":      cls,
                "class_name": self.class_names.get(cls, str(cls)),
                "bbox":       [round(float(x1)), round(float(y1)),
                               round(float(x2)), round(float(y2))],
                "conf":       round(float(track.score), 3),
                "label":      stable,
                "top_labels": [(lbl, round(c, 3)) for lbl, c in top_labels],
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "frame_id":    self._frame_id,
            "ts_ms":       t0 * 1000,
            "inference_ms": round(elapsed_ms, 1),
            "tracks":      result_tracks,
        }

    def reset(self):
        """트래커와 버퍼 초기화 (새 시퀀스 시작 시)."""
        self.tracker.reset()
        self._track_buffers.clear()
        self._frame_id = 0

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        result를 프레임에 오버레이 (bbox + track ID + 라벨).
        디버그/시각화용.
        """
        vis = frame.copy()
        for t in result["tracks"]:
            x1, y1, x2, y2 = t["bbox"]
            cls = t["class"]
            color = (0, 200, 0) if cls == 0 else (0, 100, 255)  # 초록/주황

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label_str = t["label"] or t["class_name"]
            text = f"#{t['id']} {label_str} {t['conf']:.2f}"
            cv2.putText(vis, text, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Edge-Sign E2E 파이프라인 테스트")
    parser.add_argument("--yolo",   default=str(ROOT / "model_space" / "yolov8s_signs_w8a8.onnx"))
    parser.add_argument("--ocr",    default=str(ROOT / "model_space" / "korean_ocr_net_w8a8.onnx"))
    parser.add_argument("--tsign",  default=str(ROOT / "model_space" / "traffic_sign_net_w8a8.onnx"))
    parser.add_argument("--input",  default=None, help="이미지 디렉토리 또는 영상 파일")
    parser.add_argument("--show",   action="store_true", help="결과 시각화 표시")
    parser.add_argument("--dry_run", action="store_true", help="더미 프레임으로 초기화 테스트")
    args = parser.parse_args()

    pipeline = EdgeSignPipeline(
        yolo_onnx=args.yolo,
        ocr_onnx=args.ocr,
        tsign_onnx=args.tsign,
    )

    if args.dry_run:
        print("[DryRun] 640×480 더미 프레임으로 파이프라인 테스트...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(3):
            res = pipeline.process_frame(dummy)
            print(f"  Frame {res['frame_id']}: {res['inference_ms']:.1f}ms, {len(res['tracks'])} tracks")
        print("[DryRun] 완료")
        return

    if args.input is None:
        print("--input 경로를 지정하세요 (이미지 디렉토리 or 영상 파일)")
        sys.exit(1)

    input_path = Path(args.input)

    # 이미지 디렉토리
    if input_path.is_dir():
        imgs = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        print(f"[Info] {len(imgs)}개 이미지 처리 중...")
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            res = pipeline.process_frame(frame)
            print(json.dumps(res, ensure_ascii=False))
            if args.show:
                vis = pipeline.draw(frame, res)
                cv2.imshow("Edge-Sign Pipeline", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # 영상 파일
    elif input_path.is_file():
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"[Info] 영상 처리: {input_path.name} ({fps:.0f}fps)")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = pipeline.process_frame(frame)
            if args.show:
                vis = pipeline.draw(frame, res)
                cv2.imshow("Edge-Sign Pipeline", vis)
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                    break
        cap.release()

    else:
        print(f"[Error] 경로를 찾을 수 없음: {args.input}")
        sys.exit(1)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
