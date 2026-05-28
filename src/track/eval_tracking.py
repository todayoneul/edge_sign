"""
ByteTrack 추적 평가 스크립트 (E0 기준선 FP32).

AI Hub test 시퀀스(연속 프레임 + JSON GT)에서:
  1. ONNX 검출기로 프레임별 bbox 추출
  2. ByteTrack으로 트랙 관리
  3. GT bbox와 IoU 매칭 → 프레임별 TP/FP/FN 계산
  4. pseudo-GT track ID로 IDSW 추정 → MOTA 계산
  5. IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN) 계산

사용법:
  # E0 FP32 기준선
  python src/track/eval_tracking.py --onnx model_space/yolov8s_signs_fp32.onnx

  # 양자화 모델 비교
  python src/track/eval_tracking.py --onnx model_space/yolov8s_signs_w8a8.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
DEFAULT_ONNX = ROOT / "model_space" / "yolov8s_signs_fp32.onnx"
TEST_BASE = ROOT / "data" / "aihub_traffic" / "test"

IOU_THRESH = 0.5   # GT-Pred 매칭 임계값
DET_CONF   = 0.25  # 검출 신뢰도 임계값
NMS_IOU    = 0.45


# ─────────────────────────────────────────────
# 1. ONNX 검출기 래퍼
# ─────────────────────────────────────────────

class OnnxDetector:
    def __init__(self, onnx_path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def preprocess(self, img_bgr: np.ndarray):
        """BGR → RGB → resize → normalize → NCHW"""
        h0, w0 = img_bgr.shape[:2]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        tensor = img.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]
        return tensor, w0, h0

    def _nms(self, boxes, scores, iou_thresh):
        """간단한 NMS."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """반환: [N, 6] = [x1, y1, x2, y2, conf, cls] (원본 이미지 좌표)."""
        tensor, w0, h0 = self.preprocess(img_bgr)
        out = self.sess.run(None, {self.input_name: tensor})[0]  # [1, 6, 8400]
        pred = out[0].T  # [8400, 6]: [cx, cy, w, h, cls0_conf, cls1_conf]

        # bbox 변환 (cx,cy,w,h → x1,y1,x2,y2)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - bw / 2;  y1 = cy - bh / 2
        x2 = cx + bw / 2;  y2 = cy + bh / 2

        cls_ids = pred[:, 4:].argmax(axis=1)
        confs   = pred[:, 4:].max(axis=1)

        mask = confs >= self.conf
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)

        boxes  = np.stack([x1, y1, x2, y2], axis=1)[mask]
        confs  = confs[mask]
        cls_ids = cls_ids[mask]

        # NMS per class
        result = []
        for c in np.unique(cls_ids):
            cidx = cls_ids == c
            keep = self._nms(boxes[cidx], confs[cidx], self.iou)
            for k in keep:
                b = boxes[cidx][k]
                result.append([
                    b[0] / self.imgsz * w0,
                    b[1] / self.imgsz * h0,
                    b[2] / self.imgsz * w0,
                    b[3] / self.imgsz * h0,
                    confs[cidx][k],
                    float(c),
                ])
        return np.array(result, dtype=np.float32) if result else np.zeros((0, 6), dtype=np.float32)


# ─────────────────────────────────────────────
# 2. GT 로딩 + pseudo track ID 생성
# ─────────────────────────────────────────────

def load_gt_sequence(seq_dir: Path):
    """
    JSON 어노테이션 로드 → {frame_idx: [[x1,y1,x2,y2,cls], ...]} 반환.
    클래스: traffic_sign=0, traffic_light=0 (동일 취급), signboard=1
    """
    json_dir = TEST_BASE / "labels" / seq_dir.name
    img_paths = sorted(seq_dir.glob("*.jpg"))

    gt_per_frame = {}
    for idx, img_path in enumerate(img_paths):
        # JSON 파일명 = 이미지 파일명 (확장자만 다름)
        json_path = json_dir / (img_path.stem + ".json")
        boxes = []
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            ann = data.get("annotation", [])
            imsize = data.get("image", {}).get("imsize", [1, 1])
            W, H = imsize[0], imsize[1]
            for a in ann:
                if a.get("class") in ("traffic_sign", "traffic_light"):
                    b = a["box"]  # [x1,y1,x2,y2]
                    boxes.append([b[0], b[1], b[2], b[3], 0])  # cls=0
        gt_per_frame[idx] = boxes
    return img_paths, gt_per_frame


def assign_gt_track_ids(gt_per_frame: dict, iou_thresh: float = 0.4):
    """
    프레임 간 IoU 매칭으로 GT bounding box에 track ID 부여 (pseudo-GT 트랙).
    반환: {frame_idx: [[x1,y1,x2,y2,cls,track_id], ...]}
    """
    gt_tracks = {}
    next_id = 1
    prev_boxes = []   # [(x1,y1,x2,y2,cls,track_id)]

    for fidx in sorted(gt_per_frame.keys()):
        curr_raw = gt_per_frame[fidx]
        curr_assigned = []

        if not prev_boxes or not curr_raw:
            for b in curr_raw:
                curr_assigned.append(b + [next_id])
                next_id += 1
        else:
            prev_arr = np.array([[b[0], b[1], b[2], b[3]] for b in prev_boxes])
            curr_arr = np.array([[b[0], b[1], b[2], b[3]] for b in curr_raw])

            iou_mat = _iou_matrix(curr_arr, prev_arr)  # [N_curr, N_prev]
            used_prev = set()
            assigned_curr = [None] * len(curr_raw)

            # Greedy 매칭 (신뢰도 없으므로 IoU 순서)
            flat = np.argsort(-iou_mat.flatten())
            for idx in flat:
                ci = idx // len(prev_boxes)
                pi = idx % len(prev_boxes)
                if assigned_curr[ci] is not None or pi in used_prev:
                    continue
                if iou_mat[ci, pi] >= iou_thresh:
                    assigned_curr[ci] = prev_boxes[pi][5]  # 이전 track_id 계승
                    used_prev.add(pi)

            for ci, b in enumerate(curr_raw):
                tid = assigned_curr[ci] if assigned_curr[ci] is not None else next_id
                if assigned_curr[ci] is None:
                    next_id += 1
                curr_assigned.append(b + [tid])

        gt_tracks[fidx] = curr_assigned
        prev_boxes = curr_assigned

    return gt_tracks


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [N,4], b: [M,4] → [N,M] IoU matrix."""
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1);  iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2);  iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter + 1e-6)


# ─────────────────────────────────────────────
# 3. MOT 메트릭 계산
# ─────────────────────────────────────────────

def compute_mot_metrics(gt_tracks: dict, pred_tracks: dict, iou_thresh: float = 0.5):
    """
    MOTA, IDF1, HOTA(근사) 계산.
    gt_tracks:   {frame: [[x1,y1,x2,y2,cls,gt_id], ...]}
    pred_tracks: {frame: [[x1,y1,x2,y2,conf,cls,track_id], ...]}
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    # GT→pred 매칭 추적 (id switches 계산용)
    gt_to_pred: dict = {}       # gt_id → last matched pred_id

    # IDF1 계산용
    idtp = 0
    idfp = 0
    idfn = 0

    for fidx in sorted(gt_tracks.keys()):
        gts = gt_tracks.get(fidx, [])
        preds = pred_tracks.get(fidx, [])

        total_gt += len(gts)

        if not gts and not preds:
            continue

        gt_arr   = np.array([[g[0], g[1], g[2], g[3]] for g in gts])   if gts   else np.zeros((0,4))
        pred_arr = np.array([[p[0], p[1], p[2], p[3]] for p in preds]) if preds else np.zeros((0,4))

        if len(gts) == 0:
            total_fp += len(preds)
            idfp += len(preds)
            continue
        if len(preds) == 0:
            total_fn += len(gts)
            idfn += len(gts)
            continue

        iou_mat = _iou_matrix(gt_arr, pred_arr)   # [N_gt, N_pred]
        matched_gt  = set()
        matched_pred = set()

        # Greedy 매칭
        flat = np.argsort(-iou_mat.flatten())
        for idx in flat:
            gi = idx // len(preds)
            pi = idx % len(preds)
            if gi in matched_gt or pi in matched_pred:
                continue
            if iou_mat[gi, pi] >= iou_thresh:
                matched_gt.add(gi)
                matched_pred.add(pi)

                gt_id   = gts[gi][5]
                pred_id = preds[pi][6]

                # ID switch 체크
                prev_pred = gt_to_pred.get(gt_id)
                if prev_pred is not None and prev_pred != pred_id:
                    total_idsw += 1
                gt_to_pred[gt_id] = pred_id

                idtp += 1

        fp = len(preds) - len(matched_pred)
        fn = len(gts)  - len(matched_gt)
        total_fp += fp
        total_fn += fn
        idfp += fp
        idfn += fn

    mota = 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1)
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    # HOTA 근사: DetA × AssA
    det_a = idtp / max(idtp + total_fp + total_fn, 1)     # Detection Accuracy
    ass_a = idtp / max(idtp + total_idsw, 1)               # Association Accuracy (근사)
    hota  = (det_a * ass_a) ** 0.5

    return {
        "MOTA":  round(mota, 4),
        "IDF1":  round(idf1, 4),
        "HOTA":  round(hota, 4),
        "DetA":  round(det_a, 4),
        "GT":    total_gt,
        "FP":    total_fp,
        "FN":    total_fn,
        "IDSW":  total_idsw,
    }


# ─────────────────────────────────────────────
# 4. 메인 평가 루프
# ─────────────────────────────────────────────

def evaluate_sequence(seq_dir: Path, detector: OnnxDetector, tracker, verbose: bool = True):
    """단일 시퀀스 평가 → (metrics, fps)."""
    from src.track import ByteTracker

    img_paths, gt_raw = load_gt_sequence(seq_dir)
    gt_tracks = assign_gt_track_ids(gt_raw)

    pred_tracks = {}
    total_time  = 0.0

    for fidx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        dets = detector.detect(img)
        tracks = tracker.update(dets)
        total_time += time.perf_counter() - t0

        frame_preds = []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            frame_preds.append([x1, y1, x2, y2, t.score, t.cls, t.track_id])
        pred_tracks[fidx] = frame_preds

        if verbose and fidx % 20 == 0:
            gt_n   = len(gt_tracks.get(fidx, []))
            pred_n = len(frame_preds)
            print(f"    frame {fidx:4d}: GT={gt_n}, Pred={pred_n}, tracks={len(tracks)}")

    fps = len(img_paths) / max(total_time, 1e-6)
    metrics = compute_mot_metrics(gt_tracks, pred_tracks)
    return metrics, fps


def run_all_sequences(onnx_path: str, verbose: bool = True):
    from src.track import ByteTracker

    detector = OnnxDetector(str(onnx_path), conf=DET_CONF, iou=NMS_IOU)

    test_seq_dirs = sorted((TEST_BASE / "images").iterdir())
    if not test_seq_dirs:
        print("테스트 시퀀스 없음:", TEST_BASE / "images")
        return

    print(f"\n모델: {Path(onnx_path).name}")
    print(f"테스트 시퀀스: {[d.name for d in test_seq_dirs]}")

    all_metrics = []
    all_fps = []

    for seq_dir in test_seq_dirs:
        print(f"\n[시퀀스] {seq_dir.name} ({len(list(seq_dir.glob('*.jpg')))} 프레임)")

        tracker = ByteTracker(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=5,   # 서브샘플(÷6) 영상이므로 ~5fps
        )

        metrics, fps = evaluate_sequence(seq_dir, detector, tracker, verbose)
        all_metrics.append(metrics)
        all_fps.append(fps)

        print(f"  MOTA={metrics['MOTA']:.4f}  IDF1={metrics['IDF1']:.4f}  "
              f"HOTA={metrics['HOTA']:.4f}  FPS={fps:.1f}")
        print(f"  GT={metrics['GT']}  FP={metrics['FP']}  FN={metrics['FN']}  IDSW={metrics['IDSW']}")

    # 전체 평균
    print("\n" + "="*60)
    print("전체 평균 결과")
    print("="*60)
    avg = {k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4)
           for k in ("MOTA", "IDF1", "HOTA", "DetA", "GT", "FP", "FN", "IDSW")}
    avg_fps = sum(all_fps) / len(all_fps)

    print(f"  MOTA:  {avg['MOTA']}")
    print(f"  IDF1:  {avg['IDF1']}")
    print(f"  HOTA:  {avg['HOTA']}")
    print(f"  DetA:  {avg['DetA']}")
    print(f"  FPS:   {avg_fps:.1f}")
    print(f"  GT:    {avg['GT']}  FP: {avg['FP']}  FN: {avg['FN']}  IDSW: {avg['IDSW']}")

    return avg, avg_fps


def main():
    parser = argparse.ArgumentParser(description="ByteTrack 추적 평가")
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX),
                        help="ONNX 검출기 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값")
    parser.add_argument("--quiet", action="store_true", help="프레임별 출력 생략")
    args = parser.parse_args()

    global DET_CONF
    DET_CONF = args.conf

    run_all_sequences(args.onnx, verbose=not args.quiet)


if __name__ == "__main__":
    main()
