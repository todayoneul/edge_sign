"""
v2 검출 결과 정성적 시각화 — 실제 test 시퀀스에서 v2 모델 추론 + bbox 오버레이.

생성: assets/v2/detection_samples.png  (2x3 그리드)
사용법: python scripts/plot_detection_samples.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "assets" / "v2"
ASSETS.mkdir(parents=True, exist_ok=True)

YOLO_FP32 = ROOT / "model_space" / "yolov8s_signs_fp32.onnx"
YOLO_W8A8 = ROOT / "model_space" / "yolov8s_signs_w8a8.onnx"
TEST_DIR  = ROOT / "data" / "aihub_traffic" / "test" / "images"

CLASS_NAMES  = {0: "traffic_sign", 1: "signboard"}
CLASS_COLORS = {0: "#FF4444", 1: "#44DD44"}

CONF_THRESH = 0.25
NMS_IOU     = 0.45
IMGSZ       = 640


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 추론
# ─────────────────────────────────────────────────────────────────────────────
class OnnxDetector:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _nms(boxes, scores, iou_th):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_th]
        return keep

    def detect(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMGSZ, IMGSZ))
        tensor = (img.astype(np.float32)/255.0).transpose(2,0,1)[np.newaxis]
        out = self.sess.run(None, {self.input_name: tensor})[0][0].T

        cx, cy, bw, bh = out[:,0], out[:,1], out[:,2], out[:,3]
        x1 = cx - bw/2; y1 = cy - bh/2
        x2 = cx + bw/2; y2 = cy + bh/2
        cls_ids = out[:, 4:].argmax(axis=1)
        confs   = out[:, 4:].max(axis=1)

        mask = confs >= CONF_THRESH
        if not mask.any():
            return []
        boxes  = np.stack([x1,y1,x2,y2], axis=1)[mask]
        confs  = confs[mask]
        cls_ids = cls_ids[mask]

        result = []
        for c in np.unique(cls_ids):
            cidx = cls_ids == c
            keep = self._nms(boxes[cidx], confs[cidx], NMS_IOU)
            for k in keep:
                b = boxes[cidx][k]
                result.append({
                    "x1": float(b[0]/IMGSZ * w0),
                    "y1": float(b[1]/IMGSZ * h0),
                    "x2": float(b[2]/IMGSZ * w0),
                    "y2": float(b[3]/IMGSZ * h0),
                    "conf": float(confs[cidx][k]),
                    "cls":  int(c),
                })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(ax, img_bgr, detections, title):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    for d in detections:
        w = d["x2"] - d["x1"]; h = d["y2"] - d["y1"]
        color = CLASS_COLORS[d["cls"]]
        rect = mpatches.Rectangle(
            (d["x1"], d["y1"]), w, h,
            linewidth=1.8, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label = f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        ax.text(d["x1"], max(d["y1"]-5, 12), label,
                fontsize=7, color="white",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.85))

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def pick_sample_frames():
    """주간/야간 test 시퀀스에서 객체가 보이는 프레임 선택."""
    samples = []
    for seq in sorted(TEST_DIR.iterdir()):
        jpgs = sorted(seq.glob("*.jpg"))
        if not jpgs:
            continue
        # 중간/3분의 1 지점에서 프레임 선택
        picks = [jpgs[len(jpgs)//3], jpgs[len(jpgs)*2//3]] if len(jpgs) >= 3 else [jpgs[0]]
        for p in picks[:2]:
            domain = "Night" if "night" in seq.name else "Daylight"
            samples.append((p, domain))
        if len(samples) >= 3:
            break
    return samples[:3]


def main():
    print(f"Generating detection samples → {ASSETS}/")
    if not YOLO_W8A8.exists():
        print(f"[FATAL] {YOLO_W8A8} not found"); sys.exit(1)
    if not YOLO_FP32.exists():
        print(f"[FATAL] {YOLO_FP32} not found"); sys.exit(1)

    detector_fp32 = OnnxDetector(YOLO_FP32)
    detector_w8a8 = OnnxDetector(YOLO_W8A8)

    samples = pick_sample_frames()
    if not samples:
        print("[FATAL] no test frames"); sys.exit(1)

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(n*4.5, 6.5))
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        "v2 Detection Samples — E0 FP32 vs. E1 W8A8  (AI Hub test sequences)",
        fontsize=13, fontweight="bold", y=1.0,
    )

    for col, (img_path, domain) in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # 시각화를 위해 1280 폭으로 리사이즈
        h, w = img.shape[:2]
        if w > 1280:
            scale = 1280 / w
            img = cv2.resize(img, (1280, int(h*scale)))

        dets_fp = detector_fp32.detect(img)
        dets_q  = detector_w8a8.detect(img)

        draw_overlay(axes[0, col], img, dets_fp,
                     f"E0 FP32  |  {domain}  |  {len(dets_fp)} detections")
        draw_overlay(axes[1, col], img, dets_q,
                     f"E1 W8A8  |  {domain}  |  {len(dets_q)} detections")

    # 범례
    handles = [
        mpatches.Patch(facecolor="#FF4444", edgecolor="black", label="traffic_sign"),
        mpatches.Patch(facecolor="#44DD44", edgecolor="black", label="signboard"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = ASSETS / "detection_samples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out.name}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
