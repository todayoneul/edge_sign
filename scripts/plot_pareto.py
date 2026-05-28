"""
Edge-Sign Phase 2 Pareto Frontier 시각화.

실험 E0-E7 결과 기반 Pareto 차트 생성.
출력: assets/pareto_frontier.png

사용법:
  python scripts/plot_pareto.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# 실험 데이터
# 모델 크기: 이론적 INT 배포 크기 (fake-quant ONNX 파일 크기 아님)
#   YOLOv8s FP32=42.67 MB / W8A8=10.67 MB / W4A16=5.33 MB
#   KoreanOCRNet FP32=2.75 MB / W8A8=0.69 MB / W4A16=0.34 MB / 1-Bit=0.09 MB
#   TrafficSignNet FP32=0.12 MB / W8A8=0.03 MB / W4A16=0.015 MB / 1-Bit=0.004 MB
#   SimpleReIDNet W8A8=0.06 MB (theoretical)
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "E0": {
        "size": 42.67 + 2.75 + 0.12,          # 45.54 MB
        "mota": 0.219, "ocr": 98.5, "ts": 62.8, "fps": 21.6,
        "tracker": "ByteTrack",
        "label": "E0\nFP32 All",
        "color": "#4C72B0", "marker": "o",
    },
    "E1": {
        "size": 10.67 + 2.75 + 0.12,          # 13.54 MB
        "mota": 0.221, "ocr": 98.5, "ts": 62.8, "fps": 24.8,
        "tracker": "ByteTrack",
        "label": "E1\nW8A8 Det",
        "color": "#55A868", "marker": "s",
    },
    "E2": {
        "size": 42.67 + 0.69 + 0.03,          # 43.39 MB
        "mota": 0.219, "ocr": 98.4, "ts": 63.2, "fps": 21.6,
        "tracker": "ByteTrack",
        "label": "E2\nW8A8 Rec",
        "color": "#8172B2", "marker": "p",
    },
    "E3": {
        "size": 10.67 + 0.69 + 0.03,          # 11.39 MB
        "mota": 0.221, "ocr": 98.4, "ts": 63.2, "fps": 24.8,
        "tracker": "ByteTrack",
        "label": "E3\nW8A8 All",
        "color": "#55A868", "marker": "D",
    },
    "E4": {
        "size": 5.33 + 0.34 + 0.015,          # 5.69 MB
        "mota": 0.105, "ocr": 54.6, "ts": 49.2, "fps": 25.7,
        "tracker": "ByteTrack",
        "label": "E4\nW4A16 All",
        "color": "#C44E52", "marker": "^",
    },
    "E5": {
        "size": 10.67 + 0.69 + 0.03,          # 11.39 MB
        "mota": 0.225, "ocr": 98.5, "ts": 62.8, "fps": 20.8,
        "tracker": "ByteTrack",
        "label": "E5\nSQ+W8A8",
        "color": "#CCB974", "marker": "P",
    },
    "E6": {
        "size": 10.67 + 0.06 + 0.69 + 0.03,  # 11.45 MB
        "mota": 0.108, "ocr": 98.4, "ts": 63.2, "fps": 20.4,
        "tracker": "BoT-SORT",
        "label": "E6\nBoT-SORT",
        "color": "#DD8452", "marker": "X",
    },
    "E7": {
        "size": 5.33 + 0.09 + 0.004,          # 5.42 MB
        "mota": 0.105, "ocr": 0.3,  "ts": 12.8, "fps": 25.7,
        "tracker": "ByteTrack",
        "label": "E7\n1-Bit Rec",
        "color": "#937860", "marker": "v",
    },
}

# Pareto-optimal 집합 (minimize size, maximize metric)
# MOTA Pareto: E7(5.42, 0.105), E5(11.39, 0.225)
PARETO_MOTA = {"E7", "E5"}
# OCR Pareto: E7(5.42, 0.3%), E4(5.69, 54.6%), E5(11.39, 98.5%)
PARETO_OCR  = {"E7", "E4", "E5"}


def _pareto_step_line(ids, x_key, y_key):
    """Pareto 계단 라인용 좌표 계산."""
    pts = sorted([(EXPERIMENTS[e][x_key], EXPERIMENTS[e][y_key]) for e in ids])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # step 라인: 각 점에서 다음 x까지 수평 → 수직
    sx, sy = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        sx.append(x)
        sy.append(y)
        if i < len(xs) - 1:
            sx.append(xs[i + 1])
            sy.append(y)
    # 끝을 x축 최대까지 연장
    sx.append(50)
    sy.append(ys[-1])
    return sx, sy


# ──────────────────────────────────────────────────────────────────────────────
# 그래프 그리기
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle(
    "Edge-Sign: Pareto Frontier — Model Size vs. Pipeline Performance\n"
    "(Phase 2 Quantization Experiments E0-E7, CPU ONNX Runtime, Night Sequences)",
    fontsize=13, fontweight="bold", y=1.01,
)

ANNOT_OFFSET = {  # (dx, dy) 텍스트 위치 조정
    "E0": ( 0.6,  0.004),
    "E1": ( 0.6,  0.004),
    "E2": (-9.0,  0.004),
    "E3": ( 0.6, -0.016),
    "E4": ( 0.3,  0.004),
    "E5": ( 0.6,  0.005),
    "E6": ( 0.6, -0.012),
    "E7": ( 0.3, -0.016),
}
ANNOT_OFFSET_OCR = {
    "E0": ( 0.6,  2.5),
    "E1": ( 0.6,  2.5),
    "E2": (-9.0,  2.5),
    "E3": ( 0.6, -5.0),
    "E4": ( 0.3, -8.0),
    "E5": ( 0.6,  2.5),
    "E6": ( 0.6, -5.0),
    "E7": ( 0.3,  2.5),
}

# ── (a) Size vs MOTA ──────────────────────────────────────────────────────────
ax = axes[0]

# Pareto 계단선
sx, sy = _pareto_step_line(PARETO_MOTA, "size", "mota")
ax.plot(sx, sy, color="crimson", linestyle="--", linewidth=1.8,
        alpha=0.75, zorder=2, label="Pareto Frontier")

# 15 MB 목표선
ax.axvline(15, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(15.4, 0.005, "15 MB\ntarget", fontsize=8, color="gray", va="bottom")

for eid, d in EXPERIMENTS.items():
    is_p = eid in PARETO_MOTA
    ax.scatter(d["size"], d["mota"],
               s=260 if is_p else 130,
               c=d["color"], marker=d["marker"],
               zorder=5 if is_p else 4,
               edgecolors="black" if is_p else "none",
               linewidths=1.8)
    dx, dy = ANNOT_OFFSET.get(eid, (0.6, 0.004))
    ax.annotate(d["label"],
                xy=(d["size"], d["mota"]),
                xytext=(d["size"] + dx, d["mota"] + dy),
                fontsize=8, ha="left", va="center",
                fontweight="bold" if is_p else "normal")

ax.set_xlabel("Total Model Size (MB)  [theoretical INT deployment]", fontsize=10)
ax.set_ylabel("Tracking MOTA  (higher is better)", fontsize=10)
ax.set_title("(a) Model Size vs. Tracking MOTA", fontsize=11)
ax.set_xlim(-1, 50)
ax.set_ylim(0.0, 0.27)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower right")

# ── (b) Size vs OCR Accuracy ──────────────────────────────────────────────────
ax = axes[1]

sx, sy = _pareto_step_line(PARETO_OCR, "size", "ocr")
ax.plot(sx, sy, color="crimson", linestyle="--", linewidth=1.8,
        alpha=0.75, zorder=2, label="Pareto Frontier")

ax.axvline(15, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(15.4, 2, "15 MB\ntarget", fontsize=8, color="gray", va="bottom")
ax.axhline(95, color="steelblue", linestyle=":", linewidth=1.2, alpha=0.5)
ax.text(0.5, 95.8, "95% threshold", fontsize=8, color="steelblue")

for eid, d in EXPERIMENTS.items():
    is_p = eid in PARETO_OCR
    ax.scatter(d["size"], d["ocr"],
               s=260 if is_p else 130,
               c=d["color"], marker=d["marker"],
               zorder=5 if is_p else 4,
               edgecolors="black" if is_p else "none",
               linewidths=1.8,
               label=f"{eid}: {d['label'].split(chr(10))[1]}")
    dx, dy = ANNOT_OFFSET_OCR.get(eid, (0.6, 2.5))
    ax.annotate(d["label"],
                xy=(d["size"], d["ocr"]),
                xytext=(d["size"] + dx, d["ocr"] + dy),
                fontsize=8, ha="left", va="center",
                fontweight="bold" if is_p else "normal")

ax.set_xlabel("Total Model Size (MB)  [theoretical INT deployment]", fontsize=10)
ax.set_ylabel("OCR Top-1 Accuracy %  (higher is better)", fontsize=10)
ax.set_title("(b) Model Size vs. OCR Accuracy", fontsize=11)
ax.set_xlim(-1, 50)
ax.set_ylim(-5, 107)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right", ncol=2)

plt.tight_layout()

# 저장
out = ROOT / "assets" / "pareto_frontier.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[OK] Saved: {out}  ({out.stat().st_size / 1024:.1f} KB)")
