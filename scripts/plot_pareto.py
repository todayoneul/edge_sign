"""
Edge-Sign Phase 2  Pareto Frontier Visualization.

실험 E0-E7 결과 기반으로 Model Size vs Pipeline Performance Pareto 차트를 생성한다.
크기값은 docs/EXPERIMENTS.md의 이론적 INT 배포 크기 기준.

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
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 실험 데이터  (v2 Stratified Split, docs/EXPERIMENTS.md 기준)
# 모델 크기: 이론적 INT 배포 크기 (YOLOv8s INT8≈5.4 MB / KoreanOCRNet+TS 합산)
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "E0": {"size": 22.3, "mota": 0.295, "ocr": 98.5, "fps":  8.6,
           "label": "E0", "desc": "FP32 All",         "color": "#4878CF", "marker": "o"},
    "E1": {"size":  6.2, "mota": 0.291, "ocr": 98.5, "fps":  6.9,
           "label": "E1", "desc": "W8A8 Det",         "color": "#6ACC65", "marker": "s"},
    "E2": {"size": 21.7, "mota": 0.295, "ocr": 98.4, "fps":  7.7,
           "label": "E2", "desc": "FP32 Det+W8A8 Rec","color": "#B47CC7", "marker": "p"},
    "E3": {"size":  5.6, "mota": 0.291, "ocr": 98.4, "fps":  7.0,
           "label": "E3", "desc": "W8A8 All",         "color": "#56A0C0", "marker": "D"},
    "E4": {"size":  2.8, "mota": 0.176, "ocr": 54.6, "fps":  8.2,
           "label": "E4", "desc": "W4A16 All",        "color": "#D65F5F", "marker": "^"},
    "E5": {"size":  5.6, "mota": 0.280, "ocr": 98.5, "fps":  6.0,
           "label": "E5", "desc": "SQ+W8A8",          "color": "#EE854A", "marker": "P"},
    "E6": {"size":  5.8, "mota": 0.068, "ocr": 98.4, "fps": 20.4,
           "label": "E6", "desc": "BoT-SORT",         "color": "#A9A9A9", "marker": "X"},
    "E7": {"size":  2.7, "mota": 0.176, "ocr":  0.3, "fps":  8.0,
           "label": "E7", "desc": "W4A16+1-Bit",      "color": "#8B6914", "marker": "v"},
}

# Pareto 최적 집합 (size 최소화 & metric 최대화)
# MOTA: E7(2.7, 0.176) → E3(5.6, 0.291) → E2(21.7, 0.295)
#   (E1=E3에 지배, E0=E2에 지배, E4=E7에 지배(같은 MOTA 큰 size), E5/E6=E3에 지배)
# OCR:  E7(2.7, 0.3%) → E4(2.8, 54.6%) → E5(5.6, 98.5%)
#   (E0/E2(98.5%)=E5에 지배, E1/E3/E6(98.4%)=E5에 지배)
PARETO_MOTA = {"E7", "E3", "E2"}
PARETO_OCR  = {"E7", "E4", "E5"}

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: Pareto 계단선 좌표
# ─────────────────────────────────────────────────────────────────────────────
def pareto_step(ids, y_key, x_max):
    pts = sorted([(EXPERIMENTS[e]["size"], EXPERIMENTS[e][y_key]) for e in ids])
    xs, ys = [], []
    for i, (x, y) in enumerate(pts):
        xs.append(x); ys.append(y)
        if i < len(pts) - 1:
            xs.append(pts[i+1][0]); ys.append(y)
    xs.append(x_max); ys.append(ys[-1])
    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# 레이아웃 상수
# ─────────────────────────────────────────────────────────────────────────────
XMAX      = 25.0   # MB  (v2: E0=22.3, E2=21.7)
MS_PARETO = 55     # 마커 크기 (Pareto 최적, 추가 축소)
MS_NORMAL = 28     # 마커 크기 (일반, 추가 축소)
FS_TITLE  = 12
FS_AXIS   = 10
FS_TICK   = 9
FS_ANNOT  = 7.5
FS_LEGEND = 7.5

# ─────────────────────────────────────────────────────────────────────────────
# 레이블 오프셋 (x_offset, y_offset) — 겹침 수동 조정
# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): MOTA  (y 0.02~0.33)
#   v2 분포: E6=0.068, E4/E7=0.176, E5=0.280, E1/E3=0.291, E0/E2=0.295
OFFSET_MOTA = {
    "E0": ( 0.55,  0.000),   # 오른쪽 (E2 위)
    "E1": ( 0.55,  0.000),   # 오른쪽
    "E2": (-0.55,  0.000),   # 왼쪽 (E0와 분리)
    "E3": ( 0.55,  0.000),   # 오른쪽
    "E4": ( 0.55,  0.000),   # 오른쪽
    "E5": (-0.55,  0.000),   # 왼쪽 (E3와 같은 x → 왼쪽)
    "E6": ( 0.55,  0.000),   # 오른쪽
    "E7": (-0.55,  0.000),   # 왼쪽 (E4와 같은 x → 왼쪽)
}
# Panel (b): OCR  (y -5~108)
OFFSET_OCR = {
    "E0": ( 0.55,  0.0),     # 오른쪽
    "E1": ( 0.55,  3.0),     # 오른쪽 위
    "E2": (-0.55,  0.0),     # 왼쪽
    "E3": (-0.55, -5.0),     # 왼쪽 아래 (E5와 분리)
    "E4": ( 0.55,  0.0),     # 오른쪽
    "E5": ( 0.55,  3.0),     # 오른쪽 위 (Pareto, E3 위)
    "E6": ( 0.55, -5.0),     # 오른쪽 아래
    "E7": ( 0.55,  0.0),     # 오른쪽 (낮은 OCR)
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 생성
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Edge-Sign v2: Pareto Frontier — Model Size vs. Pipeline Performance\n"
    "E0-E7 Quantization Experiments  |  v2 Stratified Split  |  Daylight+Night Test",
    fontsize=FS_TITLE, fontweight="bold", y=1.02,
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (a): Model Size vs. MOTA
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]

# Pareto 계단선
sx, sy = pareto_step(PARETO_MOTA, "mota", XMAX)
ax.plot(sx, sy, color="crimson", ls="--", lw=1.8, alpha=0.7,
        zorder=2, label="Pareto frontier")

# 15 MB 목표선
ax.axvline(15, color="#888888", ls=":", lw=1.2, alpha=0.7)
ax.text(15.2, 0.31, "15 MB\ntarget", fontsize=FS_ANNOT - 0.5,
        color="#666666", va="top", ha="left")

# 데이터 포인트
for eid, d in EXPERIMENTS.items():
    is_p = eid in PARETO_MOTA
    ax.scatter(
        d["size"], d["mota"],
        s       = MS_PARETO if is_p else MS_NORMAL,
        c       = d["color"],
        marker  = d["marker"],
        zorder  = 5 if is_p else 4,
        edgecolors = "black" if is_p else "#555555",
        linewidths = 1.0 if is_p else 0.5,
    )
    dx, dy = OFFSET_MOTA[eid]
    ax.annotate(
        d["label"],   # ID만 표기 (설명은 하단 범례)
        xy     = (d["size"], d["mota"]),
        xytext = (d["size"] + dx, d["mota"] + dy),
        fontsize   = FS_ANNOT,
        ha         = "left" if dx >= 0 else "right",
        va         = "center",
        fontweight = "bold" if is_p else "normal",
        color      = "crimson" if is_p else "#222222",
    )

ax.set_xlabel("Total Model Size (MB)  [theoretical INT deployment]", fontsize=FS_AXIS)
ax.set_ylabel("Tracking MOTA  (higher is better)", fontsize=FS_AXIS)
ax.set_title("(a)  Model Size  vs.  Tracking MOTA", fontsize=FS_AXIS + 1, pad=8)
ax.set_xlim(-0.5, XMAX)
ax.set_ylim(0.02, 0.33)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel (a)에는 Pareto/target 라인만 간단히 표시
ax.legend(
    handles=[
        plt.Line2D([0],[0], color="crimson", ls="--", lw=1.8, label="Pareto frontier"),
        plt.Line2D([0],[0], color="#888888", ls=":",  lw=1.2, label="15 MB target"),
    ],
    fontsize=FS_LEGEND, loc="lower right",
    framealpha=0.9, edgecolor="#CCCCCC",
)

# ─────────────────────────────────────────────────────────────────────────────
# Panel (b): Model Size vs. OCR Accuracy
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]

sx, sy = pareto_step(PARETO_OCR, "ocr", XMAX)
ax.plot(sx, sy, color="crimson", ls="--", lw=1.8, alpha=0.7,
        zorder=2, label="Pareto frontier")

ax.axvline(15, color="#888888", ls=":", lw=1.2, alpha=0.7)
ax.text(15.2, 3, "15 MB\ntarget", fontsize=FS_ANNOT - 0.5,
        color="#666666", va="bottom", ha="left")

ax.axhline(95, color="steelblue", ls=":", lw=1.2, alpha=0.5)
ax.text(0.3, 96, "95% usable threshold", fontsize=FS_ANNOT - 0.5,
        color="steelblue", va="bottom")

for eid, d in EXPERIMENTS.items():
    is_p = eid in PARETO_OCR
    ax.scatter(
        d["size"], d["ocr"],
        s       = MS_PARETO if is_p else MS_NORMAL,
        c       = d["color"],
        marker  = d["marker"],
        zorder  = 5 if is_p else 4,
        edgecolors = "black" if is_p else "#555555",
        linewidths = 1.0 if is_p else 0.5,
    )
    dx, dy = OFFSET_OCR[eid]
    ax.annotate(
        d["label"],   # ID만 표기 (설명은 하단 범례)
        xy     = (d["size"], d["ocr"]),
        xytext = (d["size"] + dx, d["ocr"] + dy),
        fontsize   = FS_ANNOT,
        ha         = "left" if dx >= 0 else "right",
        va         = "center",
        fontweight = "bold" if is_p else "normal",
        color      = "crimson" if is_p else "#222222",
    )

ax.set_xlabel("Total Model Size (MB)  [theoretical INT deployment]", fontsize=FS_AXIS)
ax.set_ylabel("OCR Top-1 Accuracy (%)  (higher is better)", fontsize=FS_AXIS)
ax.set_title("(b)  Model Size  vs.  OCR Top-1 Accuracy", fontsize=FS_AXIS + 1, pad=8)
ax.set_xlim(-0.5, XMAX)
ax.set_ylim(-5, 108)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 보조 주석: E5 Pareto 최적 표시 (panel b)
ax.annotate(
    "E5 (SQ+W8A8, 5.6 MB)\nOCR Pareto-optimal\n+ MOTA-optimal at same size",
    xy=(5.6, 98.5), xytext=(10.5, 75),
    fontsize=FS_ANNOT, color="crimson",
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
    ha="left",
)

# Panel (b)에도 Pareto/threshold만 표시
ax.legend(
    handles=[
        plt.Line2D([0],[0], color="crimson",  ls="--", lw=1.8, label="Pareto frontier"),
        plt.Line2D([0],[0], color="steelblue",ls=":",  lw=1.2, label="95% threshold"),
    ],
    fontsize=FS_LEGEND, loc="center right",
    framealpha=0.9, edgecolor="#CCCCCC",
)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 하단 통합 범례 (실험 ID + 설명)
# ─────────────────────────────────────────────────────────────────────────────
shared_handles = [
    plt.scatter([], [], s=MS_PARETO, c=EXPERIMENTS[e]["color"],
                marker=EXPERIMENTS[e]["marker"],
                edgecolors="black", linewidths=1.0,
                label=f"{e}: {EXPERIMENTS[e]['desc']} ({EXPERIMENTS[e]['size']} MB)")
    for e in EXPERIMENTS
]
fig.legend(
    handles=shared_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    fontsize=FS_LEGEND + 0.5,
    frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
)

# ─────────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

out = ASSETS / "pareto_frontier.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] Saved: {out}  ({out.stat().st_size / 1024:.1f} KB)")

# 검증: 저장된 파일 정보 출력
print(f"     Size: {out.stat().st_size / 1024:.1f} KB")
print(f"     Pareto MOTA: {PARETO_MOTA}")
print(f"     Pareto OCR:  {PARETO_OCR}")
print(f"     X range: 0 ~ {XMAX} MB  (E0=24.3, E2=21.8, others ≤ 13.5)")
print(f"     All experiments plotted: {list(EXPERIMENTS.keys())}")
