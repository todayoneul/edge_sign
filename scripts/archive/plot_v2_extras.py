"""
v2 Stratified Split 보조 시각화 (3 figures).

생성:
  assets/v2/experiment_comparison.png  -- E0~E7 mAP + MOTA + OCR 통합 막대
  assets/v2/compression_ratio.png      -- FP32 → INT8 모델 크기 압축률
  assets/v2/fps_comparison.png         -- fake-quant vs INT8 Static FPS 비교

사용법:
  python scripts/plot_v2_extras.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "assets" / "v2"
ASSETS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# v2 실험 데이터 (docs/EXPERIMENTS.md 기준)
# ─────────────────────────────────────────────────────────────────────────────
EXPS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
LABELS = ["FP32 All", "W8A8 Det", "FP32+W8A8 Rec", "W8A8 All",
          "W4A16 All", "SQ+W8A8", "BoT-SORT", "W4A16+1-Bit"]

# 각 지표 (정규화 안 함, 절대값)
MAP50 = [0.587, 0.587, 0.587, 0.587, 0.523, 0.587, 0.587, 0.523]
MOTA  = [0.295, 0.291, 0.295, 0.291, 0.176, 0.280, 0.068, 0.176]
OCR   = [98.5, 98.5, 98.4, 98.4, 54.6, 98.5, 98.4,  0.3]   # %
SIZE  = [22.3,  6.2, 21.7,  5.6,  2.8,  5.6,  5.8,  2.7]   # MB
FPS_FAKE = [23.3, 24.6, 24.2, 24.1, 24.7, 20.1, 20.4, 25.9]

# E0/E3 INT8 Static 결과 (별도 측정)
FPS_INT8 = {"E0": 56.3, "E3": 56.3}

# 색상: Pareto 최적 강조
COLORS = ["#888888"] * 8
COLORS[3] = "#E64A4A"   # E3 MOTA Pareto
COLORS[5] = "#E68A4A"   # E5 OCR Pareto


# ─────────────────────────────────────────────────────────────────────────────
# (1) E0~E7 통합 비교 — 3-패널 막대 그래프
# ─────────────────────────────────────────────────────────────────────────────
def plot_experiment_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "E0~E7 Quantization Comparison  (v2 Stratified Split, CPU)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x = np.arange(len(EXPS))

    # (a) Detection mAP@0.5
    ax = axes[0]
    bars = ax.bar(x, MAP50, color=COLORS, edgecolor="black", linewidth=0.7, alpha=0.9)
    ax.axhline(MAP50[0], color="#1f77b4", ls=":", lw=1.2, alpha=0.6, label=f"E0 baseline ({MAP50[0]:.3f})")
    for bar, v in zip(bars, MAP50):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(EXPS, fontsize=9)
    ax.set_ylim(0, 0.7); ax.set_ylabel("mAP@0.5", fontsize=10)
    ax.set_title("(a) Detection mAP@0.5", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) Tracking MOTA
    ax = axes[1]
    bars = ax.bar(x, MOTA, color=COLORS, edgecolor="black", linewidth=0.7, alpha=0.9)
    ax.axhline(MOTA[0], color="#1f77b4", ls=":", lw=1.2, alpha=0.6, label=f"E0 baseline ({MOTA[0]:.3f})")
    for bar, v in zip(bars, MOTA):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(EXPS, fontsize=9)
    ax.set_ylim(0, 0.36); ax.set_ylabel("MOTA", fontsize=10)
    ax.set_title("(b) Tracking MOTA", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (c) OCR Top-1
    ax = axes[2]
    bars = ax.bar(x, OCR, color=COLORS, edgecolor="black", linewidth=0.7, alpha=0.9)
    ax.axhline(OCR[0], color="#1f77b4", ls=":", lw=1.2, alpha=0.6, label=f"E0 baseline ({OCR[0]}%)")
    ax.axhline(95, color="#999999", ls=":", lw=1.0, alpha=0.5, label="95% usable")
    for bar, v in zip(bars, OCR):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(EXPS, fontsize=9)
    ax.set_ylim(-3, 115); ax.set_ylabel("OCR Top-1 Accuracy (%)", fontsize=10)
    ax.set_title("(c) OCR Top-1 Accuracy", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Pareto 최적 범례 (figure 하단)
    handles = [
        mpatches.Patch(facecolor="#E64A4A", edgecolor="black", label="E3 — MOTA Pareto (5.6 MB)"),
        mpatches.Patch(facecolor="#E68A4A", edgecolor="black", label="E5 — OCR Pareto (5.6 MB)"),
        mpatches.Patch(facecolor="#888888", edgecolor="black", label="Other experiments"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True, framealpha=0.95,
               edgecolor="#CCCCCC")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = ASSETS / "experiment_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out.name}  ({out.stat().st_size/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# (2) 모델 압축률 (FP32 → INT8 Static)
# ─────────────────────────────────────────────────────────────────────────────
def plot_compression_ratio():
    models = ["YOLOv8s\n(Detector)", "KoreanOCRNet\n(OCR)", "TrafficSignNet\n(Classifier)"]
    fp32   = [44.75, 2.88, 0.13]
    int8   = [11.66, 0.80, 0.04]
    ratios = [fp / i for fp, i in zip(fp32, int8)]
    cossim = [0.9996, 0.9838, 0.9999]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Static INT8 QDQ Compression — Size & Accuracy Preservation",
        fontsize=13, fontweight="bold", y=1.0,
    )

    bars1 = ax.bar(x - width/2, fp32, width, label="FP32",
                   color="#4878CF", edgecolor="black", linewidth=0.7, alpha=0.9)
    bars2 = ax.bar(x + width/2, int8, width, label="INT8 Static (v2)",
                   color="#E68A4A", edgecolor="black", linewidth=0.7, alpha=0.9)

    # 값 라벨
    for bar, v in zip(bars1, fp32):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.2f} MB",
                ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, int8):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.2f} MB",
                ha="center", va="bottom", fontsize=9, color="#B85800", fontweight="bold")

    # 압축률 + CosSim 주석
    for i, (r, cs) in enumerate(zip(ratios, cossim)):
        ax.text(i, max(fp32) * 0.78,
                f"{r:.2f}×\ncompression\nCosSim {cs:.4f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#FFF8DC", edgecolor="#AAAAAA", linewidth=0.8))

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Model File Size (MB)", fontsize=11)
    ax.set_ylim(0, max(fp32) * 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS / "compression_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out.name}  ({out.stat().st_size/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# (3) FPS 비교 (fake-quant vs INT8 Static)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fps_comparison():
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "Pipeline FPS — Fake-Quant vs. Static INT8 (CPU ONNX Runtime, v2)",
        fontsize=13, fontweight="bold", y=1.0,
    )

    x = np.arange(len(EXPS))
    width = 0.40

    # fake-quant FPS
    bars1 = ax.bar(x - width/2, FPS_FAKE, width, label="fake-quant (FP32 ops)",
                   color="#56A0C0", edgecolor="black", linewidth=0.7, alpha=0.9)

    # INT8 Static (E0/E3만)
    int8_vals = [FPS_INT8.get(e, 0) for e in EXPS]
    bars2 = ax.bar(x + width/2, int8_vals, width, label="Static INT8 QDQ (real)",
                   color="#E64A4A", edgecolor="black", linewidth=0.7, alpha=0.9)

    # 값 라벨
    for bar, v in zip(bars1, FPS_FAKE):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(bars2, int8_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color="#A11414", fontweight="bold")

    # 30 FPS 목표선
    ax.axhline(30, color="#666666", ls="--", lw=1.3, alpha=0.7, label="30 FPS target")
    ax.text(7.6, 31, "30 FPS target", fontsize=9, color="#444444", ha="right")

    ax.set_xticks(x); ax.set_xticklabels([f"{e}\n{l}" for e, l in zip(EXPS, LABELS)],
                                          fontsize=8.5)
    ax.set_ylabel("Pipeline FPS  (higher is better)", fontsize=11)
    ax.set_ylim(0, 65)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 가속비 주석
    for e in ["E0", "E3"]:
        i = EXPS.index(e)
        fake = FPS_FAKE[i]
        int8 = FPS_INT8[e]
        speedup = int8 / fake
        ax.annotate(
            f"{speedup:.2f}×\nspeedup",
            xy=(i + width/2, int8),
            xytext=(i + width/2, int8 + 7),
            fontsize=9, color="#A11414", fontweight="bold",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color="#A11414", lw=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS / "fps_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out.name}  ({out.stat().st_size/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating v2 extra visualizations → {ASSETS}/")
    plot_experiment_comparison()
    plot_compression_ratio()
    plot_fps_comparison()


if __name__ == "__main__":
    main()
