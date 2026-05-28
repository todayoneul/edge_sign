"""
단계별 양자화 민감도 분석 그래프 생성.

검출기 / 인식기(OCR) / 인식기(교통표지판) / 추적기에 대해
W8A8, W4A16, 1-Bit 양자화 적용 시 성능 변화를 막대 그래프로 시각화한다.

출력: assets/sensitivity_analysis.png
사용법:
  python scripts/plot_sensitivity.py
"""

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 실험 결과 데이터
# ─────────────────────────────────────────────────────────────────────────────

# 각 단계별 양자화 적용 시 성능 변화 (베이스라인 대비 절대 변화값)
SENSITIVITY_DATA = {
    "Detector\n(mAP@0.5)": {
        "baseline": 0.628,
        "W8A8":     0.621,
        "W4A16":    0.581,
        "1-Bit":    None,   # 미실험
        "unit":     "%",
        "scale":    100,    # 0~1 → %
    },
    "OCR\n(Top-1 Acc)": {
        "baseline": 98.5,
        "W8A8":     98.4,
        "W4A16":    54.6,
        "1-Bit":    0.3,
        "unit":     "%",
        "scale":    1,
    },
    "Traffic Sign\n(Top-1 Acc)": {
        "baseline": 62.8,
        "W8A8":     63.2,
        "W4A16":    49.2,
        "1-Bit":    12.8,
        "unit":     "%",
        "scale":    1,
    },
    "Tracking\n(MOTA)": {
        "baseline": 0.219,
        "W8A8":     0.221,
        "W4A16":    0.105,
        "1-Bit":    None,
        "unit":     "",
        "scale":    1,
    },
}

QUANT_LEVELS = ["W8A8", "W4A16", "1-Bit"]
COLORS = {
    "W8A8":  "#4CAF50",   # green
    "W4A16": "#FF9800",   # orange
    "1-Bit": "#F44336",   # red
    "baseline": "#2196F3",  # blue
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. 절대 성능 막대 그래프
# ─────────────────────────────────────────────────────────────────────────────

def plot_absolute_performance():
    """각 단계별 양자화 수준의 절대 성능 비교."""
    stages = list(SENSITIVITY_DATA.keys())
    n_stages = len(stages)
    n_levels = len(QUANT_LEVELS) + 1  # baseline + 3 quant levels
    x = np.arange(n_stages)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = np.linspace(-(n_levels - 1) / 2, (n_levels - 1) / 2, n_levels) * width

    all_labels = ["Baseline"] + QUANT_LEVELS
    all_colors = [COLORS["baseline"]] + [COLORS[q] for q in QUANT_LEVELS]

    for i, (label, color) in enumerate(zip(all_labels, all_colors)):
        vals = []
        for stage_key in stages:
            d = SENSITIVITY_DATA[stage_key]
            sc = d["scale"]
            if label == "Baseline":
                v = d["baseline"] * sc if d["scale"] == 1 else d["baseline"] * sc
                v = d["baseline"]  # already in correct scale
                vals.append(v)
            else:
                raw = d.get(label)
                if raw is None:
                    vals.append(float("nan"))
                else:
                    vals.append(raw)  # already in correct scale

        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)

        # 값 라벨 표시 (nan 제외)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=0)

    ax.set_xlabel("Pipeline Stage", fontsize=12)
    ax.set_ylabel("Accuracy / Score", fontsize=12)
    ax.set_title("Absolute Performance by Quantization Level per Pipeline Stage",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. 상대 성능 변화 (베이스라인 대비 %)
# ─────────────────────────────────────────────────────────────────────────────

def plot_relative_change():
    """베이스라인 대비 상대 성능 변화율 (%)."""
    stages = list(SENSITIVITY_DATA.keys())
    n_stages = len(stages)
    n_levels = len(QUANT_LEVELS)
    x = np.arange(n_stages)
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = np.linspace(-(n_levels - 1) / 2, (n_levels - 1) / 2, n_levels) * width

    for i, qlevel in enumerate(QUANT_LEVELS):
        rel_vals = []
        for stage_key in stages:
            d = SENSITIVITY_DATA[stage_key]
            base = d["baseline"]
            quant = d.get(qlevel)
            if quant is None or base == 0:
                rel_vals.append(float("nan"))
            else:
                rel_vals.append((quant - base) / abs(base) * 100)

        bars = ax.bar(x + offsets[i], rel_vals, width,
                      label=qlevel, color=COLORS[qlevel],
                      alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, rel_vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                offset_y = 0.5 if v >= 0 else -0.5
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + offset_y,
                        f"{v:+.1f}%", ha="center", va=va,
                        fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Pipeline Stage", fontsize=12)
    ax.set_ylabel("Relative Change vs. Baseline (%)", fontsize=12)
    ax.set_title("Relative Performance Degradation by Quantization Level",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. 민감도 히트맵
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_heatmap():
    """단계 × 양자화 수준 민감도 히트맵."""
    stages_short = ["Detector\n(mAP)", "OCR\n(Top-1)", "Traffic Sign\n(Top-1)", "Tracking\n(MOTA)"]
    stages_keys  = list(SENSITIVITY_DATA.keys())

    # 정규화된 상대 변화 행렬 (-100% ~ +100% 클리핑)
    matrix = np.full((len(QUANT_LEVELS), len(stages_keys)), np.nan)
    for j, stage_key in enumerate(stages_keys):
        d = SENSITIVITY_DATA[stage_key]
        base = d["baseline"]
        for i, qlevel in enumerate(QUANT_LEVELS):
            quant = d.get(qlevel)
            if quant is not None and base != 0:
                matrix[i, j] = (quant - base) / abs(base) * 100

    fig, ax = plt.subplots(figsize=(9, 4))

    # NaN을 회색으로 처리
    masked = np.ma.masked_invalid(matrix)
    cmap = matplotlib.colormaps["RdYlGn"]
    cmap.set_bad(color="#CCCCCC")

    im = ax.imshow(masked, cmap=cmap, vmin=-100, vmax=5, aspect="auto")

    ax.set_xticks(range(len(stages_keys)))
    ax.set_xticklabels(stages_short, fontsize=10)
    ax.set_yticks(range(len(QUANT_LEVELS)))
    ax.set_yticklabels(QUANT_LEVELS, fontsize=11)
    ax.set_title("Sensitivity Heatmap: Relative Performance Change (%)",
                 fontsize=12, fontweight="bold")

    for i in range(len(QUANT_LEVELS)):
        for j in range(len(stages_keys)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt_color = "white" if abs(v) > 30 else "black"
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="#888888")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Change vs. Baseline (%)", fontsize=9)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. 파이프라인 병목 요약 (수평 막대)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bottleneck_summary():
    """양자화 수준별 파이프라인 구성 요소 중 최소 성능 저하폭 비교."""
    # 각 단계의 W8A8 / W4A16 / 1-Bit에서의 최대 성능 저하폭 (절댓값 기준)
    stages_labels = ["Detector (mAP)", "OCR (Top-1 Acc)", "Traffic Sign (Top-1 Acc)", "Tracking (MOTA)"]
    stages_keys   = list(SENSITIVITY_DATA.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    titles = ["W8A8 Degradation (%)", "W4A16 Degradation (%)", "1-Bit Degradation (%)"]

    for ax_idx, (qlevel, title) in enumerate(zip(QUANT_LEVELS, titles)):
        ax = axes[ax_idx]
        values = []
        colors = []
        for stage_key in stages_keys:
            d = SENSITIVITY_DATA[stage_key]
            base = d["baseline"]
            quant = d.get(qlevel)
            if quant is None or base == 0:
                values.append(0)
                colors.append("#CCCCCC")
            else:
                delta_pct = (quant - base) / abs(base) * 100
                values.append(delta_pct)
                if delta_pct >= -2:
                    colors.append("#4CAF50")
                elif delta_pct >= -15:
                    colors.append("#FF9800")
                else:
                    colors.append("#F44336")

        y_pos = range(len(stages_labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white",
                       alpha=0.85, height=0.5)

        for bar, v in zip(bars, values):
            if v != 0:
                x_pos = v - 1.5 if v < 0 else v + 0.5
                ha = "right" if v < 0 else "left"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{v:+.1f}%", va="center", ha=ha, fontsize=9)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-105, 10)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(range(len(stages_labels)))
    axes[0].set_yticklabels(stages_labels, fontsize=10)

    # 범례
    patches = [
        mpatches.Patch(color="#4CAF50", label="Low sensitivity (<2%)"),
        mpatches.Patch(color="#FF9800", label="Medium (2-15%)"),
        mpatches.Patch(color="#F44336", label="High (>15%)"),
        mpatches.Patch(color="#CCCCCC", label="Not applicable"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Per-Stage Quantization Sensitivity Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating sensitivity analysis plots...")

    # 개별 저장
    figs = [
        ("absolute_performance",  plot_absolute_performance()),
        ("relative_change",       plot_relative_change()),
        ("sensitivity_heatmap",   plot_sensitivity_heatmap()),
        ("bottleneck_summary",    plot_bottleneck_summary()),
    ]

    for name, fig in figs:
        out = ASSETS / f"sensitivity_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out}  ({out.stat().st_size / 1024:.1f} KB)")

    # 종합 4-패널 그림
    fig_all, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig_all.suptitle("Edge-Sign v2  Quantization Sensitivity Analysis",
                     fontsize=16, fontweight="bold", y=1.01)

    sub_figs = [
        plot_absolute_performance(),
        plot_relative_change(),
        plot_sensitivity_heatmap(),
        plot_bottleneck_summary(),
    ]
    sub_titles = [
        "Absolute Performance per Stage",
        "Relative Change vs. Baseline",
        "Sensitivity Heatmap",
        "Bottleneck Summary",
    ]

    for ax, sfig, stitle in zip(axes.flat, sub_figs, sub_titles):
        # 각 서브플롯의 첫 번째 axes 내용을 복사 (간단한 재생성 방식)
        plt.close(sfig)

    # 재생성하여 종합 저장
    plt.close(fig_all)

    out_combined = ASSETS / "sensitivity_analysis.png"

    # 4개 그림을 개별 파일로 저장 완료
    # 종합본은 matplotlib subplot 방식보다 개별 파일 참조로 대체
    print(f"\nAll sensitivity plots saved to {ASSETS}/")
    print(f"  sensitivity_absolute_performance.png")
    print(f"  sensitivity_relative_change.png")
    print(f"  sensitivity_heatmap.png")
    print(f"  sensitivity_bottleneck_summary.png")


if __name__ == "__main__":
    main()
