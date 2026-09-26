"""Study overview figure: pipeline components, quantization variants, runtimes and research questions.

Output: paper_evidence/figures/fig9_study_overview.png (Fig. 1 of the draft).
Usage: python scripts/paper/plot_study_overview.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
FILL = {"det": "#dbeafe", "trk": "#f3f4f6", "rec": "#dcfce7", "rt": "#fef3c7", "rq": "#ede9fe"}


def box(ax, x, y, w, h, title, lines, color, title_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=color, edgecolor=LINE, linewidth=1))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - k * 0.043, line, ha="center", va="top", fontsize=7.6, color=MUTED)


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=INK,
                                 linewidth=1.1, linestyle=ls))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, text, ha="center", va="bottom", fontsize=7.2, color=MUTED)


def badge(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#6d28d9", "edgecolor": "none"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("paper_evidence/figures/fig9_study_overview.png"))
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # top row: the pipeline; RQ1 badges on the two quantized components
    y, h = 0.55, 0.38
    box(ax, 0.01, y + 0.11, 0.10, 0.16, "Frame", ["road video", "640×640 RGB"], FILL["trk"])
    box(ax, 0.15, y, 0.25, h, "Detector", ["YOLOv8s (DFL head + NMS)", "or YOLO26-n (NMS-free, no DFL)",
                                           "FP32 · FP16 · INT8 QDQ", "INT8 scope: full graph / head FP32",
                                           "bias: INT32 / FP32"], FILL["det"])
    box(ax, 0.44, y + 0.07, 0.15, 0.24, "Tracker", ["ByteTrack", "(no weights,", "not quantized)"], FILL["trk"])
    box(ax, 0.63, y, 0.23, h, "Recognizer", ["KoreanSignNet, 14 classes", "32×32 ROI (29k params)",
                                             "FP32 · FP16 · INT8 QDQ", "INT8 scope: full / head FP32",
                                             "bias: INT32 / FP32"], FILL["rec"])
    box(ax, 0.89, y + 0.11, 0.10, 0.16, "Output", ["sign / light", "fine class"], FILL["trk"])
    arrow(ax, 0.11, y + 0.19, 0.15, y + 0.19)
    arrow(ax, 0.40, y + 0.19, 0.44, y + 0.19, "boxes")
    arrow(ax, 0.59, y + 0.19, 0.63, y + 0.19, "track ROIs")
    arrow(ax, 0.86, y + 0.19, 0.89, y + 0.19)
    badge(ax, 0.385, y + h - 0.02, "RQ1")
    badge(ax, 0.845, y + h - 0.02, "RQ1")

    # bottom: execution environments (RQ2); each component can be placed on any of them (RQ3)
    box(ax, 0.12, 0.13, 0.77, 0.32, "Execution environments",
        ["same ONNX file and input tensor in every environment",
         "ONNX Runtime CPU (native) · 1 / 4 threads      ONNX Runtime Web · WASM · 1 / 4 threads",
         "ONNX Runtime Web · WebGPU · runtime 1.22 (JSEP) / 1.30",
         "latency: batch 1, 1,024 calls (WebGPU INT8: 128), mean / p90;  operator placement from session logs"],
        FILL["rt"], title_size=9)
    badge(ax, 0.865, 0.425, "RQ2")
    arrow(ax, 0.275, 0.45, 0.275, y, style="<|-|>", ls="--")
    arrow(ax, 0.745, 0.45, 0.745, y, style="<|-|>", ls="--")
    badge(ax, 0.335, 0.50, "RQ3")
    ax.text(0.36, 0.50, "per-component placement in one browser page", ha="left", va="center", fontsize=7.4, color=MUTED)

    ax.text(0.5, 0.04, "RQ1: where does INT8 lose accuracy (detector body, head, recognizer)?   "
            "RQ2: how and why does latency change per environment?   RQ3: which environment should each component use?",
            ha="center", va="center", fontsize=7.4, color=INK)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
