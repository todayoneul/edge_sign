"""Study overview figure: pipeline components, quantization variants, runtimes and research questions.

Drawn at the final print width (6.3 in, 7-8 pt text) in the shared paper style (paper_style.py).
Output: paper_evidence/figures/fig9_study_overview.{pdf,png} (Fig. 1 of the draft).
Usage: python scripts/paper/plot_study_overview.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.paper_style import FULL_WIDTH, OKABE_ITO, apply, save

INK, MUTED, EDGE = "#1A1A1A", "#4D4D4D", "#8C8C8C"
FILL = {"det": "#DCEBF7", "trk": "#F2F2F2", "rec": "#D9F2E9", "rt": "#FCEFD6"}


def box(ax, x, y, w, h, title, lines, color, title_size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=color, edgecolor=EDGE, linewidth=0.6))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for k, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.115 - k * 0.062, line, ha="center", va="top", fontsize=6.8, color=MUTED)


def arrow(ax, x0, y0, x1, y1, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=INK,
                                 linewidth=0.7, linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, text, ha="center", va="bottom", fontsize=6.3, color=MUTED)


def badge(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.3, fontweight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": OKABE_ITO["blue"], "edgecolor": "none"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("paper_evidence/figures/fig9_study_overview.png"))
    args = parser.parse_args()
    apply()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # top row: the pipeline; RQ1 badges on the two quantized components
    y, h = 0.47, 0.47
    mid = y + 0.25
    box(ax, 0.0, y + 0.13, 0.09, 0.24, "Frame", ["video", "640×640"], FILL["trk"])
    box(ax, 0.12, y, 0.275, h, "Detector", ["YOLOv8s (DFL head, NMS)", "YOLO26-n (NMS-free)",
                                            "FP32 · FP16 · INT8 (QDQ)", "INT8: full / head / decode FP32",
                                            "+ box normalization, calibration"], FILL["det"])
    box(ax, 0.445, y + 0.08, 0.12, 0.33, "Tracker", ["ByteTrack", "no weights,", "not quantized"], FILL["trk"])
    box(ax, 0.615, y, 0.25, h, "Recognizer", ["KoreanSignNet, 14 classes", "32×32 ROI, 29k params",
                                              "FP32 · FP16 · INT8 (QDQ)", "INT8: full / head FP32"], FILL["rec"])
    box(ax, 0.9, y + 0.13, 0.1, 0.24, "Output", ["sign / light", "fine class"], FILL["trk"])
    arrow(ax, 0.09, mid, 0.12, mid)
    arrow(ax, 0.395, mid, 0.445, mid, "boxes")
    arrow(ax, 0.565, mid, 0.615, mid, "ROIs")
    arrow(ax, 0.865, mid, 0.9, mid)
    badge(ax, 0.366, y + h - 0.045, "RQ1")
    badge(ax, 0.836, y + h - 0.045, "RQ1")

    # bottom: execution environments on two devices (RQ2); each component can be placed on any (RQ3)
    box(ax, 0.1, 0.07, 0.8, 0.29, "Execution environments (same ONNX file and input tensor)",
        ["ORT CPU 1/4 threads  ·  ORT-Web WASM 1/4 threads  ·  ORT-Web WebGPU 1.22 / 1.30",
         "two devices: Windows (Ryzen 5 9600X + RTX 5070), Mac (Apple M2 Pro)",
         "batch-1 latency, 1,024 calls, mean / p90; operator placement from session logs"],
        FILL["rt"], title_size=7.6)
    badge(ax, 0.873, 0.33, "RQ2")
    arrow(ax, 0.2575, 0.36, 0.2575, y, style="<|-|>", ls="--")
    arrow(ax, 0.74, 0.36, 0.74, y, style="<|-|>", ls="--")
    badge(ax, 0.325, 0.415, "RQ3")
    ax.text(0.362, 0.415, "per-component placement in the browser", ha="left", va="center", fontsize=6.5, color=MUTED)

    ax.text(0.5, 0.0, "RQ1: where INT8 loses accuracy  ·  RQ2: why latency changes by runtime and device  ·  "
            "RQ3: which runtime each component uses", ha="center", va="bottom", fontsize=6.3, color=INK)
    save(fig, args.out)
    print("wrote", args.out, args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
