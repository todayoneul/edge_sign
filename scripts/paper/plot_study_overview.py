"""Connected technical schematic of the studied pipeline and precision boundary.

Tensor glyphs are schematic, not layer-count or feature-dimension claims.
Solid arrows show inference data flow; dashed lines show runtime assignment.
The precision rows illustrate detector full/head-excluded INT8 conditions.
Outputs: fig9_study_overview.{png,pdf,svg}; SVG text remains editable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.paper_style import FULL_WIDTH, apply, save

INK = "#252525"
GRAY = "#62686D"
EDGE = "#92999E"
BLUE = "#3B7396"
TINT = "#DCE9F1"
LIGHT = "#F3F4F4"


def text(ax, x, y, s, size=7.2, *, color=INK, weight="normal", ha="center"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.12)


def rect(ax, x, y, w, h, fill="white", edge=EDGE, lw=0.65, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=lw, linestyle=ls))


def arrow(ax, x0, y0, x1, y1, *, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=6.5, linewidth=0.7, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def tensor(ax, x, y, w=27, h=50):
    # A generic feature-tensor glyph; not an architecture layer inventory.
    rect(ax, x+12, y+10, w, h, "#F5F7F8", EDGE, 0.5)
    rect(ax, x+6, y+5, w, h, "#E7EDF1", EDGE, 0.5)
    rect(ax, x, y, w, h, TINT, BLUE, 0.65)
    for f in (1/3, 2/3):
        ax.plot([x+w*f, x+w*f], [y, y+h], color=BLUE, alpha=0.3, linewidth=0.4)
    for f in (0.25, 0.5, 0.75):
        ax.plot([x, x+w], [y+h*f, y+h*f], color=BLUE, alpha=0.3, linewidth=0.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("paper_evidence/figures/fig9_study_overview.png"))
    args = parser.parse_args()
    apply()
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1000), ylim=(0, 410))
    ax.axis("off")

    # Continuous inference path, with the studied model boundaries expanded.
    rect(ax, 24, 279, 69, 53, LIGHT)
    rect(ax, 18, 273, 69, 53, "white")
    rect(ax, 12, 267, 69, 53, "white", GRAY)
    # Tiny road-frame symbol, explicitly schematic rather than sample data.
    ax.plot([19, 37, 51, 73], [278, 296, 280, 309], color=EDGE, linewidth=0.65)
    text(ax, 48, 244, "RGB frame", 7.2)
    text(ax, 48, 226, "640 × 640", 6.8, color=GRAY)
    arrow(ax, 98, 299, 144, 299)

    rect(ax, 149, 168, 328, 211, fill="none", edge=EDGE, lw=0.65, ls=(0, (4, 3)))
    text(ax, 313, 362, "Detector", 8.1, weight="bold")
    text(ax, 313, 344, "YOLOv8s / YOLO26-n", 7.0, color=GRAY)
    tensor(ax, 183, 274, 29, 45)
    text(ax, 207, 247, "Backbone\n+ neck", 6.8)
    arrow(ax, 233, 299, 286, 299)
    rect(ax, 289, 272, 78, 53, TINT, BLUE)
    text(ax, 328, 299, "Head", 7.2)
    arrow(ax, 369, 299, 395, 299)
    rect(ax, 398, 272, 63, 53, LIGHT)
    text(ax, 429, 299, "Decode", 6.8)
    # Two short precision strips share the same body/head column boundary.
    text(ax, 163, 216, "Full INT8", 6.8, ha="left")
    text(ax, 163, 187, "Head-excluded", 6.8, ha="left")
    for y, head_fill, head_label in ((206, TINT, "INT8"), (177, "white", "FP32")):
        rect(ax, 288, y, 74, 20, TINT, BLUE, 0.5)
        rect(ax, 366, y, 95, 20, head_fill, BLUE if head_label=="INT8" else EDGE, 0.5)
        text(ax, 325, y+10, "INT8", 6.5)
        text(ax, 414, y+10, head_label, 6.5)
    text(ax, 325, 239, "body", 6.5, color=GRAY)
    text(ax, 414, 239, "head / decode", 6.5, color=GRAY)

    arrow(ax, 478, 299, 512, 299)
    rect(ax, 514, 275, 99, 49, LIGHT, GRAY)
    text(ax, 563, 299, "ByteTrack", 7.0)
    text(ax, 563, 254, "association", 6.7, color=GRAY)
    arrow(ax, 615, 299, 644, 299)
    rect(ax, 651, 284, 29, 29, "white", GRAY, 0.65)
    rect(ax, 646, 279, 29, 29, "white", GRAY, 0.65)
    text(ax, 663, 254, "ROIs", 6.7, color=GRAY)
    arrow(ax, 684, 299, 714, 299)

    rect(ax, 717, 168, 174, 211, fill="none", edge=EDGE, lw=0.65, ls=(0, (4, 3)))
    text(ax, 804, 362, "Recognizer", 8.1, weight="bold")
    text(ax, 804, 344, "KoreanSignNet", 7.0, color=GRAY)
    tensor(ax, 735, 277, 20, 39)
    arrow(ax, 773, 299, 802, 299)
    rect(ax, 805, 278, 70, 43, TINT, BLUE)
    text(ax, 840, 299, "Classifier", 6.6)
    text(ax, 804, 251, "14 fine classes", 7.0)
    text(ax, 804, 194, "FP32 / FP16 / INT8", 6.7, color=GRAY)
    arrow(ax, 893, 299, 929, 299)
    text(ax, 960, 299, "Fine\nclass", 7.2)

    # Runtime assignment is connected to the model groups, not a separate panel.
    ax.plot([313, 313, 804, 804], [168, 113, 113, 168], color=GRAY,
            linewidth=0.65, linestyle=(0, (3, 2)))
    text(ax, 561, 136, "Independent ONNX sessions", 7.2)
    for x, name in ((348, "ORT CPU"), (518, "WASM"), (688, "WebGPU")):
        arrow(ax, x+66, 113, x+66, 92, dashed=True, color=GRAY)
        rect(ax, x, 54, 132, 38, "white", GRAY)
        text(ax, x+66, 73, name, 7.2)
    text(ax, 584, 28, "Same model variant + input  ·  Windows / Mac", 7.0, color=GRAY)
    save(fig, args.out)
    fig.savefig(args.out.with_suffix(".svg"))
    args.out.with_suffix(".caption.txt").write_text(
        "Overview of the studied detection-tracking-recognition pipeline. "
        "The detector precision strips contrast full INT8 QDQ with head-excluded "
        "INT8, in which the head and decode operations retain FP32 precision. "
        "FP32 and FP16 baselines and further diagnostic variants are described "
        "in the text. ByteTrack associates detections without learned weights; "
        "KoreanSignNet assigns the fine class to each cropped region. Solid arrows "
        "denote inference data flow, and dashed lines denote independent runtime "
        "assignment of the detector and recognizer sessions. Each model variant "
        "is profiled with the same input across execution backends on Windows "
        "and Mac. Tensor glyphs are schematic and do not encode layer counts "
        "or feature dimensions.\n", encoding="utf-8")
    plt.close(fig)
    print("wrote", args.out, args.out.with_suffix(".pdf"), args.out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
