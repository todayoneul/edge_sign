"""Retention per object-size bin (reads size_bin_retention.py output).

Panels: (a) YOLOv8s and (b) YOLO26-n on the road test set, (c) YOLO11l on COCO. Series:
INT8 with the head in FP32, INT8 with only the decode stage in FP32, decode-stage activations
quantized except the collapse tensors (coordinates quantized, scores kept), and, for the road
detectors, full INT8 with box normalization (Ultralytics-style pre-normalization). Whiskers are
paired 95% bootstrap intervals. Shared paper style (paper_style.py).

Usage: python scripts/paper/plot_size_bins.py
Output: paper_evidence/figures/fig13_size_retention.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.paper_style import FULL_WIDTH, OKABE_ITO, apply, panel_label, save, value_grid

SERIES = [  # (label, key in size_bins.json per workload, key in mitigation size_bins.json, color, marker)
    ("INT8, head FP32", "int8_head_excl", None, OKABE_ITO["blue"], "o"),
    ("INT8, decode FP32", "int8_decode_excl", None, OKABE_ITO["sky"], "s"),
    ("coordinate tensors INT8 (weights FP32)", "coord", None, OKABE_ITO["vermillion"], "^"),
    ("INT8 + box normalization (pre-decode)", None, "pre_norm_int8_full", OKABE_ITO["purple"], "D"),
]
COORD = {"road_v3": "a8sim_decode_no_outconcat", "road_v4": "a8sim_decode_no_collapse", "coco": "a8sim_decode_no_outconcat"}
PANELS = [("road_v3", "v3", "(a) YOLOv8s, road", ["<8", "8-16", "≥16"], "object side at 640 input (px)"),
          ("road_v4", "v4", "(b) YOLO26-n, road", ["<8", "8-16", "≥16"], "object side at 640 input (px)"),
          ("coco", None, "(c) YOLO11l, COCO", ["small", "medium", "large"], "COCO area range")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, default=Path("paper_evidence/extra/size_bins/size_bins.json"))
    parser.add_argument("--mitigation", type=Path, default=Path("paper_evidence/extra/mitigation/size_bins.json"))
    parser.add_argument("--out", type=Path, default=Path("paper_evidence/figures/fig13_size_retention.png"))
    args = parser.parse_args()
    apply()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))["workloads"]
    mitigation = json.loads(args.mitigation.read_text(encoding="utf-8"))["families"] if args.mitigation.exists() else {}
    bins = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.35), sharey=True)
    x = np.arange(len(bins))
    for ax, (workload, family, title, ticks, xlabel) in zip(axes, PANELS, strict=True):
        conditions = primary[workload]["conditions"]
        for k, (name, key, mkey, color, marker) in enumerate(SERIES):
            if key == "coord":
                row = conditions[COORD[workload]]
            elif key:
                row = conditions[key]
            elif mkey and family and mkey in mitigation.get(family, {}):
                row = mitigation[family][mkey]
            else:
                continue
            y = np.array([row["retention"][b] * 100 for b in bins])
            lo = np.array([row["retention_ci95"][b][0] * 100 for b in bins])
            hi = np.array([row["retention_ci95"][b][1] * 100 for b in bins])
            offset = (k - 1.5) * 0.06
            ax.errorbar(x + offset, y, yerr=[y - lo, hi - y], color=color, marker=marker, ms=3.2, lw=1.0,
                        elinewidth=0.6, capsize=1.2, label=name)
        ax.axhline(100, color=OKABE_ITO["black"], lw=0.6)
        ax.set_xticks(x, ticks)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.35, 2.35)
        ax.set_ylim(0, 108)
        panel_label(ax, title)
        value_grid(ax)
    axes[0].set_ylabel("Retention vs FP32 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save(fig, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
