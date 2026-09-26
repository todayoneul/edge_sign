"""Qualitative detections of YOLO26-n per precision on two daylight test frames.

Frame rule (fixed before looking at detections): the daylight test frame with the most
ground-truth objects, and the frame with the most objects among those at least 100
frames away from it (ties: earlier manifest order). Each panel is cropped to the union
of the ground-truth boxes plus a margin, at the frame's aspect ratio.

Detections are read from the saved ORT CPU predictions (runtime_matrix.py accuracy
runs), so the figure shows exactly what the accuracy numbers were computed from.

Usage: python scripts/paper/plot_qualitative.py --artifact-root <checkout with data/>
Output: paper_evidence/figures/fig10_qualitative.png
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VARIANTS = [("v4_fp32", "FP32"), ("v4_int8_full", "INT8 full graph"), ("v4_int8_head_excl", "INT8 head kept FP32")]
COLOR = {0: "#00e5ff", 1: "#ff6d00"}  # traffic sign, traffic light
CONF = 0.25


def predictions(folder: Path) -> list[dict]:
    plain, packed = folder / "predictions.jsonl", folder / "predictions.jsonl.gz"
    text = plain.read_text(encoding="utf-8") if plain.exists() else gzip.decompress(packed.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines()]


def pick_frames(rows: list[dict], gap: int = 100) -> list[int]:
    day = [i for i, r in enumerate(rows) if r["lighting"] == "daylight"]
    first = min(day, key=lambda i: (-len(rows[i]["classes"]), i))
    second = min((i for i in day if abs(i - first) >= gap), key=lambda i: (-len(rows[i]["classes"]), i))
    return [first, second]


def crop_box(boxes: list[list[float]], width: int, height: int, margin: float = 0.12, min_w: float = 640) -> tuple:
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w, h = (x2 - x1) * (1 + 2 * margin), (y2 - y1) * (1 + 2 * margin)
    aspect = width / height
    w = min(max(w, h * aspect, min_w), width)
    h = min(w / aspect, height)
    w = h * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0, cx - w / 2), width - w)
    top = min(max(0, cy - h / 2), height - h)
    return int(left), int(top), int(left + w), int(top + h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--out", type=Path, default=Path("paper_evidence/figures/fig10_qualitative.png"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    preds = {key: predictions(args.matrix / f"cpu_{key}") for key, _ in VARIANTS}
    frames = pick_frames(rows)
    # rows = precision variants, columns = frames; each panel keeps the crop's aspect ratio
    crops = [crop_box(rows[i]["boxes_xyxy"], rows[i]["width"], rows[i]["height"]) for i in frames]
    ratios = [(c[2] - c[0]) / (c[3] - c[1]) for c in crops]
    panel_h = 7.0 / sum(ratios)
    fig, axes = plt.subplots(len(VARIANTS), len(frames), figsize=(7.2, panel_h * len(VARIANTS) + 0.5),
                             gridspec_kw={"width_ratios": ratios, "wspace": 0.03, "hspace": 0.05})
    for c, index in enumerate(frames):
        row = rows[index]
        image = cv2.cvtColor(cv2.imread(str(args.artifact_root / row["image_rel"])), cv2.COLOR_BGR2RGB)
        left, top, right, bottom = crops[c]
        for r, (key, title) in enumerate(VARIANTS):
            record = preds[key][index]
            if record["image_rel"] != row["image_rel"]:
                raise ValueError(f"prediction order differs from the manifest at {index}")
            ax = axes[r][c]
            ax.imshow(image[top:bottom, left:right])
            for (x1, y1, x2, y2) in row["boxes_xyxy"]:
                ax.add_patch(Rectangle((x1 - left, y1 - top), x2 - x1, y2 - y1, fill=False, ec="white", lw=0.9, ls="--"))
            shown = [d for d in record["detections"] if d["confidence"] >= CONF]
            for d in shown:
                x1, y1, x2, y2 = d["box_xyxy"]
                color = COLOR[d["class_id"]]
                ax.add_patch(Rectangle((x1 - left, y1 - top), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.4))
                ax.text(x1 - left, y1 - top - 3, f"{d['confidence']:.2f}", color=color, fontsize=5.5, va="bottom")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"test frame {index} ({len(row['boxes_xyxy'])} objects)", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(title, fontsize=8.5)
            ax.text(0.01, 0.98, f"{len(shown)} detections", transform=ax.transAxes, va="top",
                    fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "lw": 0})
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.01)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("wrote", args.out, "frames", frames)


if __name__ == "__main__":
    main()
