"""Paired frame-level bootstrap CI for accuracy retention (variant / FP32).

Detectors: resample test frames with replacement (same draw for both models) and
recompute mAP@0.5:0.95 from the saved predictions.jsonl. evaluate_qdq_detection
matches greedily inside each frame in global confidence order, so every detection's
TP flag (per class and IoU threshold) depends only on its own frame. The flags are
computed once, and each resample repeats detections by their frame's draw count.
With every count equal to 1 this reproduces compute_metrics exactly (checked at start).

Recognizer: resample ROIs from recognition/variants/predictions.csv.

Frames within a sequence are correlated, so the frame interval is optimistic (too
narrow). A moving-block bootstrap (blocks of --block consecutive frames drawn within each
sequence; 25 frames = 5 s at the test split's 5 fps) is reported next to it for the
detector pairs. Both are sensitivity checks, not significance tests.

Pairs added after the first run (decode-stage-excluded INT8) draw from the same random
stream after the recognizer pairs, and the block bootstrap uses its own stream (seed + 1),
so every earlier interval is reproduced unchanged.

Usage: python scripts/paper/bootstrap_retention.py --resamples 1000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.evaluate_qdq_detection import CLASSES, IOU_THRESHOLDS, compute_metrics, iou

PAIRS = [("v4_fp32", k) for k in ("v4_fp16", "v4_int8_head_excl", "v4_int8_head_excl_fbias")] + [
    ("v3_fp32", k) for k in ("v3_fp16", "v3_int8_head_excl", "v3_int8_head_excl_fbias")]
EXTRA_PAIRS = [("v4_fp32", "v4_int8_decode_excl"), ("v3_fp32", "v3_int8_decode_excl")]


def ap(tp: np.ndarray, fp: np.ndarray, n_gt: float) -> float:
    # same envelope and 101-point trapezoid as evaluate_qdq_detection._ap
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp))
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(grid, mrec, mpre), grid))


class Flags:
    """Per class: detections in global confidence order, their frame, and TP flags per IoU threshold."""

    def __init__(self, folder: Path):
        # committed evidence stores predictions gzip-compressed; fresh runs write plain JSONL
        plain, packed = folder / "predictions.jsonl", folder / "predictions.jsonl.gz"
        text = plain.read_text(encoding="utf-8") if plain.exists() else gzip.decompress(packed.read_bytes()).decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines()]
        self.gt = [r["ground_truth"] for r in rows]
        self.pred = [r["detections"] for r in rows]
        self.cls = {}
        for c in CLASSES:
            gt = [[b for k, b in zip(g["classes"], g["boxes_xyxy"]) if k == c] for g in self.gt]
            dets = sorted(((float(p["confidence"]), i, p["box_xyxy"]) for i, frame in enumerate(self.pred)
                           for p in frame if p["class_id"] == c and p["confidence"] >= 0.001), key=lambda d: -d[0])
            frames = np.array([d[1] for d in dets], dtype=np.int64)
            tp = np.zeros((len(dets), len(IOU_THRESHOLDS)), dtype=np.int8)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                used = [set() for _ in gt]
                for n, (_, i, box) in enumerate(dets):
                    scores = [iou(box, g) if j not in used[i] else -1.0 for j, g in enumerate(gt[i])]
                    if scores:
                        best = int(np.argmax(scores))
                        if scores[best] >= threshold:
                            used[i].add(best)
                            tp[n, t] = 1
            self.cls[c] = (frames, tp, np.array([len(g) for g in gt], dtype=np.int64))

    def map50_95(self, counts: np.ndarray) -> float:
        aps = []
        for frames, tp, n_gt in self.cls.values():
            total = float((counts * n_gt).sum())
            if total == 0:
                continue
            idx = np.repeat(np.arange(len(frames)), counts[frames])
            flags = tp[idx]
            aps.append(np.mean([ap(flags[:, t], 1 - flags[:, t], total) for t in range(flags.shape[1])]))
        return float(np.mean(aps)) if aps else 0.0


def ci(values: np.ndarray) -> dict:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"ci95_low": float(lo), "ci95_high": float(hi), "share_below_0.99": float(np.mean(values < 0.99)),
            "share_below_0.95": float(np.mean(values < 0.95))}


def block_counts(sequences: list[str], block: int, rng: np.random.Generator) -> np.ndarray:
    """Frame draw counts of one moving-block resample; blocks stay inside their sequence."""
    counts = np.zeros(len(sequences), dtype=np.int64)
    start = 0
    while start < len(sequences):
        end = start
        while end < len(sequences) and sequences[end] == sequences[start]:
            end += 1
        length = end - start
        size = min(block, length)
        drawn = 0
        while drawn < length:
            first = start + int(rng.integers(0, length - size + 1))
            take = min(size, length - drawn)
            counts[first:first + take] += 1
            drawn += take
        start = end
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--recognition", type=Path, default=Path("paper_evidence/recognition/variants"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=25, help="moving-block length in frames (0 disables)")
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/runtime/matrix/bootstrap_retention.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {"resamples": args.resamples, "seed": args.seed, "unit": "frame (detector) / ROI (recognizer)",
               "note": "frames within a sequence are correlated; the interval is optimistic", "pairs": {}}
    flags = {}

    def detector_pairs(pairs: list[tuple[str, str]]) -> None:
        for ref_key, key in pairs:
            for k in (ref_key, key):
                if k not in flags:
                    flags[k] = Flags(args.matrix / f"cpu_{k}")
                    exact = compute_metrics(flags[k].gt, flags[k].pred)["mAP50_95"]
                    fast = flags[k].map50_95(np.ones(len(flags[k].gt), dtype=np.int64))
                    if abs(exact - fast) > 1e-12:
                        raise SystemExit(f"{k}: fast mAP {fast} != compute_metrics {exact}")
            n = len(flags[ref_key].gt)
            ones = np.ones(n, dtype=np.int64)
            point = flags[key].map50_95(ones) / flags[ref_key].map50_95(ones)
            ratios = []
            for _ in range(args.resamples):
                counts = np.bincount(rng.integers(0, n, n), minlength=n)
                ratios.append(flags[key].map50_95(counts) / flags[ref_key].map50_95(counts))
            results["pairs"][key] = {"reference": ref_key, "metric": "mAP50_95", "retention": point, **ci(np.array(ratios))}
            print(key, json.dumps(results["pairs"][key]), flush=True)

    detector_pairs(PAIRS)
    with (args.recognition / "predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    y = np.array([int(r["true_class_id"]) for r in rows])
    ref = np.array([int(r["fp32"]) for r in rows]) == y
    for key in ("fp16", "int8_full", "int8_full_fbias", "int8_head_excl", "int8_head_excl_fbias"):
        ok = np.array([int(r[key]) for r in rows]) == y
        idx = rng.integers(0, len(y), (args.resamples, len(y)))
        ratios = ok[idx].mean(1) / ref[idx].mean(1)
        results["pairs"][f"rec_{key}"] = {"reference": "rec_fp32", "metric": "top1", "retention": float(ok.mean() / ref.mean()), **ci(ratios)}
        print(f"rec_{key}", json.dumps(results["pairs"][f"rec_{key}"]), flush=True)
    detector_pairs(EXTRA_PAIRS)
    if args.block > 0:
        sequences = [json.loads(line)["sequence"] for line in args.manifest.read_text(encoding="utf-8").splitlines()]
        block_rng = np.random.default_rng(args.seed + 1)
        results["block"] = {"length_frames": args.block, "seed": args.seed + 1, "pairs": {}}
        for ref_key, key in PAIRS + EXTRA_PAIRS:
            if len(sequences) != len(flags[ref_key].gt):
                raise SystemExit("manifest and predictions differ in length")
            ratios = []
            for _ in range(args.resamples):
                counts = block_counts(sequences, args.block, block_rng)
                ratios.append(flags[key].map50_95(counts) / flags[ref_key].map50_95(counts))
            results["block"]["pairs"][key] = {"reference": ref_key, **ci(np.array(ratios))}
            print("block", key, json.dumps(results["block"]["pairs"][key]), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
