"""End-to-end accuracy of detector -> ByteTrack -> recognizer per precision.

Does a component's quantization loss reach the final fine class? The path and settings
are those of evaluate_pipeline.py and the web demo: YOLO26-n on the 640x640 frame, decode
at conf 0.1, ByteTrack (track 0.25, match 0.8, low 0.1, buffer 30, reset per sequence),
and the recognizer on every track box cropped from the original frame (32x32, no margin)
with a coarse-class-limited top-1. A ground-truth object with a fine label counts as
correct when a track of the same coarse class overlaps it at IoU >= 0.5 (greedy, highest
IoU first) and the track's fine class is right.

Detector outputs do not depend on the recognizer, so each detector runs once and every
recognizer variant classifies the same track boxes. Retention is paired with the FP32
detector + FP32 recognizer on the same frames; its 95% interval comes from a frame
bootstrap (frames of a sequence are correlated, so the interval is optimistic).

Usage: python scripts/paper/evaluate_end_to_end.py --artifact-root <checkout>
Output: paper_evidence/runtime/end_to_end/{summary.json, per_frame.csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_pipeline import _recognize
from scripts.paper.evaluate_qdq_detection import _preprocess, decode_v4, iou
from scripts.paper.evaluate_recognition import annotation_class
from scripts.paper.runtime_matrix import MODELS
from src.track.bytetrack import ByteTracker

DETECTORS = ["v4_fp32", "v4_fp16", "v4_int8_head_excl", "v4_int8_head_excl_fbias", "v4_int8_full"]
RECOGNIZERS = ["rec_fp32", "rec_fp16", "rec_int8_full"]


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def ground_truth(root: Path, row: dict) -> list[dict]:
    ann = json.loads((root / row["label_rel"]).read_text(encoding="utf-8"))
    return [{"coarse_class": 0 if a["class"] == "traffic_sign" else 1, "fine_class": annotation_class(a),
             "box_xyxy": a["box"][:4]} for a in ann.get("annotation", []) if a.get("class") in ("traffic_sign", "traffic_light")]


def score(outputs: list[dict], gt: list[dict]) -> tuple[int, int]:
    """(matched GT with a fine label, correct fine class) with evaluate_pipeline.py's greedy matching."""
    pairs = sorted(((iou(o["box_xyxy"], g["box_xyxy"]), ti, gi) for ti, o in enumerate(outputs)
                    for gi, g in enumerate(gt) if o["coarse_class"] == g["coarse_class"]), reverse=True)
    used_t, used_g, matched, correct = set(), set(), 0, 0
    for value, ti, gi in pairs:
        if value < 0.5:
            break
        if ti in used_t or gi in used_g:
            continue
        used_t.add(ti)
        used_g.add(gi)
        if gt[gi]["fine_class"] is not None:
            matched += 1
            correct += int(outputs[ti]["fine_class"] == gt[gi]["fine_class"])
    return matched, correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/runtime/end_to_end"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    classes = json.loads((root / "data/roi_cls/classes.json").read_text(encoding="utf-8"))
    recognizers = {k: session(root / MODELS[k][1], args.threads) for k in RECOGNIZERS}
    gts = [ground_truth(root, r) for r in rows]
    mapped = np.array([sum(g["fine_class"] is not None for g in gt) for gt in gts])
    per_frame: dict[str, dict[str, np.ndarray]] = {}
    for det_key in DETECTORS:
        detector = session(root / MODELS[det_key][1], args.threads)
        half = detector.get_inputs()[0].type == "tensor(float16)"
        counts = {k: {"tracks": np.zeros(len(rows), int), "matched": np.zeros(len(rows), int),
                      "correct": np.zeros(len(rows), int)} for k in RECOGNIZERS}
        sequence, tracker = None, None
        for i, row in enumerate(rows):
            image = cv2.imread(str(root / row["image_rel"]))
            if row["sequence"] != sequence:
                tracker = ByteTracker(track_thresh=0.25, match_thresh=0.8, track_buffer=30, frame_rate=30, low_thresh=0.1)
                sequence = row["sequence"]
            tensor = _preprocess(image)
            raw = detector.run(None, {detector.get_inputs()[0].name: tensor.astype(np.float16) if half else tensor})[0]
            dets, _ = decode_v4(raw.astype(np.float32), width=image.shape[1], height=image.shape[0], min_conf=0.1)
            array = np.asarray([[*d["box_xyxy"], d["confidence"], d["class_id"]] for d in dets], dtype=np.float32).reshape(-1, 6)
            tracks = tracker.update(array)
            for rec_key, rec in recognizers.items():
                outputs = []
                for t in tracks:
                    r = _recognize(image, t.tlbr, int(t.cls), rec, classes)
                    if r is not None:
                        outputs.append({"coarse_class": int(t.cls), "box_xyxy": [float(x) for x in t.tlbr], "fine_class": int(r[0])})
                m, c = score(outputs, gts[i])
                counts[rec_key]["tracks"][i], counts[rec_key]["matched"][i], counts[rec_key]["correct"][i] = len(outputs), m, c
            if (i + 1) % 500 == 0:
                print(f"{det_key} {i + 1}/{len(rows)}", flush=True)
        for rec_key in RECOGNIZERS:
            per_frame[f"{det_key}+{rec_key}"] = counts[rec_key]
    ref = per_frame["v4_fp32+rec_fp32"]["correct"]
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(rows), (args.resamples, len(rows)))
    summary = {"frames": len(rows), "gt_objects_with_fine_label": int(mapped.sum()), "resamples": args.resamples,
               "seed": args.seed, "note": "frame bootstrap ignores within-sequence correlation (optimistic interval)",
               "models": {k: {"path": MODELS[k][1], "sha256": sha256(root / MODELS[k][1])} for k in DETECTORS + RECOGNIZERS},
               "onnxruntime_version": ort.__version__, "combinations": {}}
    for key, c in per_frame.items():
        ratios = c["correct"][draws].sum(1) / ref[draws].sum(1)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        summary["combinations"][key] = {
            "tracks": int(c["tracks"].sum()), "matched_gt": int(c["matched"].sum()), "correct": int(c["correct"].sum()),
            "fine_top1_on_matched": float(c["correct"].sum() / max(1, c["matched"].sum())),
            "end_to_end_accuracy": float(c["correct"].sum() / mapped.sum()),
            "retention_vs_fp32": float(c["correct"].sum() / ref.sum()), "retention_ci95": [float(lo), float(hi)]}
        s = summary["combinations"][key]
        print(f"{key:32s} matched {s['matched_gt']:5d} correct {s['correct']:5d} e2e {s['end_to_end_accuracy']:.4f} "
              f"cond {s['fine_top1_on_matched']:.4f} retention {s['retention_vs_fp32']:.4f} [{lo:.4f}, {hi:.4f}]")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "image_rel", "gt_with_fine_label"] + [f"{k}:{f}" for k in per_frame for f in ("tracks", "matched", "correct")])
        for i, row in enumerate(rows):
            writer.writerow([i, row["image_rel"], int(mapped[i])] + [int(per_frame[k][f][i]) for k in per_frame for f in ("tracks", "matched", "correct")])


if __name__ == "__main__":
    main()
