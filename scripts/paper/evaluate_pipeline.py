"""Measure YOLO26 -> ByteTrack -> KoreanSignNet on independent test sequences.

Uses in-memory decoded BGR frames for pipeline timing; disk decode is timed
separately. Fine-class accuracy matches tracks to manual boxes, but no manual
identity GT exists, so this script does not compute MOTA/IDF1/HOTA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.benchmark_runtime import summarize
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_qdq_detection import _preprocess, decode_v4, iou
from scripts.paper.evaluate_recognition import annotation_class
from src.track.bytetrack import ByteTracker

DETECTORS = {
    "fp32": "model_space/yolo_v4_signs_fp32.onnx",
    "head_excluded_qdq": "model_space/yolo_v4_signs_int8_head_excluded.onnx",
}
RECOGNIZER = "model_space/korean_sign_net_fp32.onnx"


def _recognize(
    image: np.ndarray,
    box: np.ndarray,
    coarse_class: int,
    session: ort.InferenceSession,
    classes: dict,
) -> tuple[int, float] | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    rgb = cv2.cvtColor(cv2.resize(image[y1:y2, x1:x2], (32, 32)), cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    candidates = classes["sign_ids"] if coarse_class == 0 else classes["light_ids"]
    chosen = max(candidates, key=lambda idx: float(logits[idx]))
    e = np.exp(logits - logits.max())
    confidence = float(e[chosen] / e.sum())
    return chosen, confidence


def evaluate(
    artifact_root: Path,
    manifest: Path,
    output: Path,
    *,
    detector: str,
    limit_frames: int | None = None,
    warmup: int = 10,
    threads: int = 4,
) -> dict:
    if detector not in DETECTORS or warmup < 1 or threads < 1:
        raise ValueError("valid detector, positive warmup and threads required")
    root = artifact_root.resolve(strict=True)
    frames = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if limit_frames is not None:
        frames = frames[:limit_frames]
    if not frames:
        raise ValueError("test manifest empty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite pipeline evidence: {output}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    det_model = root / DETECTORS[detector]
    rec_model = root / RECOGNIZER
    detect = ort.InferenceSession(
        str(det_model), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    recognize = ort.InferenceSession(
        str(rec_model), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    classes = json.loads((root / "data/roi_cls/classes.json").read_text(encoding="utf-8"))
    first = cv2.imread(str(root / frames[0]["image_rel"]))
    if first is None:
        raise ValueError("warmup image unreadable")
    first_tensor = _preprocess(first)
    for _ in range(warmup):
        detect.run(None, {detect.get_inputs()[0].name: first_tensor})
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "detector": DETECTORS[detector],
        "detector_sha256": sha256(det_model),
        "recognizer": RECOGNIZER,
        "recognizer_sha256": sha256(rec_model),
        "detector_size_bytes": det_model.stat().st_size,
        "recognizer_size_bytes": rec_model.stat().st_size,
        "total_deployed_model_size_bytes": det_model.stat().st_size + rec_model.stat().st_size,
        "test_manifest_sha256": sha256(manifest),
        "frames": len(frames),
        "subset": limit_frames is not None,
        "warmup_count": warmup,
        "threads": threads,
        "execution_provider": "CPUExecutionProvider",
        "input_resolution": [640, 640],
        "confidence_threshold": 0.25,
        "ByteTrack": {
            "track_thresh": 0.25,
            "match_thresh": 0.8,
            "low_thresh": 0.1,
            "track_buffer": 30,
        },
        "timing_scope": "in-memory decoded BGR frame -> detector -> ByteTrack -> recognizer; disk decode separate; excludes transfer/render",
        "recognition_scope": "predicted track ROI, coarse-class-limited 14-class top-1; matched to manual object boxes at IoU>=0.5",
    }
    (output / "config.yaml").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(
        f"onnxruntime={ort.__version__}\nproviders={detect.get_providers()}\n",
        encoding="utf-8",
    )
    traces, totals = (
        [],
        {
            "gt_objects": 0,
            "mapped_gt_objects": 0,
            "matched_tracks": 0,
            "correct_fine_class": 0,
            "recognizer_calls": 0,
        },
    )
    sequence = None
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for frame_index, row in enumerate(frames):
            t_read = time.perf_counter()
            image = cv2.imread(str(root / row["image_rel"]))
            t_decoded = time.perf_counter()
            if image is None:
                raise ValueError(f"image unreadable: {row['image_rel']}")
            if row["sequence"] != sequence:
                tracker = ByteTracker(
                    track_thresh=0.25,
                    match_thresh=0.8,
                    track_buffer=30,
                    frame_rate=30,
                    low_thresh=0.1,
                )
                sequence = row["sequence"]
            tensor = _preprocess(image)
            raw = detect.run(None, {detect.get_inputs()[0].name: tensor})[0]
            height, width = image.shape[:2]
            dets, _ = decode_v4(raw, width=width, height=height, min_conf=0.1)
            t_detect = time.perf_counter()
            array = np.asarray(
                [[*d["box_xyxy"], d["confidence"], d["class_id"]] for d in dets], dtype=np.float32
            ).reshape(-1, 6)
            tracks = tracker.update(array)
            t_track = time.perf_counter()
            outputs = []
            for track in tracks:
                recognition = _recognize(image, track.tlbr, int(track.cls), recognize, classes)
                if recognition is None:
                    continue
                totals["recognizer_calls"] += 1
                outputs.append(
                    {
                        "track_id": int(track.track_id),
                        "coarse_class": int(track.cls),
                        "box_xyxy": [float(x) for x in track.tlbr],
                        "fine_class": int(recognition[0]),
                        "confidence": float(recognition[1]),
                    }
                )
            t_recognized = time.perf_counter()
            ann = json.loads((root / row["label_rel"]).read_text(encoding="utf-8"))
            gt = []
            for object_ann in ann.get("annotation", []):
                coarse_name = object_ann.get("class")
                if coarse_name not in ("traffic_sign", "traffic_light"):
                    continue
                fine = annotation_class(object_ann)
                gt.append(
                    {
                        "coarse_class": 0 if coarse_name == "traffic_sign" else 1,
                        "fine_class": fine,
                        "box_xyxy": object_ann["box"][:4],
                    }
                )
            totals["gt_objects"] += len(gt)
            totals["mapped_gt_objects"] += sum(g["fine_class"] is not None for g in gt)
            candidates = sorted(
                (
                    (iou(track["box_xyxy"], obj["box_xyxy"]), ti, gi)
                    for ti, track in enumerate(outputs)
                    for gi, obj in enumerate(gt)
                    if track["coarse_class"] == obj["coarse_class"]
                ),
                reverse=True,
            )
            used_tracks, used_gt = set(), set()
            for score, ti, gi in candidates:
                if score < 0.5:
                    break
                if ti in used_tracks or gi in used_gt:
                    continue
                used_tracks.add(ti)
                used_gt.add(gi)
                outputs[ti]["matched_gt_index"] = gi
                outputs[ti]["matched_gt_iou"] = score
                if gt[gi]["fine_class"] is not None:
                    totals["matched_tracks"] += 1
                    totals["correct_fine_class"] += int(
                        outputs[ti]["fine_class"] == gt[gi]["fine_class"]
                    )
            trace = {
                "frame_index": frame_index,
                "sequence": sequence,
                "decode_ms": (t_decoded - t_read) * 1000,
                "detect_ms": (t_detect - t_decoded) * 1000,
                "track_ms": (t_track - t_detect) * 1000,
                "recognize_ms": (t_recognized - t_track) * 1000,
                "pipeline_total_ms": (t_recognized - t_decoded) * 1000,
                "with_decode_ms": (t_recognized - t_read) * 1000,
                "track_count": len(outputs),
            }
            traces.append(trace)
            predictions.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "image_rel": row["image_rel"],
                        "tracks": outputs,
                        "ground_truth": gt,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if (frame_index + 1) % 500 == 0:
                print(f"pipeline {frame_index + 1}/{len(frames)} frames", flush=True)
    with (output / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    result = {
        "status": "completed",
        "frames": len(frames),
        "counts": totals,
        "fine_top1_on_matched_tracks": totals["correct_fine_class"]
        / max(1, totals["matched_tracks"]),
        "end_to_end_fine_correct_per_mapped_gt": totals["correct_fine_class"]
        / max(1, totals["mapped_gt_objects"]),
        "timing": {
            key: summarize([float(x[key]) for x in traces])
            for key in (
                "decode_ms",
                "detect_ms",
                "track_ms",
                "recognize_ms",
                "pipeline_total_ms",
                "with_decode_ms",
            )
        },
        "model_size_bytes": config["total_deployed_model_size_bytes"],
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", choices=list(DETECTORS), default="head_excluded_qdq")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = evaluate(
        args.artifact_root,
        args.manifest,
        args.output,
        detector=args.detector,
        limit_frames=args.limit_frames,
        warmup=args.warmup,
        threads=args.threads,
    )
    print(
        {
            "frames": result["frames"],
            "pipeline_fps": result["timing"]["pipeline_total_ms"]["fps"],
            "fine_accuracy_on_matched": result["fine_top1_on_matched_tracks"],
            "fine_correct_per_gt": result["end_to_end_fine_correct_per_mapped_gt"],
        }
    )


if __name__ == "__main__":
    main()
