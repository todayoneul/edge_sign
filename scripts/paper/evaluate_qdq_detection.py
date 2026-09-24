"""Evaluate the same independent test frames with three YOLO26 ONNX variants.

The exported YOLO26 output is [1,300,6] xyxy/conf/class at 640 pixels.
It already selects up to 300 detections, so no external NMS is applied.
AP uses confidence >=0.001; reported precision/recall/count use >=0.25.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256

VARIANTS = {
    "yolo26_fp32": "model_space/yolo_v4_signs_fp32.onnx",
    "yolo26_full_qdq": "model_space/yolo_v4_signs_int8_static.onnx",
    "yolo26_head_excluded_qdq": "model_space/yolo_v4_signs_int8_head_excluded.onnx",
}
IOU_THRESHOLDS = [round(0.5 + i * 0.05, 2) for i in range(10)]
CLASSES = {0: "traffic_sign", 1: "traffic_light"}


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_v4(
    output: np.ndarray,
    *,
    width: int,
    height: int,
    min_conf: float = 0.001,
) -> tuple[list[dict], int]:
    if output.shape != (1, 300, 6):
        raise ValueError(f"unsupported YOLO26 output shape: {output.shape}")
    detections, dropped = [], 0
    for x1, y1, x2, y2, confidence, cls in output[0]:
        if not math.isfinite(float(confidence)) or confidence < min_conf:
            continue
        if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2, cls)):
            dropped += 1
            continue
        class_id = int(round(float(cls)))
        if class_id not in CLASSES or abs(float(cls) - class_id) > 0.01:
            dropped += 1
            continue
        box = [
            max(0.0, min(float(width), float(x1) * width / 640)),
            max(0.0, min(float(height), float(y1) * height / 640)),
            max(0.0, min(float(width), float(x2) * width / 640)),
            max(0.0, min(float(height), float(y2) * height / 640)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            dropped += 1
            continue
        detections.append(
            {
                "box_xyxy": box,
                "confidence": float(confidence),
                "class_id": class_id,
                "class_name": CLASSES[class_id],
            }
        )
    return detections, dropped


def _match(
    ground_truth: list[dict],
    predictions: list[list[dict]],
    *,
    class_id: int,
    threshold: float,
    min_conf: float,
) -> tuple[list[int], list[int], int]:
    gt = {
        i: [box for c, box in zip(row["classes"], row["boxes_xyxy"], strict=True) if c == class_id]
        for i, row in enumerate(ground_truth)
    }
    n_gt = sum(map(len, gt.values()))
    used = {i: set() for i in gt}
    all_preds = sorted(
        (
            (float(pred["confidence"]), i, pred["box_xyxy"])
            for i, frame in enumerate(predictions)
            for pred in frame
            if pred["class_id"] == class_id and pred["confidence"] >= min_conf
        ),
        key=lambda item: -item[0],
    )
    tp, fp = [], []
    for _, frame_id, box in all_preds:
        scores = [
            iou(box, target) if j not in used[frame_id] else -1.0
            for j, target in enumerate(gt[frame_id])
        ]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        matched = best is not None and scores[best] >= threshold
        if matched:
            used[frame_id].add(best)
        tp.append(int(matched))
        fp.append(int(not matched))
    return tp, fp, n_gt


def _ap(tp: list[int], fp: list[int], n_gt: int) -> float:
    if n_gt == 0 or not tp:
        return 0.0
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp))
    # Match Ultralytics compute_ap: precision envelope and 101-point trapezoid.
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    recall_grid = np.linspace(0, 1, 101)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(recall_grid, mrec, mpre), recall_grid))


def compute_metrics(
    ground_truth: list[dict],
    predictions: list[list[dict]],
    *,
    task_conf: float = 0.25,
) -> dict:
    if len(ground_truth) != len(predictions):
        raise ValueError("ground truth/prediction frame count differs")
    per_class = {}
    totals = {"tp": 0, "fp": 0, "gt": 0}
    for class_id, name in CLASSES.items():
        aps = []
        for threshold in IOU_THRESHOLDS:
            tp, fp, n_gt = _match(
                ground_truth, predictions, class_id=class_id, threshold=threshold, min_conf=0.001
            )
            aps.append(_ap(tp, fp, n_gt))
        tp, fp, n_gt = _match(
            ground_truth, predictions, class_id=class_id, threshold=0.5, min_conf=task_conf
        )
        counts = {"tp": sum(tp), "fp": sum(fp), "gt": n_gt}
        for key in totals:
            totals[key] += counts[key]
        per_class[name] = {
            "AP50": aps[0],
            "AP50_95": float(np.mean(aps)),
            "precision": counts["tp"] / max(1, counts["tp"] + counts["fp"]),
            "recall": counts["tp"] / max(1, n_gt),
            "detection_count": counts["tp"] + counts["fp"],
            "ground_truth_count": n_gt,
        }
    active = [value for value in per_class.values() if value["ground_truth_count"] > 0]
    return {
        "mAP50": float(np.mean([value["AP50"] for value in active])) if active else 0.0,
        "mAP50_95": float(np.mean([value["AP50_95"] for value in active])) if active else 0.0,
        "precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "recall": totals["tp"] / max(1, totals["gt"]),
        "detection_count": totals["tp"] + totals["fp"],
        "ground_truth_count": totals["gt"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["gt"] - totals["tp"],
        "per_class": per_class,
        "AP_confidence_floor": 0.001,
        "AP_integration": "Ultralytics-compatible 101-point precision-envelope trapezoid",
        "task_confidence_threshold": task_conf,
        "IoU_thresholds": IOU_THRESHOLDS,
    }


def _preprocess(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.transpose(rgb.astype(np.float32) / 255, (2, 0, 1))[None]


def _cosine_and_sqnr(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float | None, float | None]:
    a = reference.astype(np.float64).ravel()
    b = candidate.astype(np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denominator) if denominator else None
    error = float(np.dot(a - b, a - b))
    signal = float(np.dot(a, a))
    sqnr = 10 * math.log10(signal / error) if error > 0 and signal > 0 else None
    return cosine, sqnr


def _fp32_matches(detections: list[dict], baseline: list[dict], task_conf: float) -> None:
    for pred in detections:
        if pred["confidence"] < task_conf:
            continue
        candidates = [
            iou(pred["box_xyxy"], ref["box_xyxy"])
            for ref in baseline
            if ref["class_id"] == pred["class_id"] and ref["confidence"] >= task_conf
        ]
        pred["matched_fp32_iou"] = max(candidates, default=0.0)


def evaluate(
    artifact_root: Path,
    manifest: Path,
    output: Path,
    *,
    limit_frames: int | None = None,
    threads: int = 4,
    task_conf: float = 0.25,
) -> dict:
    root = artifact_root.resolve(strict=True)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if limit_frames is not None:
        rows = rows[:limit_frames]
    if not rows or threads < 1 or not 0 < task_conf < 1:
        raise ValueError("nonempty frames, positive threads, and 0<task_conf<1 required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    for row in rows:
        if not (root / row["image_rel"]).is_file():
            raise FileNotFoundError(root / row["image_rel"])
    sessions = {}
    for name, relative in VARIANTS.items():
        model = root / relative
        if not model.is_file():
            raise FileNotFoundError(model)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        sessions[name] = ort.InferenceSession(
            str(model), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        if sessions[name].get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError(f"unexpected input shape for {name}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(manifest)
    config = {
        "test_manifest_sha256": manifest_hash,
        "frames": len(rows),
        "subset": limit_frames is not None,
        "input_size": [640, 640],
        "preprocessing": "OpenCV BGR decode; stretch resize 640x640; RGB; float32/255; NCHW",
        "postprocessing": "exported top-300 xyxy/conf/class; no additional NMS; clip boxes to image",
        "AP_confidence_floor": 0.001,
        "task_confidence_threshold": task_conf,
        "task_IoU_threshold": 0.5,
        "AP_IoU_thresholds": IOU_THRESHOLDS,
        "execution_provider": "CPUExecutionProvider",
        "intra_op_threads": threads,
        "onnxruntime_version": ort.__version__,
    }
    ground_truth = [{"classes": row["classes"], "boxes_xyxy": row["boxes_xyxy"]} for row in rows]
    predictions = {name: [] for name in VARIANTS}
    dropped = dict.fromkeys(VARIANTS, 0)
    streams = {}
    try:
        for name, relative in VARIANTS.items():
            run_dir = output / name
            run_dir.mkdir()
            model = root / relative
            model_config = dict(
                config,
                model_id=name,
                model_path=relative,
                model_sha256=sha256(model),
                model_size_bytes=model.stat().st_size,
            )
            (run_dir / "config.yaml").write_text(
                json.dumps(model_config, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
            (run_dir / "environment.txt").write_text(
                f"platform={platform.platform()}\npython={sys.version}\n"
                f"onnxruntime={ort.__version__}\nproviders={sessions[name].get_providers()}\n",
                encoding="utf-8",
            )
            streams[name] = (run_dir / "predictions.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            )
        for frame_id, row in enumerate(rows):
            image = cv2.imread(str(root / row["image_rel"]))
            if image is None:
                raise ValueError(f"image unreadable: {row['image_rel']}")
            height, width = image.shape[:2]
            if [width, height] != [row["width"], row["height"]]:
                raise ValueError(f"image dimensions changed: {row['image_rel']}")
            tensor = _preprocess(image)
            outputs = {
                name: sessions[name].run(None, {sessions[name].get_inputs()[0].name: tensor})[0]
                for name in VARIANTS
            }
            decoded = {
                name: decode_v4(value, width=width, height=height)
                for name, value in outputs.items()
            }
            baseline = decoded["yolo26_fp32"][0]
            for name, (detections, n_dropped) in decoded.items():
                _fp32_matches(detections, baseline, task_conf)
                dropped[name] += n_dropped
                predictions[name].append(detections)
                cosine, sqnr = _cosine_and_sqnr(outputs["yolo26_fp32"], outputs[name])
                record = {
                    "frame_index": frame_id,
                    "image_rel": row["image_rel"],
                    "sequence": row["sequence"],
                    "lighting": row["lighting"],
                    "ground_truth": ground_truth[frame_id],
                    "detections": detections,
                    "detection_count_task_threshold": sum(
                        d["confidence"] >= task_conf for d in detections
                    ),
                    "dropped_invalid_predictions": n_dropped,
                    "raw_output_cosine_vs_fp32": cosine,
                    "raw_output_sqnr_db_vs_fp32": sqnr,
                }
                streams[name].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if (frame_id + 1) % 100 == 0:
                print(f"evaluated {frame_id + 1}/{len(rows)} frames", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    results = {}
    for name in VARIANTS:
        result = compute_metrics(ground_truth, predictions[name], task_conf=task_conf)
        result.update(
            frames_evaluated=len(rows), dropped_invalid_predictions=dropped[name], complete=True
        )
        (output / name / "metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        results[name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/detection"))
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--task-conf", type=float, default=0.25)
    args = parser.parse_args()
    result = evaluate(
        args.artifact_root,
        args.manifest,
        args.output,
        limit_frames=args.limit_frames,
        threads=args.threads,
        task_conf=args.task_conf,
    )
    print(
        json.dumps(
            {
                name: {
                    k: value
                    for k, value in row.items()
                    if k in ("mAP50", "mAP50_95", "precision", "recall", "detection_count")
                }
                for name, row in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
