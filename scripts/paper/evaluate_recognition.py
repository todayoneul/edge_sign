"""Evaluate KoreanSignNet on independent AI Hub test-sequence ground-truth ROIs.

The 14-class annotation mapping and 8% ROI margin mirror
scripts/prepare_korean_traffic.py. This is oracle-box classification, not
end-to-end detection-conditioned recognition accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256

MODELS = {
    "fp32": "model_space/korean_sign_net_fp32.onnx",
    "w8a8": "model_space/korean_sign_net_w8a8.onnx",
}


def annotation_class(ann: dict) -> int | None:
    if ann.get("class") == "traffic_sign":
        sign_type = ann.get("type", "")
        text = str(ann.get("text", "")).strip()
        if sign_type == "restriction":
            return (
                ["30", "40", "50", "60", "70", "80"].index(text)
                if text in ("30", "40", "50", "60", "70", "80")
                else 6
            )
        return {"instruction": 7, "caution": 8}.get(sign_type)
    if ann.get("class") == "traffic_light":
        attrs = ann.get("attribute") or [{}]
        on = [key for key, value in attrs[0].items() if value == "on"]
        if any("left_arrow" in key for key in on):
            return 12
        for key, class_id in (("red", 9), ("yellow", 11), ("green", 10)):
            if key in on:
                return class_id
        return 13
    return None


def _crop(image: np.ndarray, box: list[float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    mx, my = max(2, int((x2 - x1) * 0.08)), max(2, int((y2 - y1) * 0.08))
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(width, x2 + mx), min(height, y2 + my)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.resize(image[y1:y2, x1:x2], (32, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose((rgb - 0.5) / 0.5, (2, 0, 1))[None]


def evaluate(
    artifact_root: Path, manifest: Path, output: Path, *, limit_frames: int | None = None
) -> dict:
    root = artifact_root.resolve(strict=True)
    frames = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if limit_frames is not None:
        frames = frames[:limit_frames]
    if not frames:
        raise ValueError("test manifest is empty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite recognition evidence: {output}")
    classes = json.loads((root / "data/roi_cls/classes.json").read_text(encoding="utf-8"))["names"]
    if len(classes) != 14:
        raise ValueError(f"expected KoreanSignNet 14 classes, got {len(classes)}")
    sessions = {}
    unavailable = {}
    for name, relative in MODELS.items():
        try:
            session = ort.InferenceSession(str(root / relative), providers=["CPUExecutionProvider"])
            if session.get_outputs()[0].shape[-1] != 14:
                raise ValueError(f"unexpected output class count: {name}")
            sessions[name] = session
        except Exception as exc:
            unavailable[name] = f"{type(exc).__name__}: {exc}"
    if not sessions:
        raise RuntimeError(f"no runnable recognition model: {unavailable}")
    output.mkdir(parents=True, exist_ok=True)
    for name, reason in unavailable.items():
        (output / f"{name}_metrics.json").write_text(
            json.dumps(
                {
                    "status": "unsupported",
                    "model": MODELS[name],
                    "reason": reason,
                    "onnxruntime_version": ort.__version__,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    streams = {
        name: (output / f"{name}_predictions.jsonl").open("w", encoding="utf-8")
        for name in sessions
    }
    confusion = {name: np.zeros((14, 14), dtype=np.int64) for name in sessions}
    counts = {"annotations_total": 0, "mapped": 0, "unmapped": 0, "invalid_crop": 0}
    try:
        for frame_index, frame in enumerate(frames):
            image = cv2.imread(str(root / frame["image_rel"]))
            if image is None:
                raise ValueError(f"unreadable image: {frame['image_rel']}")
            annotation = json.loads((root / frame["label_rel"]).read_text(encoding="utf-8"))
            for annotation_index, ann in enumerate(annotation.get("annotation", [])):
                if ann.get("class") not in ("traffic_sign", "traffic_light"):
                    continue
                counts["annotations_total"] += 1
                label = annotation_class(ann)
                if label is None:
                    counts["unmapped"] += 1
                    continue
                tensor = _crop(image, ann.get("box", [])) if len(ann.get("box", [])) >= 4 else None
                if tensor is None:
                    counts["invalid_crop"] += 1
                    continue
                counts["mapped"] += 1
                for name, session in sessions.items():
                    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
                    pred = int(np.argmax(logits))
                    confusion[name][label, pred] += 1
                    record = {
                        "frame_index": frame_index,
                        "image_rel": frame["image_rel"],
                        "annotation_index": annotation_index,
                        "sequence": frame["sequence"],
                        "lighting": frame["lighting"],
                        "box_xyxy": ann["box"][:4],
                        "true_class_id": label,
                        "predicted_class_id": pred,
                        "logits": [float(value) for value in logits],
                    }
                    streams[name].write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
            if (frame_index + 1) % 500 == 0:
                print(f"recognized frame {frame_index + 1}/{len(frames)}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    results = {}
    for name, matrix in confusion.items():
        per_class = []
        for class_id, class_name in enumerate(classes):
            total = int(matrix[class_id].sum())
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "correct": int(matrix[class_id, class_id]),
                    "total": total,
                    "accuracy": float(matrix[class_id, class_id] / total) if total else None,
                }
            )
        total = int(matrix.sum())
        result = {
            "model": MODELS[name],
            "model_sha256": sha256(root / MODELS[name]),
            "test_manifest_sha256": sha256(manifest),
            "test_frames": len(frames),
            "subset": limit_frames is not None,
            "evaluation_scope": "oracle ground-truth box classification; independent test sequences",
            "top1_accuracy": float(np.trace(matrix) / total) if total else None,
            "total_rois": total,
            "annotation_counts": counts,
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
            "class_order": classes,
        }
        (output / f"{name}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results[name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/recognition"))
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()
    result = evaluate(
        args.artifact_root, args.manifest, args.output, limit_frames=args.limit_frames
    )
    print(
        {
            name: {"top1_accuracy": value["top1_accuracy"], "total_rois": value["total_rois"]}
            for name, value in result.items()
        }
    )


if __name__ == "__main__":
    main()
