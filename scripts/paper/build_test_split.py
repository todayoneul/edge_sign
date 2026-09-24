"""Freeze sequence-disjoint YOLO26 train/calibration/test image manifests.

Reads ignored artifacts from --artifact-root; never copies or changes source images.
The test labels come from original AI Hub JSON and use the v3/v4 two-class
taxonomy (traffic_sign=0, traffic_light=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

CLASSES = {0: "traffic_sign", 1: "traffic_light"}
SOURCE = "AI Hub traffic light and road sign recognition video"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence(name: str) -> str:
    if "__" not in name:
        raise ValueError(f"generated YOLO filename has no sequence prefix: {name}")
    return name.split("__", 1)[0]


def _lighting(sequence: str) -> str:
    if "night" in sequence.lower():
        return "night"
    if "daylight" in sequence.lower():
        return "daylight"
    return "unknown"


def _yolo_records(root: Path, split: str, limit: int | None = None) -> list[dict]:
    dataset = root / "data" / "yolo_signs_v2"
    images = sorted((dataset / "images" / split).glob("*.jpg"))
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"no {split} images in {dataset}")
    records = []
    for image in images:
        label = dataset / "labels" / split / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"missing training/calibration label: {label}")
        boxes = []
        classes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"malformed YOLO label in {label}: {line}")
            cls = int(fields[0])
            if cls not in CLASSES:
                raise ValueError(f"class {cls} outside two-class v3/v4 taxonomy: {label}")
            box = [float(v) for v in fields[1:]]
            if not all(0 <= v <= 1 for v in box):
                raise ValueError(f"out-of-range YOLO box: {label}")
            classes.append(cls)
            boxes.append(box)
        with Image.open(image) as im:
            width, height = im.size
        seq = _sequence(image.stem)
        records.append({
            "image_rel": image.relative_to(root).as_posix(),
            "label_rel": label.relative_to(root).as_posix(),
            "sequence": seq,
            "lighting": _lighting(seq),
            "width": width,
            "height": height,
            "classes": classes,
            "boxes_yolo": boxes,
            "image_sha256": _hash(image),
            "source_dataset": SOURCE,
        })
    return records


def _test_records(root: Path, limit: int | None = None) -> tuple[list[dict], list[dict]]:
    dataset = root / "data" / "aihub_traffic" / "test"
    images = sorted((dataset / "images").rglob("*.jpg"))
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"no independent test images in {dataset}")
    records, exclusions = [], []
    for image in images:
        rel = image.relative_to(dataset / "images")
        label = dataset / "labels" / rel.with_suffix(".json")
        if not label.is_file():
            exclusions.append({"image_rel": image.relative_to(root).as_posix(),
                               "reason": "missing JSON annotation"})
            continue
        try:
            data = json.loads(label.read_text(encoding="utf-8"))
            size = data["image"]["imsize"]
            width, height = int(size[0]), int(size[1])
            if width <= 0 or height <= 0:
                raise ValueError("invalid image dimensions")
            classes, boxes = [], []
            for ann in data.get("annotation", []):
                cls_name = ann.get("class")
                if cls_name not in CLASSES.values():
                    continue
                xyxy = ann.get("box", [])
                if len(xyxy) < 4:
                    raise ValueError("missing bounding-box coordinates")
                x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("nonpositive bounding box")
                classes.append(0 if cls_name == "traffic_sign" else 1)
                boxes.append([x1, y1, x2, y2])
            seq = rel.parent.name
            records.append({
                "image_rel": image.relative_to(root).as_posix(),
                "label_rel": label.relative_to(root).as_posix(),
                "sequence": seq,
                "lighting": _lighting(seq),
                "width": width,
                "height": height,
                "classes": classes,
                "boxes_xyxy": boxes,
                "image_sha256": _hash(image),
                "source_dataset": SOURCE,
            })
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            exclusions.append({"image_rel": image.relative_to(root).as_posix(),
                               "reason": str(exc)})
    if not records:
        raise ValueError("independent test split is empty after validation")
    return records, exclusions


def _summary(records: list[dict]) -> dict:
    class_counts = Counter(CLASSES[c] for row in records for c in row["classes"])
    lighting = Counter(row["lighting"] for row in records)
    sequences = sorted({row["sequence"] for row in records})
    return {
        "images": len(records),
        "objects": sum(len(row["classes"]) for row in records),
        "objects_by_class": {name: class_counts[name] for name in CLASSES.values()},
        "sequence_count": len(sequences),
        "sequences": sequences,
        "images_by_lighting": dict(sorted(lighting.items())),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifests(
    artifact_root: Path, output: Path, *, n_calib: int = 150,
    limit_test: int | None = None,
) -> dict:
    if n_calib < 1 or (limit_test is not None and limit_test < 1):
        raise ValueError("n_calib and limit_test must be positive")
    root = artifact_root.resolve(strict=True)
    if not (root / "data" / "yolo_signs_v2" / "dataset.yaml").is_file():
        raise ValueError(f"v3/v4 dataset.yaml missing under {root}")
    train = _yolo_records(root, "train")
    calibration = _yolo_records(root, "val", n_calib)
    test, exclusions = _test_records(root, limit_test)
    splits = {"train": train, "calibration": calibration, "test": test}
    sequences = {name: {row["sequence"] for row in rows} for name, rows in splits.items()}
    hashes = {name: {row["image_sha256"] for row in rows} for name, rows in splits.items()}
    for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if sequences[a] & sequences[b]:
            raise ValueError(f"sequence overlap between {a} and {b}: {sequences[a] & sequences[b]}")
        if hashes[a] & hashes[b]:
            raise ValueError(f"image hash overlap between {a} and {b}")
    summary = {
        "taxonomy": CLASSES,
        "source_dataset": SOURCE,
        "selection_rule": {
            "train": "existing yolo_signs_v2/images/train, sorted filenames",
            "calibration": f"first {n_calib} sorted yolo_signs_v2/images/val frames",
            "test": "all AI Hub test sequences and frames sorted by relative path"
                    if limit_test is None else f"first {limit_test} sorted AI Hub test frames (dry run)",
            "random_seed": None,
            "test_negative_frames_included": True,
        },
        "splits": {name: _summary(rows) for name, rows in splits.items()},
        "test_exclusions": exclusions,
        "leakage_checks": "sequence names and exact JPEG SHA-256 disjoint across all splits",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        _write_jsonl(output / f"{name}_manifest.jsonl", rows)
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/splits"))
    parser.add_argument("--n-calib", type=int, default=150)
    parser.add_argument("--limit-test", type=int, default=None,
                        help="dry run using the first N test frames; use a separate output dir")
    args = parser.parse_args()
    summary = build_manifests(args.artifact_root, args.output,
                              n_calib=args.n_calib, limit_test=args.limit_test)
    print(json.dumps(summary["splits"], ensure_ascii=False, sort_keys=True))
    print(f"test exclusions: {len(summary['test_exclusions'])}")


if __name__ == "__main__":
    main()
