import json

import cv2
import numpy as np
import pytest

from scripts.paper.build_test_split import build_manifests


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))


def _yolo(root, split, sequence, stem, cls, value):
    base = root / "data" / "yolo_signs_v2"
    (base / "dataset.yaml").parent.mkdir(parents=True, exist_ok=True)
    (base / "dataset.yaml").write_text(
        "names: ['traffic_sign', 'traffic_light']\n", encoding="utf-8"
    )
    name = f"{sequence}__{stem}"
    _image(base / "images" / split / f"{name}.jpg", value)
    label = base / "labels" / split / f"{name}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(f"{cls} 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def _test_frame(root, sequence, stem, cls, value):
    base = root / "data" / "aihub_traffic" / "test"
    _image(base / "images" / sequence / f"{stem}.jpg", value)
    label = base / "labels" / sequence / f"{stem}.json"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(
        json.dumps(
            {
                "image": {"imsize": [8, 8]},
                "annotation": [
                    {"class": cls, "box": [1, 1, 6, 6]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_builds_disjoint_manifests_with_class_counts(tmp_path):
    _yolo(tmp_path, "train", "train_day", "001", 0, 10)
    _yolo(tmp_path, "val", "cal_night", "002", 1, 20)
    _test_frame(tmp_path, "test_day", "003", "traffic_sign", 30)
    out = tmp_path / "evidence"

    summary = build_manifests(tmp_path, out, n_calib=1)

    assert summary["splits"]["train"]["images"] == 1
    assert summary["splits"]["calibration"]["objects_by_class"] == {
        "traffic_sign": 0,
        "traffic_light": 1,
    }
    assert summary["splits"]["test"]["objects_by_class"] == {
        "traffic_sign": 1,
        "traffic_light": 0,
    }
    test = json.loads((out / "test_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert test["boxes_xyxy"] == [[1, 1, 6, 6]]
    assert test["image_rel"].startswith("data/aihub_traffic/test/images/")
    first_bytes = {p.name: p.read_bytes() for p in out.iterdir()}
    build_manifests(tmp_path, out, n_calib=1)
    assert first_bytes == {p.name: p.read_bytes() for p in out.iterdir()}


def test_rejects_sequence_overlap(tmp_path):
    _yolo(tmp_path, "train", "same_seq", "001", 0, 10)
    _yolo(tmp_path, "val", "cal_seq", "002", 1, 20)
    _test_frame(tmp_path, "same_seq", "003", "traffic_sign", 30)

    with pytest.raises(ValueError, match="sequence overlap"):
        build_manifests(tmp_path, tmp_path / "evidence", n_calib=1)


def test_rejects_exact_image_overlap_across_sequences(tmp_path):
    _yolo(tmp_path, "train", "train_seq", "001", 0, 10)
    _yolo(tmp_path, "val", "cal_seq", "002", 1, 20)
    _test_frame(tmp_path, "test_seq", "003", "traffic_sign", 10)

    with pytest.raises(ValueError, match="image hash overlap"):
        build_manifests(tmp_path, tmp_path / "evidence", n_calib=1)
