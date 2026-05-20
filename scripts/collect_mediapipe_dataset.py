import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    raise RuntimeError("mediapipe is required: pip install mediapipe") from exc

POSE_MAPPING = [
    0,        # Nose
    [11, 12], # Neck (mid-shoulder)
    12,       # R-Shoulder
    14,       # R-Elbow
    16,       # R-Wrist
    11,       # L-Shoulder
    13,       # L-Elbow
    15,       # L-Wrist
    [23, 24], # MidHip
    24,       # R-Hip
    26,       # R-Knee
    28,       # R-Ankle
    23,       # L-Hip
    25,       # L-Knee
    27,       # L-Ankle
    5,        # R-Eye
    2,        # L-Eye
    8,        # R-Ear
    7,        # L-Ear
    31,       # L-BigToe
    29,       # L-SmallToe
    31,       # L-Heel (approx)
    32,       # R-BigToe
    30,       # R-SmallToe
    32,       # R-Heel (approx)
]

FACE_POINTS = 70


def add_2d(features, offset, point):
    if point is None:
        return offset + 3
    features[offset] = point.x
    features[offset + 1] = point.y
    vis = getattr(point, "visibility", 1.0)
    features[offset + 2] = vis if vis is not None else 1.0
    return offset + 3


def add_3d(features, offset, point):
    if point is None:
        return offset + 4
    features[offset] = point.x
    features[offset + 1] = point.y
    features[offset + 2] = point.z if point.z is not None else 0.0
    vis = getattr(point, "visibility", 1.0)
    features[offset + 3] = vis if vis is not None else 1.0
    return offset + 4


def avg_point(points):
    if not points:
        return None
    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)
    z = sum(getattr(p, "z", 0.0) for p in points) / len(points)
    vis_list = [getattr(p, "visibility", 1.0) for p in points]
    vis = min(vis_list) if vis_list else 1.0
    class Dummy:
        pass
    p = Dummy()
    p.x, p.y, p.z, p.visibility = x, y, z, vis
    return p


def extract_features(results):
    features = np.zeros(959, dtype=np.float32)
    offset = 0

    pose = results.pose_landmarks.landmark if results.pose_landmarks else None
    face = results.face_landmarks.landmark if results.face_landmarks else None
    left = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
    right = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

    pose_world = (
        results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
    )

    # Pose 2D (25 * 3 = 75)
    if pose:
        for mapping in POSE_MAPPING:
            if isinstance(mapping, list):
                pts = [pose[i] for i in mapping if i < len(pose)]
                offset = add_2d(features, offset, avg_point(pts))
            else:
                point = pose[mapping] if mapping < len(pose) else None
                offset = add_2d(features, offset, point)
    else:
        offset += 75

    # Face 2D (70 * 3 = 210)
    if face:
        for i in range(FACE_POINTS):
            point = face[i] if i < len(face) else None
            offset = add_2d(features, offset, point)
    else:
        offset += 210

    # Left Hand 2D (21 * 3 = 63)
    if left:
        for i in range(21):
            point = left[i] if i < len(left) else None
            offset = add_2d(features, offset, point)
    else:
        offset += 63

    # Right Hand 2D (21 * 3 = 63)
    if right:
        for i in range(21):
            point = right[i] if i < len(right) else None
            offset = add_2d(features, offset, point)
    else:
        offset += 63

    # Pose 3D (25 * 4 = 100)
    if pose_world:
        for mapping in POSE_MAPPING:
            if isinstance(mapping, list):
                pts = [pose_world[i] for i in mapping if i < len(pose_world)]
                offset = add_3d(features, offset, avg_point(pts))
            else:
                point = pose_world[mapping] if mapping < len(pose_world) else None
                offset = add_3d(features, offset, point)
    else:
        offset += 100

    # Face 3D (70 * 4 = 280)
    if face:
        for i in range(FACE_POINTS):
            point = face[i] if i < len(face) else None
            offset = add_3d(features, offset, point)
    else:
        offset += 280

    # Left Hand 3D (21 * 4 = 84)
    if left:
        for i in range(21):
            point = left[i] if i < len(left) else None
            offset = add_3d(features, offset, point)
    else:
        offset += 84

    # Right Hand 3D (21 * 4 = 84)
    if right:
        for i in range(21):
            point = right[i] if i < len(right) else None
            offset = add_3d(features, offset, point)
    else:
        offset += 84

    if offset != 959:
        raise RuntimeError(f"Feature length mismatch: {offset}")

    return features


def load_labels(labels_path):
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_labels(labels_path, labels):
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def append_manifest(manifest_path, rows):
    file_exists = manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "label_id", "label"])
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Collect MediaPipe keypoint dataset.")
    parser.add_argument("--label", required=True, help="Label name to record.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to record.")
    parser.add_argument("--frames", type=int, default=40, help="Frames per sample.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--out-dir", default="./dataset/mediapipe", help="Output directory.")
    parser.add_argument("--min-det", type=float, default=0.5, help="Min detection confidence.")
    parser.add_argument("--min-track", type=float, default=0.5, help="Min tracking confidence.")
    parser.add_argument("--model-complexity", type=int, default=1, help="Holistic model complexity (0-2).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.json"
    manifest_path = out_dir / "manifest.csv"

    labels = load_labels(labels_path)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    if args.label not in label_to_id:
        label_to_id[args.label] = len(labels)
        labels.append(args.label)
        save_labels(labels_path, labels)

    label_id = label_to_id[args.label]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
    )

    print("Press 'r' to record a sample. Press 'q' to quit.")
    saved = 0
    recording = False
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        status = "Recording" if recording else "Idle"
        cv2.putText(display, f"Label: {args.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(display, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(display, f"Saved: {saved}/{args.samples}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imshow("MediaPipe Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and not recording:
            recording = True
            buffer = []

        if recording:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            features = extract_features(results)
            buffer.append(features)

            if len(buffer) >= args.frames:
                sample = np.stack(buffer, axis=0)
                ts = int(time.time() * 1000)
                filename = f"mp_{args.label}_{ts}_{saved:03d}.npz"
                out_path = out_dir / filename
                np.savez_compressed(out_path, data=sample)
                norm_path = str(out_path.resolve()).replace("\\", "/")
                append_manifest(manifest_path, [(norm_path, label_id, args.label)])
                saved += 1
                recording = False
                buffer = []

                if saved >= args.samples:
                    break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
