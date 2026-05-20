import argparse
import csv
import json
import os

# Suppress MediaPipe/TensorFlow C++ warnings BEFORE importing mediapipe
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
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
    pose_world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

    # Pose 2D
    if pose:
        for mapping in POSE_MAPPING:
            if isinstance(mapping, list):
                pts = [pose[i] for i in mapping if i < len(pose)]
                offset = add_2d(features, offset, avg_point(pts))
            else:
                point = pose[mapping] if mapping < len(pose) else None
                offset = add_2d(features, offset, point)
    else: offset += 75

    # Face 2D
    if face:
        for i in range(FACE_POINTS):
            point = face[i] if i < len(face) else None
            offset = add_2d(features, offset, point)
    else: offset += 210

    # Left Hand 2D
    if left:
        for i in range(21):
            point = left[i] if i < len(left) else None
            offset = add_2d(features, offset, point)
    else: offset += 63

    # Right Hand 2D
    if right:
        for i in range(21):
            point = right[i] if i < len(right) else None
            offset = add_2d(features, offset, point)
    else: offset += 63

    # Pose 3D
    if pose_world:
        for mapping in POSE_MAPPING:
            if isinstance(mapping, list):
                pts = [pose_world[i] for i in mapping if i < len(pose_world)]
                offset = add_3d(features, offset, avg_point(pts))
            else:
                point = pose_world[mapping] if mapping < len(pose_world) else None
                offset = add_3d(features, offset, point)
    else: offset += 100

    # Face 3D
    if face:
        for i in range(FACE_POINTS):
            point = face[i] if i < len(face) else None
            offset = add_3d(features, offset, point)
    else: offset += 280

    # Left Hand 3D
    if left:
        for i in range(21):
            point = left[i] if i < len(left) else None
            offset = add_3d(features, offset, point)
    else: offset += 84

    # Right Hand 3D
    if right:
        for i in range(21):
            point = right[i] if i < len(right) else None
            offset = add_3d(features, offset, point)
    else: offset += 84

    return features


# Global variables for worker processes
OUT_DIR = None
MORPHEME_INDEX = None
MAX_FRAMES = 300

import sys

def init_worker(out_dir, morpheme_index, max_frames):
    global OUT_DIR, MORPHEME_INDEX, MAX_FRAMES
    OUT_DIR = out_dir
    MORPHEME_INDEX = morpheme_index
    MAX_FRAMES = max_frames
    
    # [핵심] C++ 백엔드(MediaPipe)에서 강제로 쏘는 stderr 출력을 OS 레벨에서 블랙홀(devnull)로 리다이렉션
    sys.stderr = open(os.devnull, 'w')
    try:
        # C++ 수준의 파일 디스크립터까지 닫아버림 (fd 2 = stderr)
        os.dup2(sys.stderr.fileno(), 2)
    except Exception:
        pass

def process_video(video_path):
    results_list = []
    stem = "Unknown"
    try:
        video_path = Path(video_path)
        stem = video_path.stem
        
        morpheme_path = MORPHEME_INDEX.get(stem)
        if not morpheme_path or not morpheme_path.exists():
            return [("ERROR", stem, "Morpheme JSON not found")]
            
        with open(morpheme_path, 'r', encoding='utf-8') as f:
            m_data = json.load(f)
            
        if not m_data.get('data'):
            return [("ERROR", stem, "No 'data' field in morpheme JSON")]

        cap = None
        holistic = None

        for idx, word_info in enumerate(m_data['data']):
            # 1. 안전한 라벨 추출
            attributes = word_info.get('attributes', [])
            if not attributes or 'name' not in attributes[0]:
                results_list.append(("ERROR", stem, f"Missing label name in segment {idx}"))
                continue
            
            label_text = attributes[0]['name']
            start_time = word_info.get('start', 0.0)
            end_time = word_info.get('end', 0.0)
            
            # 2. 비정상 구간 검증
            if end_time <= start_time:
                results_list.append(("ERROR", stem, f"Invalid timestamps ({start_time}-{end_time}) in segment {idx}"))
                continue
                
            file_name = f"{stem}.npz" if len(m_data['data']) == 1 else f"{stem}_{idx}.npz"
            out_file = OUT_DIR / file_name
            norm_path = str(out_file.absolute()).replace("\\", "/")

            if out_file.exists():
                results_list.append(("SUCCESS", norm_path, label_text))
                continue

            if cap is None:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return [("ERROR", stem, "Failed to open video file")]

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30.0

                holistic = mp.solutions.holistic.Holistic(
                    model_complexity=1,
                    smooth_landmarks=True,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # 3. 정확한 프레임 시킹 (처음으로 되돌린 후 순차 읽기)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
            while current_frame < start_frame:
                ret, _ = cap.read()
                if not ret:
                    break
                current_frame += 1
            
            buffer = []
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                features = extract_features(res)
                buffer.append(features)
                current_frame += 1

            if buffer:
                # 4. 시퀀스 길이 제한 및 다운샘플링 적용
                if len(buffer) > MAX_FRAMES:
                    indices = np.linspace(0, len(buffer) - 1, MAX_FRAMES, dtype=int)
                    buffer = [buffer[i] for i in indices]
                    
                sample = np.stack(buffer, axis=0)
                np.savez_compressed(out_file, data=sample)
                results_list.append(("SUCCESS", norm_path, label_text))
            else:
                results_list.append(("ERROR", stem, f"No frames extracted for segment {idx}"))

        if cap is not None:
            cap.release()
        if holistic is not None:
            holistic.close()

        return results_list

    except Exception as e:
        return [("ERROR", stem, f"Exception occurred: {str(e)}")]

def build_morpheme_index(base_dir):
    index = {}
    for p in base_dir.rglob("*_morpheme.json"):
        stem = p.stem.replace("_morpheme", "")
        index[stem] = p
    return index

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from AIhub videos.")
    parser.add_argument("--video-dir", default="AIhub/수어 영상/1.Training", help="Directory containing mp4 files.")
    parser.add_argument("--out-dir", default="./dataset/mediapipe_from_videos", help="Output directory for npz files.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to process (0 for all).")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0 for auto: min(8, cpu//2)).")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames per segment (uniform downsampling).")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    labels_path = out_dir / "labels.json"
    error_log_path = out_dir / "error_log.txt"

    print("Building morpheme index...")
    morpheme_index = build_morpheme_index(video_dir)
    print(f"Found {len(morpheme_index)} morpheme JSON files.")

    video_files = list(video_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files.")

    if not video_files:
        return

    if args.limit > 0:
        video_files = video_files[:args.limit]
        print(f"Processing limited to {args.limit} videos.")
    else:
        print(f"Processing all {len(video_files)} videos.")

    labels = []
    label_to_id = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
            label_to_id = {lbl: idx for idx, lbl in enumerate(labels)}
            
    existing_paths = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row["path"])
    
    # 6. 워커 수 동적 조절 (메모리 OOM 방지)
    workers = args.workers
    if workers <= 0:
        workers = max(1, min(8, os.cpu_count() // 2))

    results = []
    errors = []
    
    print(f"Starting extraction with {workers} workers (Max frames/seg: {args.max_frames})...")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(out_dir, morpheme_index, args.max_frames)
    ) as executor:
        from tqdm import tqdm
        for res_list in tqdm(executor.map(process_video, video_files), total=len(video_files), desc="Extracting"):
            if res_list:
                for item in res_list:
                    if item[0] == "SUCCESS":
                        results.append((item[1], item[2]))
                    elif item[0] == "ERROR":
                        errors.append(f"{item[1]}: {item[2]}")
                        
    # 에러 로그 작성
    if errors:
        with open(error_log_path, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(e + "\n")
        print(f"\n[Warning] Logged {len(errors)} extraction errors to {error_log_path}")

    if results:
        new_labels_set = set()
        for _, label_text in results:
            if label_text not in label_to_id:
                new_labels_set.add(label_text)
                
        for label_text in sorted(list(new_labels_set)):
            label_to_id[label_text] = len(labels)
            labels.append(label_text)

        new_rows = []
        for path, label_text in results:
            if path in existing_paths:
                continue
            
            label_id = label_to_id[label_text]
            new_rows.append((path, label_id, label_text))
            existing_paths.add(path)
            
        if new_rows:
            new_rows.sort(key=lambda x: x[0])
            
            write_header = not manifest_path.exists()
            with open(manifest_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["path", "label_id", "label"])
                for row in new_rows:
                    writer.writerow(row)
                    
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
                
            print(f"Saved {len(new_rows)} new features to {manifest_path}")
            print(f"Updated labels.json. Total classes: {len(labels)}")
        else:
            print("No new files were appended to the manifest.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()