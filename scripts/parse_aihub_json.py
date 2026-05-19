import os
import json
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

try:
    import orjson as _orjson
except Exception:
    _orjson = None

KP_INDEX = None
KEYPOINT_BASE_DIR = None
OUT_DIR = None


def load_json(path):
    if _orjson is not None:
        with open(path, 'rb') as f:
            return _orjson.loads(f.read())
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_features(json_path):
    try:
        data = load_json(json_path)
        if 'people' not in data or not data['people']:
            return None
        
        p = data['people']
        # Depending on format, 'people' might be a dict or list
        if isinstance(p, list):
            p = p[0] if len(p) > 0 else {}
            
        features = []
        # Extract both 2D and 3D features (Total 959 dims)
        keys = [
            'pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
            'pose_keypoints_3d', 'face_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d'
        ]
        for k in keys:
            if k in p and p[k]:
                features.extend(p[k])
            else:
                # If missing, fill with zeros based on expected lengths
                expected_len = {
                    'pose_keypoints_2d': 75, 'face_keypoints_2d': 210, 'hand_left_keypoints_2d': 63, 'hand_right_keypoints_2d': 63,
                    'pose_keypoints_3d': 100, 'face_keypoints_3d': 280, 'hand_left_keypoints_3d': 84, 'hand_right_keypoints_3d': 84
                }[k]
                features.extend([0.0] * expected_len)
        return features
    except Exception as e:
        return None

def process_morpheme(morpheme_file):
    try:
        m_data = load_json(morpheme_file)
            
        if not m_data.get('data'):
            return None
            
        stem = morpheme_file.stem.replace("_morpheme", "")
        out_path = OUT_DIR / f"{stem}.npz"
        norm_path = str(out_path.absolute()).replace("\\", "/")
        
        word_info = m_data['data'][0]
        label = word_info['attributes'][0]['name']
        
        # [이어하기 기능] 이미 생성된 npz 파일이 있으면 키포인트 파싱을 생략하고 경로와 라벨만 반환
        if out_path.exists():
            return (norm_path, label)
        
        # 1. 뼈대 폴더 찾기 최적화 (하드코딩된 규칙 우선 시도)
        # 예: morpheme_file이 ...\morpheme\01\NIA_SL... 인 경우
        # keypoint_dir은 ...\01_real_word_keypoint\01\NIA_SL...
        parent_num = morpheme_file.parent.name
        kp_dir = KEYPOINT_BASE_DIR / "[라벨]01_real_word_keypoint" / parent_num / stem
        
        if not kp_dir.exists():
            # 실패 시 사전 인덱스에서 조회
            kp_dir = KP_INDEX.get(stem)
            if not kp_dir:
                return None
            
        word_info = m_data['data'][0]
        start_time = word_info['start']
        end_time = word_info['end']
        label = word_info['attributes'][0]['name']
        
        # 30 fps
        start_frame = int(start_time * 30)
        end_frame = int(end_time * 30)
        
        sequence = []
        # 프레임별 JSON 읽기
        for f_idx in range(start_frame, end_frame + 1):
            f_name = f"{stem}_{f_idx:012d}_keypoints.json"
            f_path = kp_dir / f_name
            if f_path.exists():
                feats = extract_features(f_path)
                if feats is not None:
                    sequence.append(feats)
        
        if not sequence:
            return None
            
        sequence = np.array(sequence, dtype=np.float32)
        out_path = OUT_DIR / f"{stem}.npz"
        np.savez_compressed(out_path, data=sequence)
        
        # Windows 경로를 / 로 통일
        norm_path = str(out_path.absolute()).replace("\\", "/")
        return (norm_path, label)
    except Exception as e:
        return None


def build_keypoint_index(keypoint_base_dir):
    kp_root = keypoint_base_dir / "[라벨]01_real_word_keypoint"
    search_root = kp_root if kp_root.exists() else keypoint_base_dir
    index = {}
    for p in search_root.rglob("*"):
        if p.is_dir():
            index.setdefault(p.name, p)
    return index


def init_worker(kp_index, keypoint_base_dir, out_dir):
    global KP_INDEX, KEYPOINT_BASE_DIR, OUT_DIR
    KP_INDEX = kp_index
    KEYPOINT_BASE_DIR = keypoint_base_dir
    OUT_DIR = out_dir

def main():
    base_dir = Path("AIhub/수어 영상/1.Training")
    morpheme_files = list(base_dir.rglob("*_morpheme.json"))
    
    out_dir = Path("dataset/landmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    labels_path = out_dir / "labels.json"
    
    print(f"Found {len(morpheme_files)} morpheme JSON files.")

    print("Building keypoint index...")
    kp_index = build_keypoint_index(base_dir)
    results = []
    existing_paths = set()
    labels = []
    label_to_id = {}

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        label_to_id = {l: i for i, l in enumerate(labels)}

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_paths.add(row["path"])
    
    # 병렬 처리로 초고속 JSON 파싱
    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker,
        initargs=(kp_index, base_dir, out_dir),
    ) as executor:
        for res in tqdm(
            executor.map(process_morpheme, morpheme_files, chunksize=100),
            total=len(morpheme_files),
            desc="Parsing JSON Data",
        ):
            if res:
                results.append(res)
                
    if not results:
        print("No new data parsed. Existing manifest/labels left unchanged.")
        return

    new_rows = []
    for path, label in results:
        if path in existing_paths:
            continue
        if label not in label_to_id:
            label_to_id[label] = len(labels)
            labels.append(label)
        new_rows.append((path, label_to_id[label], label))

    if not new_rows:
        print("No new samples to append. Existing manifest/labels left unchanged.")
        return

    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["path", "label_id", "label"])
        for row in new_rows:
            writer.writerow(row)

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n[Success] Incremental processing complete!")
    print(f"- Total unique classes (Labels): {len(labels)}")
    print(f"- New extracted samples: {len(new_rows)}")
    print(f"- Saved Manifest: {manifest_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
