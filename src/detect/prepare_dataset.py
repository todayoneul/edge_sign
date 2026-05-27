"""
GTSDB → YOLO 포맷 변환 + AI Hub 데이터 통합 스크립트.

GTSDB 43개 세부 클래스를 Edge-Sign 프로젝트의 2개 상위 클래스로 매핑:
  0: traffic_sign (교통표지판)
  1: signboard    (간판) — AI Hub 한글 간판 데이터 도착 시 추가

출력 구조 (YOLO 포맷):
  data/yolo_signs/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── dataset.yaml

사용법:
  python src/detect/prepare_dataset.py --source gtsdb
  python src/detect/prepare_dataset.py --source aihub_traffic --aihub_dir data/aihub_traffic
  python src/detect/prepare_dataset.py --source aihub_signboard --aihub_dir data/aihub_signboard
  python src/detect/prepare_dataset.py --source all
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
YOLO_DIR = DATA_DIR / "yolo_signs"

CLASS_NAMES = ["traffic_sign", "signboard"]

# GTSDB 43개 세부 클래스 → 모두 traffic_sign (class 0)
# 상위 카테고리: 0-12 위험, 13-31 규제, 32-42 지시
GTSDB_TO_YOLO_CLASS = 0  # 모두 traffic_sign


def setup_yolo_dirs():
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml():
    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

names:
  0: traffic_sign
  1: signboard

nc: 2
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Written {yaml_path}")


def convert_gtsdb(val_ratio=0.2):
    """GTSDB PPM 이미지 + gt.txt → YOLO 포맷."""
    gtsdb_dir = DATA_DIR / "GTSDB" / "FullIJCNN2013"
    gt_file = gtsdb_dir / "gt.txt"

    if not gt_file.exists():
        print(f"GTSDB gt.txt not found at {gt_file}")
        print("Run: python scripts/download_gtsdb.py")
        return 0

    annotations = {}
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 6:
                continue
            fname = parts[0]
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            annotations.setdefault(fname, []).append((x1, y1, x2, y2))

    ppm_files = sorted(gtsdb_dir.glob("*.ppm"))
    if not ppm_files:
        print("No PPM files found in GTSDB directory")
        return 0

    random.seed(42)
    random.shuffle(ppm_files)
    split_idx = int(len(ppm_files) * (1 - val_ratio))
    splits = {
        "train": ppm_files[:split_idx],
        "val": ppm_files[split_idx:],
    }

    total = 0
    for split, files in splits.items():
        for ppm_path in files:
            img = Image.open(ppm_path)
            w, h = img.size

            jpg_name = ppm_path.stem + ".jpg"
            img.save(YOLO_DIR / "images" / split / jpg_name, "JPEG", quality=95)

            bboxes = annotations.get(ppm_path.name, [])
            label_path = YOLO_DIR / "labels" / split / (ppm_path.stem + ".txt")
            with open(label_path, "w") as lf:
                for x1, y1, x2, y2 in bboxes:
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))
                    lf.write(f"{GTSDB_TO_YOLO_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            total += 1

    print(f"GTSDB: {total} images converted (train: {len(splits['train'])}, val: {len(splits['val'])})")
    return total


def convert_aihub_traffic(aihub_dir, val_ratio=0.15, max_images=5000):
    """AI Hub 신호등/도로표지판 JSON → YOLO 포맷.

    AI Hub JSON 구조 (예상):
      { "images": [...], "annotations": [{"bbox": [x,y,w,h], "category_id": ...}] }
    또는
      { "image": {"file_name": ...}, "annotation": [{"box": [x1,y1,x2,y2], ...}] }

    실제 구조는 다운로드 후 확인하여 이 함수를 수정해야 합니다.
    """
    aihub_path = Path(aihub_dir)
    if not aihub_path.exists():
        print(f"AI Hub traffic directory not found: {aihub_path}")
        print("AI Hub 데이터 다운로드 후 --aihub_dir 경로를 지정하세요.")
        return 0

    json_files = list(aihub_path.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {aihub_path}")
        return 0

    print(f"Found {len(json_files)} JSON annotation files")
    print("Sampling first file to detect format...")

    with open(json_files[0], "r", encoding="utf-8") as f:
        sample = json.load(f)
    print(f"  Keys: {list(sample.keys())}")

    # TODO: AI Hub 데이터 도착 후 실제 JSON 구조에 맞게 파싱 로직 구현
    # 아래는 일반적인 COCO-like 포맷 가정
    print("AI Hub traffic data conversion: TODO (데이터 도착 후 구현)")
    return 0


def convert_aihub_signboard(aihub_dir, val_ratio=0.15, max_images=5000):
    """AI Hub 야외 한글 간판 JSON → YOLO 포맷.

    AI Hub JSON 구조 (예상):
      각 이미지별 JSON: {"image": {...}, "annotations": [{"bbox": [x,y,w,h], "text": "..."}]}

    signboard는 class 1로 매핑.
    """
    aihub_path = Path(aihub_dir)
    if not aihub_path.exists():
        print(f"AI Hub signboard directory not found: {aihub_path}")
        print("AI Hub 데이터 다운로드 후 --aihub_dir 경로를 지정하세요.")
        return 0

    json_files = list(aihub_path.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {aihub_path}")
        return 0

    print(f"Found {len(json_files)} JSON annotation files")
    print("Sampling first file to detect format...")

    with open(json_files[0], "r", encoding="utf-8") as f:
        sample = json.load(f)
    print(f"  Keys: {list(sample.keys())}")

    # TODO: AI Hub 데이터 도착 후 실제 JSON 구조에 맞게 파싱 로직 구현
    print("AI Hub signboard data conversion: TODO (데이터 도착 후 구현)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="데이터셋을 YOLO 포맷으로 변환")
    parser.add_argument(
        "--source",
        choices=["gtsdb", "aihub_traffic", "aihub_signboard", "all"],
        default="gtsdb",
        help="변환할 데이터 소스",
    )
    parser.add_argument("--aihub_dir", type=str, default=None, help="AI Hub 데이터 디렉토리 경로")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="검증 데이터 비율")
    parser.add_argument("--max_images", type=int, default=5000, help="AI Hub에서 사용할 최대 이미지 수")
    args = parser.parse_args()

    setup_yolo_dirs()

    total = 0
    if args.source in ("gtsdb", "all"):
        total += convert_gtsdb(args.val_ratio)

    if args.source in ("aihub_traffic", "all"):
        aihub_dir = args.aihub_dir or str(DATA_DIR / "aihub_traffic")
        total += convert_aihub_traffic(aihub_dir, args.val_ratio, args.max_images)

    if args.source in ("aihub_signboard", "all"):
        aihub_dir = args.aihub_dir or str(DATA_DIR / "aihub_signboard")
        total += convert_aihub_signboard(aihub_dir, args.val_ratio, args.max_images)

    if total > 0:
        write_dataset_yaml()
        print(f"\nTotal: {total} images prepared in {YOLO_DIR}")
    else:
        print("\nNo images converted. Check data paths.")


if __name__ == "__main__":
    main()
