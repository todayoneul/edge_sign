"""
GTSDB / AI Hub 데이터 → YOLO 포맷 변환 스크립트

지원 소스:
  gtsdb          : GTSDB 900장 PPM + gt.txt
  aihub_traffic  : AIhub 신호등-도로표지판 (extract_frames.py 출력 기반)
  aihub_signboard: AIhub 030.야외 실제 촬영 한글 이미지 (이미 해제됨)
  all            : 위 세 소스 합산

YOLO 클래스:
  0: traffic_sign  (GTSDB 교통표지판 + AI Hub traffic_sign / traffic_light)
  1: signboard     (AI Hub 030 한글 간판)

어노테이션 포맷:
  GTSDB gt.txt      : x1;y1;x2;y2 (xyxy 절대 픽셀)
  신호등-도로표지판 JSON : {"annotation":[{"box":[x1,y1,x2,y2], "class":"traffic_sign"/"traffic_light"}],
                          "image":{"filename":"...", "imsize":[w,h]}}
  030.야외 한글 이미지 JSON: {"images":[{"width":w,"height":h,"file_name":"..."}],
                             "annotations":[{"bbox":[x,y,w,h], "text":"..."}]}

출력 구조:
  data/yolo_signs/
  ├── images/
  │   ├── train/  (모든 학습 이미지 - 플랫 구조)
  │   └── val/
  ├── labels/
  │   ├── train/  (YOLO .txt 파일)
  │   └── val/
  └── dataset.yaml

사용법:
  python src/detect/prepare_dataset.py --source gtsdb
  python src/detect/prepare_dataset.py --source aihub_traffic
  python src/detect/prepare_dataset.py --source aihub_signboard
  python src/detect/prepare_dataset.py --source all
  python src/detect/prepare_dataset.py --source all --max_images 10000
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT     = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
YOLO_DIR = DATA_DIR / "yolo_signs"

CLASS_NAMES = ["traffic_sign", "signboard"]  # class 0, 1

# GTSDB: 모든 클래스를 traffic_sign (0) 으로 통합
GTSDB_TO_YOLO_CLASS = 0

# AI Hub 신호등-도로표지판 클래스 매핑
AIHUB_TRAFFIC_CLASS_MAP = {
    "traffic_sign":  0,
    "traffic_light": 0,  # 신호등도 traffic_sign(0)으로 통합
}

# AI Hub 030 간판 클래스
AIHUB_SIGNBOARD_CLASS = 1


# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────

def setup_yolo_dirs():
    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml():
    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
  0: traffic_sign
  1: signboard
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  → {yaml_path}")


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """절대 픽셀 xyxy → YOLO 정규화 cx cy w h."""
    cx = clamp((x1 + x2) / 2.0 / W)
    cy = clamp((y1 + y2) / 2.0 / H)
    bw = clamp((x2 - x1) / W)
    bh = clamp((y2 - y1) / H)
    return cx, cy, bw, bh


def xywh_to_yolo(x, y, w, h, W, H):
    """절대 픽셀 xywh (COCO) → YOLO 정규화 cx cy w h."""
    cx = clamp((x + w / 2.0) / W)
    cy = clamp((y + h / 2.0) / H)
    bw = clamp(w / W)
    bh = clamp(h / H)
    return cx, cy, bw, bh


def write_yolo_label(label_path: Path, entries: list):
    """entries: [(class_id, cx, cy, bw, bh), ...]"""
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, cx, cy, bw, bh in entries:
            if bw > 0 and bh > 0:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ─────────────────────────────────────────────
# 1. GTSDB
# ─────────────────────────────────────────────

def convert_gtsdb(val_ratio=0.2):
    """GTSDB PPM + gt.txt → YOLO 포맷."""
    gtsdb_dir = DATA_DIR / "GTSDB" / "FullIJCNN2013"
    gt_file   = gtsdb_dir / "gt.txt"

    if not gt_file.exists():
        print(f"  ⚠️  GTSDB gt.txt 없음: {gt_file}")
        print("  실행: python scripts/download_gtsdb.py")
        return 0

    # gt.txt 파싱
    annotations: dict = {}
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
        print("  ⚠️  PPM 파일 없음")
        return 0

    random.seed(42)
    random.shuffle(ppm_files)
    split_idx = int(len(ppm_files) * (1 - val_ratio))
    splits = {"train": ppm_files[:split_idx], "val": ppm_files[split_idx:]}

    total = 0
    for split, files in splits.items():
        for ppm_path in files:
            img = Image.open(ppm_path)
            W, H = img.size

            jpg_name = ppm_path.stem + ".jpg"
            img.save(YOLO_DIR / "images" / split / jpg_name, "JPEG", quality=95)

            bboxes = annotations.get(ppm_path.name, [])
            entries = [
                (GTSDB_TO_YOLO_CLASS, *xyxy_to_yolo(x1, y1, x2, y2, W, H))
                for x1, y1, x2, y2 in bboxes
            ]
            write_yolo_label(
                YOLO_DIR / "labels" / split / (ppm_path.stem + ".txt"),
                entries,
            )
            total += 1

    print(f"  GTSDB: {total}장 변환 (train {len(splits['train'])}, val {len(splits['val'])})")
    return total


# ─────────────────────────────────────────────
# 2. AI Hub 신호등-도로표지판
#    (extract_frames.py 출력 기반)
# ─────────────────────────────────────────────

def convert_aihub_traffic(aihub_dir=None, max_images=None):
    """
    data/aihub_traffic/ (extract_frames.py 출력) → YOLO 포맷

    입력 구조:
      aihub_dir/
      ├── train/
      │   ├── images/{seq_name}/*.jpg
      │   └── labels/{seq_name}/*.json   ← AI Hub JSON 어노테이션
      └── val/  (동일 구조)

    JSON 포맷:
      {"annotation": [{"box": [x1,y1,x2,y2], "class": "traffic_sign"/"traffic_light"}],
       "image": {"filename": "...", "imsize": [w, h]}}
    """
    base = Path(aihub_dir) if aihub_dir else DATA_DIR / "aihub_traffic"

    if not base.exists():
        print(f"  ⚠️  AI Hub traffic 디렉토리 없음: {base}")
        print("  먼저 실행: python scripts/extract_frames.py")
        return 0

    total = 0
    for split in ("train", "val"):
        img_root = base / split / "images"
        lbl_root = base / split / "labels"

        if not img_root.exists():
            print(f"  ⚠️  {split} 디렉토리 없음: {img_root}")
            continue

        # 모든 JPG 수집 (per-sequence 서브디렉토리 포함)
        jpg_files = sorted(img_root.rglob("*.jpg"))
        if max_images and split == "train":
            random.seed(42)
            random.shuffle(jpg_files)
            jpg_files = jpg_files[:max_images]

        count = 0
        for jpg_path in jpg_files:
            # 대응 JSON 경로: images/{seq}/frame.jpg → labels/{seq}/frame.json
            rel = jpg_path.relative_to(img_root)
            json_path = lbl_root / rel.with_suffix(".json")

            if not json_path.exists():
                continue  # 라벨 없는 프레임 건너뜀

            # JSON 파싱
            try:
                data = json.loads(json_path.read_bytes().decode("utf-8"))
            except Exception:
                continue

            img_info = data.get("image", {})
            imsize   = img_info.get("imsize", [])
            if len(imsize) < 2:
                # imsize 없는 경우 이미지에서 직접 읽기
                try:
                    with Image.open(jpg_path) as im:
                        W, H = im.size
                except Exception:
                    continue
            else:
                W, H = imsize[0], imsize[1]

            annotations = data.get("annotation", [])
            entries = []
            for ann in annotations:
                cls_name = ann.get("class", "traffic_sign")
                cls_id   = AIHUB_TRAFFIC_CLASS_MAP.get(cls_name, 0)
                box      = ann.get("box", [])
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                entries.append((cls_id, *xyxy_to_yolo(x1, y1, x2, y2, W, H)))

            # YOLO 디렉토리에 플랫 구조로 저장 (시퀀스명__프레임명)
            seq_name    = jpg_path.parent.name
            unique_stem = f"{seq_name}__{jpg_path.stem}"

            out_jpg = YOLO_DIR / "images" / split / (unique_stem + ".jpg")
            out_txt = YOLO_DIR / "labels" / split / (unique_stem + ".txt")

            shutil.copy2(jpg_path, out_jpg)
            write_yolo_label(out_txt, entries)
            count += 1

        print(f"  AI Hub traffic [{split}]: {count:,}장")
        total += count

    return total


# ─────────────────────────────────────────────
# 3. AI Hub 030.야외 실제 촬영 한글 이미지
#    (이미 압축 해제된 상태)
# ─────────────────────────────────────────────

def convert_aihub_signboard(aihub_dir=None, max_images=None):
    """
    AIhub/030.야외 실제 촬영 한글 이미지/01.데이터/ → YOLO 포맷

    입력 구조:
      aihub_dir/01.데이터/
      ├── 1.Training/
      │   ├── 원천데이터_230216_add/1.간판/{subclass}/*.jpg
      │   └── 라벨링데이터_230216_add/1.간판/{subclass}/*.json
      └── 2.Validation/  (동일 구조)

    JSON 포맷 (이미지당 1 JSON):
      {"images": [{"width": W, "height": H, "file_name": "..."}],
       "annotations": [{"text": "텍스트", "bbox": [x, y, w, h]}],
       "metadata": {"metaclass": "실외간판", "subclass": "가로형간판", ...}}

    bbox 포맷: [x_topleft, y_topleft, width, height] (COCO 스타일)
    class 매핑: 모두 signboard (1)
    """
    base = Path(aihub_dir) if aihub_dir else (
        ROOT / "AIhub" / "030.야외 실제 촬영 한글 이미지"
    )
    data_root = base / "01.데이터"

    if not data_root.exists():
        print(f"  ⚠️  야외 한글 이미지 디렉토리 없음: {data_root}")
        return 0

    split_map = {
        "1.Training":  "train",
        "2.Validation": "val",
    }

    total = 0
    for src_split, yolo_split in split_map.items():
        src_dir = data_root / src_split
        if not src_dir.exists():
            continue

        # 간판만 사용 (책표지 제외)
        img_root = src_dir / "원천데이터_230216_add" / "1.간판"
        lbl_root = src_dir / "라벨링데이터_230216_add" / "1.간판"

        if not img_root.exists():
            print(f"  ⚠️  간판 디렉토리 없음: {img_root}")
            continue

        # 모든 JPG 수집
        jpg_files = sorted(img_root.rglob("*.jpg"))
        if max_images and yolo_split == "train":
            random.seed(42)
            random.shuffle(jpg_files)
            jpg_files = jpg_files[:max_images]

        count = 0
        for jpg_path in jpg_files:
            # 대응 JSON: 원천데이터 → 라벨링데이터 경로 치환
            rel = jpg_path.relative_to(img_root)
            json_path = (lbl_root / rel).with_suffix(".json")

            if not json_path.exists():
                continue

            # JSON 파싱
            try:
                data = json.loads(json_path.read_bytes().decode("utf-8"))
            except Exception:
                continue

            images_info = data.get("images", [])
            if not images_info:
                continue
            img_info = images_info[0]
            W = img_info.get("width", 0)
            H = img_info.get("height", 0)
            if W <= 0 or H <= 0:
                try:
                    with Image.open(jpg_path) as im:
                        W, H = im.size
                except Exception:
                    continue

            annotations = data.get("annotations", [])
            entries = []
            for ann in annotations:
                bbox = ann.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                # "xxx" 텍스트는 가림막(occluded) → 학습 데이터에 포함
                entries.append(
                    (AIHUB_SIGNBOARD_CLASS, *xywh_to_yolo(x, y, w, h, W, H))
                )

            # YOLO 디렉토리에 저장 (서브클래스명__파일명 형식)
            subclass    = jpg_path.parent.name
            unique_stem = f"sign_{subclass}__{jpg_path.stem}"

            out_jpg = YOLO_DIR / "images" / yolo_split / (unique_stem + ".jpg")
            out_txt = YOLO_DIR / "labels" / yolo_split / (unique_stem + ".txt")

            shutil.copy2(jpg_path, out_jpg)
            write_yolo_label(out_txt, entries)
            count += 1

        print(f"  AI Hub signboard [{yolo_split}]: {count:,}장")
        total += count

    return total


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="데이터셋 → YOLO 포맷 변환")
    parser.add_argument(
        "--source",
        choices=["gtsdb", "aihub_traffic", "aihub_signboard", "all"],
        default="gtsdb",
        help="변환할 데이터 소스",
    )
    parser.add_argument(
        "--aihub_dir",
        type=str,
        default=None,
        help="AI Hub 데이터 디렉토리 경로 (기본값 자동 설정)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="GTSDB 검증 비율 (기본 0.2)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="각 AI Hub 소스에서 사용할 최대 학습 이미지 수 (기본 제한 없음)",
    )
    args = parser.parse_args()

    setup_yolo_dirs()
    total = 0

    print(f"=== YOLO 포맷 변환 → {YOLO_DIR} ===\n")

    if args.source in ("gtsdb", "all"):
        print("[GTSDB]")
        total += convert_gtsdb(args.val_ratio)

    if args.source in ("aihub_traffic", "all"):
        print("\n[AI Hub 신호등-도로표지판]")
        aihub_dir = args.aihub_dir or str(DATA_DIR / "aihub_traffic")
        total += convert_aihub_traffic(aihub_dir, args.max_images)

    if args.source in ("aihub_signboard", "all"):
        print("\n[AI Hub 030.야외 한글 간판]")
        aihub_dir = args.aihub_dir or None  # 자동 경로 사용
        total += convert_aihub_signboard(aihub_dir, args.max_images)

    if total > 0:
        print(f"\n[dataset.yaml 생성]")
        write_dataset_yaml()
        print(f"\n총 {total:,}장 변환 완료 → {YOLO_DIR}")
    else:
        print("\n변환된 이미지 없음. 데이터 경로를 확인하세요.")


if __name__ == "__main__":
    main()
