"""
AIhub 신호등-도로표지판 인지 영상(수도권) TAR 아카이브 처리 스크립트

원천 데이터: AIhub/신호등-도로표지판 인지 영상(수도권)/Validation/
  - [원천]*.tar: 이미 추출된 JPG 프레임 (비디오 파일이 아님!)
  - [라벨]*.tar: JSON 어노테이션 (프레임당 1개)

TAR 1개 = 촬영 시퀀스 1개 (9개 시퀀스 총 ~110,900 프레임)
  c_validation_1280_720_daylight_1,2,3 : 1280×720 해상도, 주간
  c_validation_1280_720_night_1        : 1280×720, 야간
  c_validation_1920_1200_daylight_1    : 1920×1200, 주간
  c_validation_1920_1200_night_1       : 1920×1200, 야간
  d_validation_1920_1080_daylight_1,2  : 1920×1080, 주간
  d_validation_1920_1080_night_1       : 1920×1080, 야간

처리 과정:
  1. [원천]*.tar + [라벨]*.tar 쌍 탐색
  2. 시퀀스를 크기 순으로 train / val / test 분할
     ★ 시퀀스 단위 분할 = 인접 프레임 리크 방지
  3. 각 시퀀스에서 매 N번째 프레임 서브샘플링
     (기본 N=6: 30fps 원본 → 5fps 시뮬레이션)
  4. 서브샘플된 JPG + 대응 JSON 을 출력 경로에 구조화

출력 구조:
  data/aihub_traffic/
  ├── train/
  │   ├── images/{seq_name}/*.jpg
  │   └── labels/{seq_name}/*.json   ← AI Hub 원본 JSON
  ├── val/
  │   ├── images/{seq_name}/
  │   └── labels/{seq_name}/
  └── test/                          ← ByteTrack 추적 평가용 연속 시퀀스
      ├── images/{seq_name}/
      └── labels/{seq_name}/

JSON 어노테이션 포맷 (신호등-도로표지판):
  {
    "annotation": [
      {
        "box": [x1, y1, x2, y2],      ← 절대 픽셀 좌표 (xyxy)
        "class": "traffic_sign",       ← 또는 "traffic_light"
        "shape": "circle"/"rectangle",
        "color": "blue"/"white"/...
      }
    ],
    "image": {
      "filename": "s01776609.jpg",
      "imsize": [width, height]
    }
  }

사용법:
  # 드라이런 (계획만 출력)
  python scripts/extract_frames.py --dry_run

  # 실제 추출
  python scripts/extract_frames.py \\
    --input "AIhub/신호등-도로표지판 인지 영상(수도권)/Validation" \\
    --output data/aihub_traffic \\
    --sample_rate 6

  # 전체 프레임 (서브샘플 없이)
  python scripts/extract_frames.py --sample_rate 1
"""

import argparse
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AIhub 신호등-도로표지판 TAR 해제 + 시퀀스 분할 + 서브샘플링"
    )
    parser.add_argument(
        "--input",
        default="AIhub/신호등-도로표지판 인지 영상(수도권)/Validation",
        help="TAR 파일이 있는 입력 디렉토리",
    )
    parser.add_argument(
        "--output",
        default="data/aihub_traffic",
        help="출력 디렉토리 (기본: data/aihub_traffic)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=6,
        help="프레임 서브샘플 비율 N: 매 N번째 프레임 추출 (기본 6 → 30fps→5fps 시뮬레이션)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.67,
        help="train 시퀀스 비율 (기본 0.67 → 9개 중 6개)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.11,
        help="val 시퀀스 비율 (기본 0.11 → 9개 중 1개)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="실제 추출 없이 분할 계획만 출력",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# 1. 시퀀스 탐색
# ──────────────────────────────────────────────

def find_sequence_pairs(input_dir: Path) -> dict:
    """
    [원천]*.tar 와 [라벨]*.tar 쌍을 탐색하여 시퀀스 딕셔너리 반환.
    Returns: {seq_name: {"source": Path, "label": Path, "size_gb": float}}
    """
    sequences = {}

    for tar_file in sorted(input_dir.glob("*.tar")):
        name = tar_file.name
        if not name.startswith("[원천]"):
            continue

        # "[원천]c_validation_1280_720_daylight_1.tar" → "c_validation_1280_720_daylight_1"
        seq_name = tar_file.stem[len("[원천]"):]

        label_tar = input_dir / f"[라벨]{seq_name}.tar"
        if not label_tar.exists():
            print(f"  [WARN] 라벨 TAR 없음, 건너뜀: {name}")
            continue

        sequences[seq_name] = {
            "source": tar_file,
            "label": label_tar,
            "size_gb": tar_file.stat().st_size / 1e9,
        }

    return sequences


# ──────────────────────────────────────────────
# 2. 시퀀스 분할 계획
# ──────────────────────────────────────────────

def plan_split(sequences: dict, train_ratio: float, val_ratio: float) -> dict:
    """
    시퀀스를 train / val / test 로 배정.
    크기 내림차순 정렬 후:
      - 큰 시퀀스(주간 + 고해상도) → train
      - 중간 → val
      - 작은 시퀀스(야간 등) → test (ByteTrack 평가용)
    """
    n = len(sequences)
    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))
    n_test  = n - n_train - n_val

    # 크기 내림차순으로 정렬
    by_size = sorted(sequences.keys(), key=lambda s: sequences[s]["size_gb"], reverse=True)

    split_map = {}
    for i, seq in enumerate(by_size):
        if i < n_train:
            split_map[seq] = "train"
        elif i < n_train + n_val:
            split_map[seq] = "val"
        else:
            split_map[seq] = "test"

    return split_map


# ──────────────────────────────────────────────
# 3. 단일 시퀀스 추출
# ──────────────────────────────────────────────

def extract_sequence(
    seq_name: str,
    source_tar: Path,
    label_tar: Path,
    output_dir: Path,
    split: str,
    sample_rate: int,
) -> int:
    """
    단일 시퀀스 TAR 해제 + 서브샘플링.
    Returns: 추출된 프레임 수.
    """
    img_out = output_dir / split / "images" / seq_name
    lbl_out = output_dir / split / "labels" / seq_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # ── 원천 TAR: JPG 목록 정렬 후 매 N번째 선택 ──
    with tarfile.open(source_tar, "r") as tf:
        all_jpgs = sorted(
            [m for m in tf.getmembers() if m.name.lower().endswith(".jpg")],
            key=lambda m: m.name,
        )
        selected = all_jpgs[::sample_rate]

        selected_stems: set = set()
        for member in selected:
            fname = Path(member.name).name          # 파일명만 (경로 제거)
            out_path = img_out / fname
            if not out_path.exists():               # 이미 추출된 경우 건너뜀
                f = tf.extractfile(member)
                if f:
                    out_path.write_bytes(f.read())
            selected_stems.add(Path(fname).stem)

    # ── 라벨 TAR: 선택된 프레임과 매칭되는 JSON만 추출 ──
    with tarfile.open(label_tar, "r") as tf:
        for member in tf.getmembers():
            if not member.name.lower().endswith(".json"):
                continue
            stem = Path(member.name).stem
            if stem in selected_stems:
                fname = Path(member.name).name
                out_path = lbl_out / fname
                if not out_path.exists():
                    f = tf.extractfile(member)
                    if f:
                        out_path.write_bytes(f.read())

    print(f"  [OK]  [{split:5}] {seq_name}: {len(selected_stems):,} frames")
    return len(selected_stems)


# ──────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] 입력 디렉토리 없음: {input_dir}")
        print("  TAR 파일 경로를 --input 으로 지정하세요.")
        return

    print(f"[IN ] 입력: {input_dir}")
    print(f"[OUT] 출력: {output_dir}")
    print(f"[N  ] 서브샘플 비율: 1/{args.sample_rate} "
          f"(=> {30 // args.sample_rate}fps 시뮬레이션)")
    print()

    # 시퀀스 쌍 탐색
    sequences = find_sequence_pairs(input_dir)
    if not sequences:
        print("[ERROR] [원천]*.tar + [라벨]*.tar 쌍을 찾을 수 없습니다.")
        return

    print(f"[OK ] 발견된 시퀀스: {len(sequences)}개")
    for name, info in sorted(sequences.items(), key=lambda x: x[1]["size_gb"], reverse=True):
        print(f"  {name:50s}  {info['size_gb']:.2f} GB")
    print()

    # 분할 계획
    split_map = plan_split(sequences, args.train_ratio, args.val_ratio)
    print("[PLAN] 시퀀스 분할 계획 (시퀀스 단위 = 리크 없음):")
    for split in ["train", "val", "test"]:
        seqs = [n for n, s in split_map.items() if s == split]
        total_gb = sum(sequences[n]["size_gb"] for n in seqs)
        print(f"  {split:5}: {len(seqs)}개 시퀀스 ({total_gb:.1f} GB)")
        for s in sorted(seqs):
            print(f"    - {s}")
    print()

    if args.dry_run:
        print("[DRY RUN] 실제 추출 없이 종료합니다.")
        print("실제 추출 시: --dry_run 옵션을 제거하고 다시 실행하세요.")
        return

    # 추출 실행
    print("[START] TAR 해제 시작 (시간이 소요됩니다)...")
    total = 0
    for seq_name in sorted(split_map.keys()):
        split = split_map[seq_name]
        count = extract_sequence(
            seq_name=seq_name,
            source_tar=sequences[seq_name]["source"],
            label_tar=sequences[seq_name]["label"],
            output_dir=output_dir,
            split=split,
            sample_rate=args.sample_rate,
        )
        total += count

    print(f"\n[DONE] 추출 완료!")
    print(f"   총 프레임: {total:,}개")
    print(f"   출력 위치: {output_dir}/")
    print()
    print("[NEXT] 다음 단계: YOLO 포맷 변환")
    print("   python src/detect/prepare_dataset.py --source aihub_traffic")


if __name__ == "__main__":
    main()
