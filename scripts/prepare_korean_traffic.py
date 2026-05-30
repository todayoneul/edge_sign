"""
[Phase 8] AI Hub 신호등-도로표지판 → 한국 도메인 검출/분류 데이터셋 (단일 패스)

기존 파이프라인의 한계 해결:
  - 검출기가 신호등을 traffic_sign(0)에 통합 → 신호등 색상 구분 불가
  - 분류기가 독일 GTSDB 43클래스 → 한국 표지판 오분류

한 번의 프레임 순회로 두 데이터셋을 동시 생성:
  1) 검출기 (data/yolo_signs_v2):  프레임 + YOLO 라벨, 0=traffic_sign 1=traffic_light
  2) 분류기 (data/roi_cls):        ROI 크롭 + 한국어 14클래스

클래스 (분류기):
  표지판: 속도제한30/40/50/60/70/80, 규제표지, 지시표지, 주의표지
  신호등: 신호등_빨강/초록/노랑/좌회전/기타

사용법:
  python scripts/prepare_korean_traffic.py            # 전체 생성
  python scripts/prepare_korean_traffic.py --max 2000 # split당 프레임 제한(빠른 테스트)
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "data" / "aihub_traffic"
YOLO_V2 = ROOT / "data" / "yolo_signs_v2"
ROI_DIR = ROOT / "data" / "roi_cls"
ROI_SIZE = 32

# 검출기 클래스 (신호등 분리)
DET_SIGN, DET_LIGHT = 0, 1

# 분류기 14클래스 (인덱스 고정)
CLS_NAMES = [
    "속도제한30", "속도제한40", "속도제한50", "속도제한60", "속도제한70", "속도제한80",  # 0-5
    "규제표지", "지시표지", "주의표지",                                                  # 6-8
    "신호등_빨강", "신호등_초록", "신호등_노랑", "신호등_좌회전", "신호등_기타",          # 9-13
]
CLS_IDX = {n: i for i, n in enumerate(CLS_NAMES)}
# 추론 시 검출 클래스로 후보 서브셋 제한용
SIGN_CLS_IDS  = list(range(0, 9))
LIGHT_CLS_IDS = list(range(9, 14))


def sign_class(ann) -> str | None:
    t = ann.get("type", "")
    txt = str(ann.get("text", "")).strip()
    if t == "restriction":
        if txt in ("30", "40", "50", "60", "70", "80"):
            return "속도제한" + txt
        return "규제표지"
    if t == "instruction":
        return "지시표지"
    if t == "caution":
        return "주의표지"
    return None  # type 불명은 제외


def light_class(ann) -> str:
    attr_list = ann.get("attribute") or [{}]   # 빈 리스트([])도 방어
    attr = attr_list[0] if attr_list else {}
    on = [k for k, v in attr.items() if v == "on"]
    if any("left_arrow" in o for o in on):
        return "신호등_좌회전"
    if "red" in on:
        return "신호등_빨강"
    if "yellow" in on:
        return "신호등_노랑"
    if "green" in on:
        return "신호등_초록"
    return "신호등_기타"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def setup_dirs():
    # ROI 폴더는 클래스 인덱스(ASCII)로 — OpenCV가 한글 경로 read/write 실패하므로
    for split in ("train", "val"):
        (YOLO_V2 / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_V2 / "labels" / split).mkdir(parents=True, exist_ok=True)
        for ci in range(NUM_CLASSES):
            (ROI_DIR / split / f"{ci:02d}").mkdir(parents=True, exist_ok=True)


NUM_CLASSES = len(CLS_NAMES)


def write_yaml():
    (YOLO_V2 / "dataset.yaml").write_text(
        f"path: {YOLO_V2.resolve()}\n"
        "train: images/train\nval: images/val\n\n"
        "nc: 2\nnames:\n  0: traffic_sign\n  1: traffic_light\n",
        encoding="utf-8",
    )
    # 분류기 클래스 인덱스 파일 (학습/추론 공유)
    (ROI_DIR / "classes.json").write_text(
        json.dumps({"names": CLS_NAMES,
                    "sign_ids": SIGN_CLS_IDS, "light_ids": LIGHT_CLS_IDS},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process(max_frames=None):
    setup_dirs()
    write_yaml()
    det_count = Counter()
    cls_count = Counter()
    n_frame = 0

    for split in ("train", "val"):
        img_root = SRC / split / "images"
        lbl_root = SRC / split / "labels"
        if not img_root.exists():
            print(f"[건너뜀] {img_root} 없음")
            continue

        jpgs = sorted(img_root.rglob("*.jpg"))
        if max_frames:
            jpgs = jpgs[:max_frames]

        kept = 0
        for jpg in jpgs:
            rel = jpg.relative_to(img_root)
            jf = lbl_root / rel.with_suffix(".json")
            if not jf.exists():
                continue
            try:
                d = json.loads(jf.read_bytes().decode("utf-8"))
            except Exception:
                continue

            imsize = d.get("image", {}).get("imsize", [])
            W = H = None
            if len(imsize) >= 2:
                W, H = imsize[0], imsize[1]

            anns = d.get("annotation", [])
            yolo_lines = []
            rois = []  # (cls_idx, box)
            for a in anns:
                c = a.get("class")
                box = a.get("box", [])
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = box[:4]
                if c == "traffic_sign":
                    det = DET_SIGN
                    sc = sign_class(a)
                    cls_idx = CLS_IDX.get(sc) if sc else None
                elif c == "traffic_light":
                    det = DET_LIGHT
                    cls_idx = CLS_IDX.get(light_class(a))
                else:
                    continue  # traffic_information 등 제외

                if W is None:
                    img_tmp = cv2.imread(str(jpg))
                    if img_tmp is None:
                        break
                    H, W = img_tmp.shape[:2]
                cx = clamp((x1 + x2) / 2 / W, 0, 1)
                cy = clamp((y1 + y2) / 2 / H, 0, 1)
                bw = clamp((x2 - x1) / W, 0, 1)
                bh = clamp((y2 - y1) / H, 0, 1)
                if bw <= 0 or bh <= 0:
                    continue
                yolo_lines.append(f"{det} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                det_count[det] += 1
                if cls_idx is not None:
                    rois.append((cls_idx, (int(x1), int(y1), int(x2), int(y2))))

            if not yolo_lines:
                continue

            seq = jpg.parent.name
            stem = f"{seq}__{jpg.stem}"
            # 검출기: 프레임 복사 + 라벨
            shutil.copy2(jpg, YOLO_V2 / "images" / split / f"{stem}.jpg")
            (YOLO_V2 / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(yolo_lines) + "\n", encoding="utf-8")

            # 분류기: ROI 크롭 저장
            if rois:
                img = cv2.imread(str(jpg))
                if img is not None:
                    for j, (ci, (bx1, by1, bx2, by2)) in enumerate(rois):
                        mx = max(2, int((bx2 - bx1) * 0.08))
                        my = max(2, int((by2 - by1) * 0.08))
                        cx1, cy1 = max(0, bx1 - mx), max(0, by1 - my)
                        cx2, cy2 = min(W, bx2 + mx), min(H, by2 + my)
                        crop = img[cy1:cy2, cx1:cx2]
                        if crop.size == 0:
                            continue
                        crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE),
                                          interpolation=cv2.INTER_AREA)
                        out = ROI_DIR / split / f"{ci:02d}" / f"{stem}_{j}.jpg"
                        cv2.imwrite(str(out), crop)  # ASCII 경로 → 정상 동작
                        cls_count[CLS_NAMES[ci]] += 1
            kept += 1
            n_frame += 1

        print(f"[{split}] 프레임 {kept:,}장 처리")

    print(f"\n총 프레임 {n_frame:,}")
    print(f"검출 박스: traffic_sign={det_count[0]:,}  traffic_light={det_count[1]:,}")
    print("분류기 ROI 클래스별:")
    for c in CLS_NAMES:
        print(f"  {c}: {cls_count[c]:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="split당 프레임 제한")
    args = ap.parse_args()
    process(args.max)
