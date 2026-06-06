"""Phase 12 검출기 학습 — 채택 모델(YOLO26n, P1 결정)을 기존 한국 도로 데이터로 재학습.
데이터: data/yolo_signs_v2/dataset.yaml (2클래스: traffic_sign, traffic_light).
v3(edge_sign_v3_lights) 학습 관행 계승: imgsz 1280, close_mosaic, patience.

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_v4_detector.py --model yolo26n --epochs 40
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", help="P1에서 채택한 모델 (yolo26n)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=str(ROOT / "data" / "yolo_signs_v2" / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        close_mosaic=10,
        patience=20,
        workers=0,
        name="edge_sign_v4",
        project=str(ROOT / "runs" / "detect"),
    )
    best = ROOT / "runs" / "detect" / "edge_sign_v4" / "weights" / "best.pt"
    print(f"[train] done. best: {best}")


if __name__ == "__main__":
    main()
