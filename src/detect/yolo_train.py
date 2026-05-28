"""
YOLOv8-Nano 학습/평가 스크립트.

Edge-Sign v2 검출기: 교통표지판(traffic_sign) + 간판(signboard) 2-클래스 검출.

사용법:
  # 학습
  python src/detect/yolo_train.py --mode train --epochs 100

  # 평가
  python src/detect/yolo_train.py --mode val

  # 추론 테스트 (이미지/영상)
  python src/detect/yolo_train.py --mode predict --source path/to/image_or_video

  # 모델 정보 확인
  python src/detect/yolo_train.py --mode info
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
YOLO_DATASET = DATA_DIR / "yolo_signs" / "dataset.yaml"
RUNS_DIR = ROOT / "runs" / "detect"


def check_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("Ultralytics not installed. Run:")
        print("  pip install ultralytics")
        sys.exit(1)


def train(args):
    YOLO = check_ultralytics()

    if not YOLO_DATASET.exists():
        print(f"Dataset YAML not found: {YOLO_DATASET}")
        print("Run: python src/detect/prepare_dataset.py --source gtsdb")
        sys.exit(1)

    model = YOLO(args.model)
    print(f"  모델: {args.model}  imgsz: {args.imgsz}  batch: {args.batch_size}")

    results = model.train(
        data=str(YOLO_DATASET),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.run_name,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # 경량화 목적 하이퍼파라미터
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        # 소형 표지판 검출 개선
        copy_paste=args.copy_paste,   # 표지판 합성 증강 (기본 0.5)
        fliplr=args.fliplr,           # 한글/표지판 좌우반전 비활성 (기본 0.0)
        multi_scale=False,            # imgsz 고정 (True 시 VRAM 초과 위험)
    )

    print(f"\nTraining complete. Results saved to: {RUNS_DIR / args.run_name}")
    return results


def validate(args):
    YOLO = check_ultralytics()

    model_path = args.weights or str(RUNS_DIR / args.run_name / "weights" / "best.pt")
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train first: python src/detect/yolo_train.py --mode train")
        sys.exit(1)

    model = YOLO(model_path)
    results = model.val(
        data=str(YOLO_DATASET),
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        plots=True,
        verbose=True,
    )

    print("\n=== Validation Results ===")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:     {results.box.mp:.4f}")
    print(f"  Recall:        {results.box.mr:.4f}")

    return results


def predict(args):
    YOLO = check_ultralytics()

    model_path = args.weights or str(RUNS_DIR / args.run_name / "weights" / "best.pt")
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    if not args.source:
        print("--source required for predict mode")
        sys.exit(1)

    model = YOLO(model_path)
    results = model.predict(
        source=args.source,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=args.device,
        save=True,
        project=str(RUNS_DIR),
        name=f"{args.run_name}_predict",
        verbose=True,
    )

    for r in results:
        print(f"  {r.path}: {len(r.boxes)} detections")

    return results


def info(args):
    YOLO = check_ultralytics()
    model = YOLO(args.model)
    model.info(verbose=True)

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")

    import torch
    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model.model,
        dummy,
        "/dev/null" if sys.platform != "win32" else "NUL",
        opset_version=14,
        do_constant_folding=True,
    )
    print("  ONNX export: compatible (opset 14)")


def main():
    parser = argparse.ArgumentParser(description="Edge-Sign YOLOv8n 학습/평가")
    parser.add_argument("--mode", choices=["train", "val", "predict", "info"], default="train")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=8, help="배치 크기")
    parser.add_argument("--imgsz", type=int, default=1280, help="입력 이미지 크기")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLO 모델 (yolov8n.pt / yolov8s.pt 등)")
    parser.add_argument("--copy_paste", type=float, default=0.5, help="Copy-Paste 증강 확률 (소형 객체 검출 개선)")
    parser.add_argument("--fliplr", type=float, default=0.0, help="좌우반전 확률 (한글/표지판은 0.0 권장)")
    parser.add_argument("--device", type=str, default="0", help="GPU 디바이스 (0, cpu 등)")
    parser.add_argument("--run_name", type=str, default="edge_sign_v2", help="실험 이름")
    parser.add_argument("--weights", type=str, default=None, help="모델 가중치 경로 (val/predict)")
    parser.add_argument("--source", type=str, default=None, help="추론 입력 (이미지/영상 경로)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "val":
        validate(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "info":
        info(args)


if __name__ == "__main__":
    main()
