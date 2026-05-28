"""
E1~E5 양자화 실험 자동 실행 스크립트.

각 실험별로:
  1. 양자화 적용 → ONNX 내보내기
  2. ultralytics val로 mAP 측정
  3. 결과 출력 (EXPERIMENTS.md에 수동으로 기입)

사용법:
  python src/quant/run_experiments.py               # E1~E5 전체
  python src/quant/run_experiments.py --exp E1      # 특정 실험만
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "edge_sign_v2_e0_full3" / "weights" / "best.pt"
WEIGHTS         = DEFAULT_WEIGHTS  # 호환성을 위해 모듈 전역 유지 — main()에서 override
MODEL_SPACE     = ROOT / "model_space"
YOLO_YAML       = ROOT / "data" / "yolo_signs" / "dataset.yaml"


# ────────────────────────────────────────────
# ONNX val (ultralytics)
# ────────────────────────────────────────────

def val_onnx(onnx_path: Path, batch: int = 16, device: str = "cpu") -> dict:
    """ONNX 모델 평가 — ONNX Runtime은 CPU 실행 (GPU DLL 의존성 없음)."""
    from ultralytics import YOLO

    model = YOLO(str(onnx_path))
    results = model.val(
        data=str(YOLO_YAML),
        imgsz=640,
        batch=batch,
        device="cpu",   # ONNX Runtime GPU DLL 없으므로 항상 CPU
        verbose=False,
        plots=False,
    )
    return {
        "map50":      round(results.box.map50, 4),
        "map":        round(results.box.map,   4),
        "precision":  round(results.box.mp,    4),
        "recall":     round(results.box.mr,    4),
    }


# ────────────────────────────────────────────
# 개별 실험
# ────────────────────────────────────────────

def run_e1(device="0"):
    """E1: W8A8 YOLOv8s (검출기만)."""
    from src.quant.quantize_yolo import run_w8a8
    print("\n" + "="*50)
    print("E1: W8A8 PTQ 검출기")
    print("="*50)
    onnx_path = run_w8a8(WEIGHTS)
    metrics = val_onnx(onnx_path, device=device)
    return "E1", onnx_path, metrics


def run_e4(device="0"):
    """E4: W4A16 전체 (검출기 기준)."""
    from src.quant.quantize_yolo import run_w4a16
    print("\n" + "="*50)
    print("E4: W4A16 PTQ 검출기")
    print("="*50)
    onnx_path = run_w4a16(WEIGHTS)
    metrics = val_onnx(onnx_path, device=device)
    return "E4", onnx_path, metrics


def run_e5(device="0", calib_batches=10):
    """E5: SmoothQuant 전체 (검출기 기준)."""
    from src.quant.quantize_yolo import run_smoothquant
    print("\n" + "="*50)
    print("E5: SmoothQuant + W8A8 검출기")
    print("="*50)
    onnx_path = run_smoothquant(WEIGHTS, calib_batches=calib_batches)
    metrics = val_onnx(onnx_path, device=device)
    return "E5", onnx_path, metrics


# ────────────────────────────────────────────
# 결과 출력
# ────────────────────────────────────────────

def print_results(results: list):
    print("\n" + "="*60)
    print("검출 실험 결과 요약")
    print("="*60)
    print(f"{'ID':<6} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'P':<8} {'R':<8} {'크기(MB)':<10}")
    print("-"*60)

    # E0 기준선
    e0_onnx = MODEL_SPACE / "yolov8s_signs_fp32.onnx"
    e0_size = e0_onnx.stat().st_size / 1024 / 1024 if e0_onnx.exists() else 42.67
    print(f"{'E0':<6} {'0.6275':<10} {'0.4371':<14} {'0.722':<8} {'0.543':<8} {e0_size:<10.2f}  ← FP32 기준선")

    for exp_id, onnx_path, m in results:
        size_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"{exp_id:<6} {m['map50']:<10} {m['map']:<14} {m['precision']:<8} {m['recall']:<8} {size_mb:<10.2f}")

    print("\n[민감도 분석]")
    if results:
        e0_map = 0.6275
        for exp_id, _, m in results:
            delta = m['map50'] - e0_map
            pct = delta / e0_map * 100
            print(f"  {exp_id}: mAP50 {e0_map:.4f} → {m['map50']:.4f}  ({delta:+.4f}, {pct:+.1f}%)")


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

EXPERIMENT_MAP = {
    "E1": run_e1,
    "E4": run_e4,
    "E5": run_e5,
}

def main():
    parser = argparse.ArgumentParser(description="양자화 실험 E1/E4/E5 실행")
    parser.add_argument("--exp", choices=list(EXPERIMENT_MAP.keys()) + ["all"],
                        default="all", help="실행할 실험 ID")
    parser.add_argument("--device", type=str, default="0", help="GPU 디바이스")
    parser.add_argument("--calib_batches", type=int, default=10,
                        help="SmoothQuant 캘리브레이션 배치 수")
    parser.add_argument("--weights", type=str, default=None,
                        help="검출기 가중치 (.pt) — 미지정 시 기본 v1 경로 사용")
    args = parser.parse_args()

    # 가중치 전역 override
    if args.weights:
        global WEIGHTS
        WEIGHTS = Path(args.weights)
        print(f"[INFO] 가중치 override: {WEIGHTS}")

    targets = list(EXPERIMENT_MAP.keys()) if args.exp == "all" else [args.exp]
    results = []

    for exp_id in targets:
        fn = EXPERIMENT_MAP[exp_id]
        if exp_id == "E5":
            result = fn(device=args.device, calib_batches=args.calib_batches)
        else:
            result = fn(device=args.device)
        results.append(result)

    print_results(results)
    print("\n위 결과를 docs/EXPERIMENTS.md 검출 결과 표에 기입하세요.")


if __name__ == "__main__":
    main()
