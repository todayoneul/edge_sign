"""
YOLOv8n → ONNX 내보내기 + 선택적 INT8 양자화.

기존 export_onnx.py 패턴을 따르되, Ultralytics 내장 export 기능을 활용.

사용법:
  # FP32 ONNX 내보내기
  python src/detect/export_yolo_onnx.py --weights runs/detect/edge_sign_v2/weights/best.pt

  # FP16 ONNX
  python src/detect/export_yolo_onnx.py --weights best.pt --half

  # INT8 동적 양자화 (ONNX Runtime)
  python src/detect/export_yolo_onnx.py --weights best.pt --quantize int8

출력:
  model_space/yolov8n_signs_fp32.onnx
  model_space/yolov8n_signs_fp16.onnx
  model_space/yolov8n_signs_int8.onnx
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL_SPACE = ROOT / "model_space"


def export_onnx(weights, half=False, simplify=True):
    from ultralytics import YOLO

    MODEL_SPACE.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    precision = "fp16" if half else "fp32"

    result = model.export(
        format="onnx",
        imgsz=640,
        half=half,
        simplify=simplify,
        opset=14,
        dynamic=False,
    )

    src_path = Path(result)
    dst_path = MODEL_SPACE / f"yolov8n_signs_{precision}.onnx"
    if src_path.exists():
        import shutil
        shutil.copy2(src_path, dst_path)
        size_mb = dst_path.stat().st_size / (1024 * 1024)
        print(f"\nExported: {dst_path} ({size_mb:.2f} MB)")
    return dst_path


def quantize_int8(onnx_path):
    """ONNX Runtime 동적 INT8 양자화 (src/quantize_int8.py 패턴)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    output_path = MODEL_SPACE / "yolov8n_signs_int8.onnx"

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    orig_mb = onnx_path.stat().st_size / (1024 * 1024)
    ratio = orig_mb / size_mb
    print(f"\nINT8 quantized: {output_path}")
    print(f"  Original: {orig_mb:.2f} MB → INT8: {size_mb:.2f} MB ({ratio:.1f}x compression)")
    return output_path


def verify_onnx(onnx_path):
    """ONNX 모델 추론 검증."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("Skipping verification (onnxruntime not available)")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()

    print(f"\nModel verification: {onnx_path.name}")
    print(f"  Input:  {input_info.name} {input_info.shape} {input_info.type}")
    for out in output_info:
        print(f"  Output: {out.name} {out.shape} {out.type}")

    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {input_info.name: dummy})
    print(f"  Inference OK: output shape = {outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n ONNX 내보내기")
    parser.add_argument("--weights", type=str, required=True, help="학습된 모델 가중치 (.pt)")
    parser.add_argument("--half", action="store_true", help="FP16으로 내보내기")
    parser.add_argument("--quantize", choices=["none", "int8"], default="none", help="양자화 방법")
    parser.add_argument("--verify", action="store_true", default=True, help="내보내기 후 검증")
    args = parser.parse_args()

    onnx_path = export_onnx(args.weights, half=args.half)

    if args.quantize == "int8":
        onnx_path = quantize_int8(onnx_path)

    if args.verify:
        verify_onnx(onnx_path)


if __name__ == "__main__":
    main()
