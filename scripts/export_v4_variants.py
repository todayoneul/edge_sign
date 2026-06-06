"""Phase 12 검출기(YOLO26-n) 4정밀도 변형 생성.
  best.pt -> yolo_v4_signs_fp32.onnx        (Ultralytics export, NMS-free (1,300,6))
          -> yolo_v4_signs_fp16.onnx        (onnxconverter-common float16, keep_io_types)
          -> yolo_v4_signs_int8_static.onnx (QDQ static; YOLO26는 DFL-free → 기본 풀헤드 양자화)
          -> yolo_v4_signs_w4a16.onnx       (fake-quant, 붕괴 시연용)

YOLO26 출력은 (1,300,6)=[x1,y1,x2,y2,conf,cls] end-to-end (DECISION-yolo26-vs-yolo11.md 참조).

사용법:
  KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_v4_variants.py \
      --weights runs/detect/edge_sign_v4/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MS = ROOT / "model_space"
FP32 = MS / "yolo_v4_signs_fp32.onnx"
FP16 = MS / "yolo_v4_signs_fp16.onnx"
INT8 = MS / "yolo_v4_signs_int8_static.onnx"
W4A16 = MS / "yolo_v4_signs_w4a16.onnx"
CALIB_DIR = ROOT / "data" / "yolo_signs_v2" / "images" / "val"


def _mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def export_fp32(weights: str, imgsz: int) -> None:
    from ultralytics import YOLO

    p = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=14, dynamic=False, simplify=True)
    Path(p).replace(FP32)
    print(f"[fp32] {FP32.name} ({_mb(FP32):.1f}MB)")


def export_fp16() -> None:
    m = onnx.shape_inference.infer_shapes(onnx.load(str(FP32)))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    del m16.graph.value_info[:]
    onnx.save(m16, str(FP16))
    print(f"[fp16] {FP16.name} ({_mb(FP16):.1f}MB)")


def export_int8(quant_head: bool, n_calib: int = 150) -> None:
    from onnxruntime.quantization import (
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    from scripts.quantize_v3_detector import FlatYoloCalib  # 캘리브레이션 리더 재사용

    # YOLO26은 DFL 없음 → 기본 풀헤드 양자화(exclude=[]). 헤드 제외 비교가 필요하면 노드명으로 지정.
    exclude: list[str] = []
    prep = MS / "_prep_v4.onnx"
    quant_pre_process(
        input_model_path=str(FP32),
        output_model_path=str(prep),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )
    quantize_static(
        model_input=str(prep),
        model_output=str(INT8),
        calibration_data_reader=FlatYoloCalib(CALIB_DIR, n=n_calib),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=exclude,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        },
    )
    prep.unlink(missing_ok=True)
    print(f"[int8] {INT8.name} ({_mb(INT8):.1f}MB) head={'quant' if quant_head else 'excluded'}")


def export_w4a16(weights: str) -> None:
    # src/quant/quantize_yolo.py 의 fake-quant 경로 조합 (load → apply_w4a16_ptq → export).
    from src.quant.quantize_yolo import apply_w4a16_ptq, export_to_onnx, load_yolo_model

    yolo = load_yolo_model(weights)
    n = apply_w4a16_ptq(yolo.model)
    print(f"[w4a16] 양자화 레이어 {n}개")
    out = export_to_onnx(yolo, W4A16.name)  # MODEL_SPACE/<name> 으로 저장
    print(f"[w4a16] {Path(out).name} ({_mb(Path(out)):.1f}MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640, help="배포 추론 해상도(학습 1280과 별개)")
    ap.add_argument("--exclude_head", action="store_true", help="INT8에서 헤드 제외(비교용)")
    args = ap.parse_args()
    export_fp32(args.weights, args.imgsz)
    export_fp16()
    export_int8(quant_head=not args.exclude_head)
    export_w4a16(args.weights)
    print("[done] 4 variants in model_space/")


if __name__ == "__main__":
    main()
