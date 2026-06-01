"""v3 신호등 분리 검출기 → Static INT8 QDQ 양자화.

A/B 양자화 토글(서버 기본 v3 택소노미)에서 fp32 ⇄ int8을 같은 택소노미로 노출하려면
v3 검출기의 INT8 static 버전이 필요하다. 기존 dynamic INT8(ConvInteger)은 CPU EP
미지원이라 반드시 static(QDQ)으로 생성한다.

재활용: scripts/archive/quantize_onnx_real.py 의 static_quantize + YoloCalibReader.

사용법:
  python scripts/quantize_v3_detector.py            # v3_fp32 → v3_int8_static
  python scripts/quantize_v3_detector.py --bench     # 양자화 + 레이턴시 비교
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from onnxruntime.quantization import (
    QuantType,
    QuantFormat,
    CalibrationDataReader,
    quantize_static,
    quant_pre_process,
)
from scripts.archive.quantize_onnx_real import verify, compare_latency

# YOLOv8 Detect 헤드 prefix — 이 노드들은 양자화 제외(FP32 유지)해야 신뢰도 보존.
#   cv2=박스회귀, cv3=클래스점수, dfl. 헤드를 INT8화하면 conf가 임계값 아래로 붕괴.
HEAD_PREFIX = "/model.22"

MODEL_DIR = ROOT / "model_space"
# v3 학습 도메인(신호등 분리) 실프레임 — 더미 대신 실제 활성화 분포로 캘리브레이션.
CALIB_DIR = ROOT / "data" / "yolo_signs_v2" / "images" / "val"


def _preprocess(img: np.ndarray) -> np.ndarray:
    t = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.transpose(t.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]  # [1,3,640,640]


class FlatYoloCalib(CalibrationDataReader):
    """평평한 디렉토리(*.jpg)에서 YOLOv8s 캘리브레이션 프레임 로드."""

    def __init__(self, img_dir: Path, n: int = 100):
        self._dir, self._n = img_dir, n
        self._iter = iter(self._load())

    def _load(self):
        paths = sorted(self._dir.glob("*.jpg"))[: self._n]
        used = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            used += 1
            yield {"images": _preprocess(img)}
        print(f"  [Calib v3] {used} real frames from {self._dir.name}")

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._load())


def _real_frame() -> np.ndarray:
    """검증용 실프레임 1장 (없으면 랜덤)."""
    for p in sorted(CALIB_DIR.glob("*.jpg"))[:1]:
        img = cv2.imread(str(p))
        if img is not None:
            return _preprocess(img)
    return np.random.rand(1, 3, 640, 640).astype(np.float32)


def _head_nodes(onnx_path: Path) -> list[str]:
    """Detect 헤드(/model.22/*) 노드 이름 — 양자화 제외 대상."""
    g = onnx.load(str(onnx_path)).graph
    return [n.name for n in g.node if n.name.startswith(HEAD_PREFIX)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="양자화 후 FP32/INT8 레이턴시 비교")
    ap.add_argument("--n_calib", type=int, default=150, help="캘리브레이션 프레임 수")
    ap.add_argument(
        "--quant_head", action="store_true", help="검출 헤드까지 양자화(정확도 붕괴 위험, 비교용)"
    )
    args = ap.parse_args()

    src = MODEL_DIR / "yolov8s_signs_v3_fp32.onnx"
    dst = MODEL_DIR / "yolov8s_signs_v3_int8_static.onnx"
    if not src.exists():
        print(f"[ERROR] v3 FP32 검출기 없음: {src}")
        sys.exit(1)
    if not CALIB_DIR.exists():
        print(f"[WARN] 캘리브레이션 실프레임 디렉토리 없음: {CALIB_DIR} (랜덤 더미 사용)")

    exclude = [] if args.quant_head else _head_nodes(src)

    print(f"\nv3 검출기 Static INT8 QDQ 양자화")
    print(f"  {src.name} -> {dst.name}")
    print(f"  헤드 제외 노드: {len(exclude)}개 (FP32 유지)" if exclude else "  (헤드까지 양자화)")
    print("=" * 60)

    # Step 1: 전처리 (shape 추론 + BN-Conv 융합)
    prep = dst.parent / f"_prep_{src.stem}.onnx"
    quant_pre_process(
        input_model_path=str(src),
        output_model_path=str(prep),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )

    # Step 2: 정적 양자화 (헤드 노드 제외)
    quantize_static(
        model_input=str(prep),
        model_output=str(dst),
        calibration_data_reader=FlatYoloCalib(CALIB_DIR, n=args.n_calib),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,  # SiLU 이후 양수 분포 → UInt8
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
    orig_mb, q_mb = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"  {orig_mb:.2f} MB -> {q_mb:.2f} MB ({orig_mb / q_mb:.2f}x smaller)")

    real = _real_frame()  # 실프레임 기준 FP32↔INT8 출력 일치도
    verify(src, dst, real)
    if args.bench:
        print(f"  Latency:")
        compare_latency(src, dst, real, n=50)

    print(f"\n생성: {dst} ({q_mb:.2f} MB)")


if __name__ == "__main__":
    main()
