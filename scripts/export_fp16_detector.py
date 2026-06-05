"""v3 검출기 FP32 ONNX → FP16 ONNX 변환 (브라우저 WebGPU 배포용).

배경(타당성 스파이크 실측): INT8 검출기는 브라우저 WebGPU에서 실행 불가
(`int32 DequantizeLinear` 미지원), WASM에선 ~2 FPS로 실시간 불가. 반면
FP32/WebGPU는 ~62 FPS. WebGPU는 fp16을 잘 지원하므로, fp16은 "속도 유지 +
크기 절반(43→~21.5MB)"으로 브라우저 엣지의 현실적 정답이다.

keep_io_types=True: 입력/출력은 float32로 유지(Cast 노드 삽입)하여 클라이언트
전처리(float32 NCHW)·후처리를 그대로 재사용한다. 내부 가중치/연산만 fp16.

사용법:
  python scripts/export_fp16_detector.py            # fp32 → fp16 변환 + 검증
  python scripts/export_fp16_detector.py --bench     # + CPU 레이턴시/출력차 비교
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "model_space" / "yolov8s_signs_v3_fp32.onnx"
DST = ROOT / "model_space" / "yolov8s_signs_v3_fp16.onnx"


def _mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def convert() -> None:
    if not SRC.exists():
        raise SystemExit(f"원본 없음: {SRC}")
    print(f"로드: {SRC.name} ({_mb(SRC):.1f}MB)")
    model = onnx.load(str(SRC))

    # 변환 전 명시적 shape inference — keep_io_types가 출력 cast를 올바른 타입으로
    # 배치하려면 그래프에 추론된 타입 정보가 있어야 한다(YOLO Resize 출력 cast 불일치 회피).
    model = onnx.shape_inference.infer_shapes(model)

    # keep_io_types: I/O는 float32 유지 → 클라 전처리(float32) 무변경. 내부만 fp16.
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    # 변환이 남긴 중간 텐서 타입 주석(value_info)이 삽입된 Cast와 충돌해
    # ORT 로드 시 "type mismatch"(예: Resize_output_cast0)를 낸다. value_info를 비우면
    # ORT가 노드 attribute 기준으로 타입을 새로 추론해 충돌이 해소된다.
    del model_fp16.graph.value_info[:]
    onnx.save(model_fp16, str(DST))
    print(f"저장: {DST.name} ({_mb(DST):.1f}MB)  [{_mb(SRC) / _mb(DST):.2f}× 축소]")


def verify(bench: bool) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess32 = ort.InferenceSession(str(SRC), so, providers=["CPUExecutionProvider"])
    sess16 = ort.InferenceSession(str(DST), so, providers=["CPUExecutionProvider"])

    inp = sess16.get_inputs()[0]
    name = inp.name
    h = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    w = inp.shape[3] if isinstance(inp.shape[3], int) else 640
    print(f"입력: {name} [1,3,{h},{w}] (dtype={inp.type})")

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, h, w), dtype=np.float32)

    out32 = sess32.run(None, {name: x})[0]
    out16 = sess16.run(None, {name: x})[0]
    print(f"출력 shape: fp32={out32.shape}  fp16={out16.shape}")
    diff = np.abs(out32.astype(np.float32) - out16.astype(np.float32))
    print(f"출력 max|Δ|={diff.max():.4f}  mean|Δ|={diff.mean():.5f}  (fp16 정밀도 손실 가늠)")

    if bench:
        for nm, sess in [("fp32", sess32), ("fp16", sess16)]:
            for _ in range(3):
                sess.run(None, {name: x})  # warmup
            t0 = time.perf_counter()
            for _ in range(20):
                sess.run(None, {name: x})
            ms = (time.perf_counter() - t0) / 20 * 1000
            print(f"  CPU 레이턴시 {nm}: {ms:.1f}ms/frame")
        print("  (참고: 브라우저 WebGPU FPS는 스파이크 /detection/spike/ 에서 실측)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="CPU 레이턴시/출력차 비교")
    args = ap.parse_args()
    convert()
    verify(args.bench)
    print(f"\n완료. 스파이크 측정용으로 /models/{DST.name} 에서 서빙됨 (서버 재시작 후).")


if __name__ == "__main__":
    main()
