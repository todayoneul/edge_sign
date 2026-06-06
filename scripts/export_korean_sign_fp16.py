"""KoreanSignNet FP32 ONNX → FP16 ONNX (폰 온디바이스 인식기용).

검출기 fp16(export_fp16_detector.py)과 동일한 keep_io_types 전략: I/O는 float32로 유지해
클라 전처리/후처리를 그대로 재사용하고, 내부 가중치/연산만 fp16으로 낮춰 크기/대역폭을 줄인다.
소형 모델이라 정확도 영향은 미미해야 하며 verify로 max|Δ|를 확인한다.

사용법:
  python scripts/export_korean_sign_fp16.py
  python scripts/export_korean_sign_fp16.py --bench
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "model_space" / "korean_sign_net_fp32.onnx"
DST = ROOT / "model_space" / "korean_sign_net_fp16.onnx"


def _mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def convert() -> None:
    if not SRC.exists():
        raise SystemExit(f"원본 없음: {SRC}")
    print(f"로드: {SRC.name} ({_mb(SRC):.3f}MB)")
    model = onnx.shape_inference.infer_shapes(onnx.load(str(SRC)))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    del model_fp16.graph.value_info[:]
    onnx.save(model_fp16, str(DST))
    print(f"저장: {DST.name} ({_mb(DST):.3f}MB)  [{_mb(SRC) / _mb(DST):.2f}× 축소]")


def verify(bench: bool) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s32 = ort.InferenceSession(str(SRC), so, providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(str(DST), so, providers=["CPUExecutionProvider"])

    inp = s16.get_inputs()[0]
    name = inp.name
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    print(f"입력: {name} {shape} (dtype={inp.type})")

    rng = np.random.default_rng(0)
    x = rng.random(shape, dtype=np.float32)
    out32 = s32.run(None, {name: x})[0]
    out16 = s16.run(None, {name: x})[0]
    diff = np.abs(out32.astype(np.float32) - out16.astype(np.float32))
    same_argmax = int(out32.argmax(-1).flatten()[0]) == int(out16.argmax(-1).flatten()[0])
    print(f"출력 shape={out32.shape}  max|Δ|={diff.max():.5f}  argmax 일치={same_argmax}")

    if bench:
        import time

        for nm, s in [("fp32", s32), ("fp16", s16)]:
            for _ in range(3):
                s.run(None, {name: x})
            t0 = time.perf_counter()
            for _ in range(50):
                s.run(None, {name: x})
            print(f"  CPU {nm}: {(time.perf_counter() - t0) / 50 * 1000:.3f}ms/frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    convert()
    verify(args.bench)


if __name__ == "__main__":
    main()
