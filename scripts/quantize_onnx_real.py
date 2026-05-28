"""
Real INT8 ONNX Static Quantization — ORT QDQ (Quantize-DeQuantize) 방식.

CNN(YOLOv8 등)에는 Dynamic quant이 역효과(Conv 실시간 양자화 오버헤드).
Static QDQ format이 CPU VNNI/AVX2에서 실제 INT8 커널을 사용해 2-4× 빠름.

처리 흐름:
  1. quant_pre_process()  — shape 추론 + 그래프 최적화 + BN-Conv 융합
  2. quantize_static()    — 캘리브레이션 데이터로 활성화 범위 고정 → QDQ INT8

사용법:
  python scripts/quantize_onnx_real.py               # 전체 static quant
  python scripts/quantize_onnx_real.py --models yolo # 검출기만
  python scripts/quantize_onnx_real.py --bench        # 양자화 후 벤치마크까지
"""
import sys
import io
import argparse
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "model_space"
TEST_DIR  = ROOT / "data" / "aihub_traffic" / "test" / "images"
OCR_DIR   = ROOT / "data" / "korean_ocr" / "val"
GTSDB_DIR = ROOT / "data" / "GTSDB" / "FullIJCNN2013"

try:
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationDataReader,
        quant_pre_process,
    )
except ImportError as e:
    print(f"[ERROR] {e}\npip install onnxruntime onnx")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Data Readers
# ─────────────────────────────────────────────────────────────────────────────

class YoloCalibReader(CalibrationDataReader):
    """YOLOv8s 캘리브레이션: 640×640 RGB float [0,1]."""

    def __init__(self, n: int = 80):
        self._n = n
        self._iter = iter(self._load())

    def _load(self):
        imgs = []
        if TEST_DIR.exists():
            for seq in sorted(TEST_DIR.iterdir()):
                for p in sorted(seq.glob("*.jpg")):
                    img = cv2.imread(str(p))
                    if img is not None:
                        imgs.append(img)
                    if len(imgs) >= self._n:
                        break
                if len(imgs) >= self._n:
                    break
        if not imgs:
            print("  [Calib YOLO] 더미 이미지 사용")
            imgs = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                    for _ in range(self._n)]
        print(f"  [Calib YOLO] {len(imgs)} frames")
        for img in imgs:
            t = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
            nchw = np.transpose(t.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]  # [1,3,640,640]
            yield {"images": nchw}

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._load())


class OcrCalibReader(CalibrationDataReader):
    """KoreanOCRNet 캘리브레이션: 1×64×64 gray float."""

    def __init__(self, n: int = 200):
        self._n = n
        self._iter = iter(self._load())

    def _load(self):
        imgs = []
        if OCR_DIR.exists():
            for cls_dir in sorted(OCR_DIR.iterdir()):
                for p in sorted(cls_dir.glob("*.jpg"))[:10]:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imgs.append(img)
                    if len(imgs) >= self._n:
                        break
                if len(imgs) >= self._n:
                    break
        if not imgs:
            imgs = [np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                    for _ in range(self._n)]
        print(f"  [Calib OCR] {len(imgs)} samples")
        for img in imgs:
            t = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
            yield {"image": t[np.newaxis, np.newaxis]}     # [1, 1, 64, 64]  (input name='image')

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._load())


class TSignCalibReader(CalibrationDataReader):
    """TrafficSignNet 캘리브레이션: 3×32×32 RGB float (GTSDB 크롭)."""

    def __init__(self, n: int = 200):
        self._n = n
        self._iter = iter(self._load())

    def _load(self):
        imgs = []
        # GTSDB gt.txt 크롭
        gt_path = GTSDB_DIR / "gt.txt"
        if gt_path.exists():
            import random; random.seed(42)
            lines = open(str(gt_path)).readlines()
            random.shuffle(lines)
            for line in lines[:self._n * 3]:
                parts = line.strip().split(";")
                if len(parts) < 5:
                    continue
                img_file = GTSDB_DIR / parts[0]
                if not img_file.exists():
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                imgs.append(roi)
                if len(imgs) >= self._n:
                    break
        if not imgs:
            imgs = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    for _ in range(self._n)]
        print(f"  [Calib TSign] {len(imgs)} crops")
        for img in imgs:
            t = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2RGB)
            tensor = (t.astype(np.float32) / 255.0 - 0.5) / 0.5
            yield {"images": np.transpose(tensor, (2, 0, 1))[np.newaxis]}  # [1,3,32,32]  (input name='images')

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._load())


# ─────────────────────────────────────────────────────────────────────────────
# Static Quantization
# ─────────────────────────────────────────────────────────────────────────────

def static_quantize(
    src: Path,
    dst: Path,
    calib_reader: CalibrationDataReader,
    act_type=QuantType.QUInt8,   # ReLU/SiLU 이후 양수 → UInt8 적합
    skip_preproc: bool = False,
):
    """FP32 ONNX → Static INT8 QDQ ONNX."""
    print(f"\n[StaticQuant] {src.name}")
    t0 = time.time()

    # Step 1: quant_pre_process (shape inference + 그래프 최적화)
    if skip_preproc:
        prep_path = src
        print(f"  PreProcess: skip")
    else:
        prep_path = dst.parent / f"_prep_{src.stem}.onnx"
        try:
            print(f"  PreProcess... ", end="", flush=True)
            quant_pre_process(
                input_model_path=str(src),
                output_model_path=str(prep_path),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=True,
            )
            print("done")
        except Exception as e:
            print(f"failed ({e}) — 원본으로 진행")
            prep_path = src

    # Step 2: quantize_static
    print(f"  Quantize (calibration)... ", end="", flush=True)
    try:
        quantize_static(
            model_input=str(prep_path),
            model_output=str(dst),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=act_type,
            per_channel=True,
            reduce_range=False,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "EnableSubgraph": True,
            },
        )
        elapsed = time.time() - t0
        orig_mb = src.stat().st_size / 1e6
        quant_mb = dst.stat().st_size / 1e6
        print(f"done [{elapsed:.1f}s]")
        print(f"  {src.name}: {orig_mb:.2f} MB -> {quant_mb:.2f} MB "
              f"({orig_mb/quant_mb:.2f}x smaller)")

        # PreProcess 임시파일 삭제
        if prep_path != src and prep_path.exists():
            prep_path.unlink()
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        if prep_path != src and prep_path.exists():
            prep_path.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Latency Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench(path: Path, dummy: np.ndarray, label: str, n: int = 50):
    if not path.exists():
        return None
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {inp: dummy})
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {inp: dummy})
        ms = (time.perf_counter() - t0) / n * 1000
        return ms
    except Exception as e:
        print(f"  [bench ERROR {label}]: {e}")
        return None


def compare_latency(fp32: Path, int8: Path, dummy: np.ndarray, n: int = 50):
    fp_ms = bench(fp32, dummy, "FP32", n)
    it_ms = bench(int8, dummy, "INT8", n)
    if fp_ms and it_ms:
        speedup = fp_ms / it_ms
        fp_mb  = fp32.stat().st_size / 1e6
        it_mb  = int8.stat().st_size / 1e6
        marker = "✓" if speedup >= 1.5 else ("~" if speedup >= 0.9 else "✗")
        print(f"  {marker} FP32 {fp_ms:7.2f}ms ({fp_mb:.2f}MB)  "
              f"INT8 {it_ms:7.2f}ms ({it_mb:.2f}MB)  "
              f"-> {speedup:.2f}x speed  {fp_mb/it_mb:.2f}x size")
        return speedup
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Verify (출력 분포 비교)
# ─────────────────────────────────────────────────────────────────────────────

def verify(fp32: Path, int8: Path, dummy: np.ndarray):
    if not int8.exists():
        return
    try:
        s0 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
        s1 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
        o0 = s0.run(None, {s0.get_inputs()[0].name: dummy})[0]
        o1 = s1.run(None, {s1.get_inputs()[0].name: dummy})[0]
        cos = float(np.dot(o0.ravel(), o1.ravel()) /
                    (np.linalg.norm(o0.ravel()) * np.linalg.norm(o1.ravel()) + 1e-9))
        maxd = float(np.abs(o0 - o1).max())
        print(f"  Accuracy: CosSim={cos:.4f}  MaxDiff={maxd:.4f}")
    except Exception as e:
        print(f"  [verify ERROR] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["all", "yolo", "ocr", "tsign"],
                        default=["all"])
    parser.add_argument("--bench",    action="store_true", help="양자화 후 벤치마크")
    parser.add_argument("--n_runs",   type=int, default=50)
    parser.add_argument("--skip_preproc", action="store_true",
                        help="quant_pre_process 건너뜀 (빠르지만 정확도 저하 가능)")
    args = parser.parse_args()

    do_all   = "all" in args.models
    do_yolo  = do_all or "yolo"  in args.models
    do_ocr   = do_all or "ocr"   in args.models
    do_tsign = do_all or "tsign" in args.models

    print(f"\nStatic INT8 QDQ Quantization")
    print(f"  ORT {ort.__version__}  Providers: {ort.get_available_providers()}")
    print("="*60)

    results = {}

    # ── YOLOv8s ──────────────────────────────────────────────────────────────
    if do_yolo:
        src = MODEL_DIR / "yolov8s_signs_fp32.onnx"
        dst = MODEL_DIR / "yolov8s_signs_int8_static.onnx"
        ok = static_quantize(
            src, dst,
            calib_reader=YoloCalibReader(n=100),
            act_type=QuantType.QUInt8,  # SiLU이후 양수 분포 → UInt8
            skip_preproc=args.skip_preproc,
        )
        if ok:
            dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
            verify(src, dst, dummy)
            if args.bench:
                print(f"  Latency ({args.n_runs} runs):")
                sp = compare_latency(src, dst, dummy, args.n_runs)
                results["YOLOv8s"] = sp

    # ── KoreanOCRNet ─────────────────────────────────────────────────────────
    if do_ocr:
        src = MODEL_DIR / "korean_ocr_net_fp32.onnx"
        dst = MODEL_DIR / "korean_ocr_net_int8_static.onnx"
        ok = static_quantize(
            src, dst,
            calib_reader=OcrCalibReader(n=300),
            act_type=QuantType.QUInt8,
            skip_preproc=args.skip_preproc,
        )
        if ok:
            dummy = np.random.rand(1, 1, 64, 64).astype(np.float32)
            verify(src, dst, dummy)
            if args.bench:
                print(f"  Latency ({args.n_runs} runs):")
                sp = compare_latency(src, dst, dummy, args.n_runs)
                results["OCRNet"] = sp

    # ── TrafficSignNet ────────────────────────────────────────────────────────
    if do_tsign:
        src = MODEL_DIR / "traffic_sign_net_fp32.onnx"
        dst = MODEL_DIR / "traffic_sign_net_int8_static.onnx"
        ok = static_quantize(
            src, dst,
            calib_reader=TSignCalibReader(n=200),
            act_type=QuantType.QUInt8,
            skip_preproc=args.skip_preproc,
        )
        if ok:
            dummy = np.random.rand(1, 3, 32, 32).astype(np.float32)
            verify(src, dst, dummy)
            if args.bench:
                print(f"  Latency ({args.n_runs} runs):")
                sp = compare_latency(src, dst, dummy, args.n_runs)
                results["TrafficSign"] = sp

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("생성된 INT8 ONNX 파일:")
    for f in sorted(MODEL_DIR.glob("*int8_static*.onnx")):
        fp32_name = f.name.replace("_int8_static", "_fp32")
        fp32 = MODEL_DIR / fp32_name
        if fp32.exists():
            ratio = fp32.stat().st_size / f.stat().st_size
            print(f"  {f.name:<45} {f.stat().st_size/1e6:.2f} MB  ({ratio:.2f}x)")
        else:
            print(f"  {f.name:<45} {f.stat().st_size/1e6:.2f} MB")

    if results:
        print("\nSpeedup Summary:")
        for name, sp in results.items():
            if sp:
                print(f"  {name:<20} {sp:.2f}x")


if __name__ == "__main__":
    main()
