"""
Phase 5: Edge-Sign 파이프라인 ONNX Runtime CPU 벤치마크.

각 구성요소의 레이턴시와 전체 파이프라인 FPS를 측정합니다.

사용법:
  python scripts/benchmark_pipeline.py
  python scripts/benchmark_pipeline.py --n_warmup 5 --n_runs 100
"""
import sys
import time
import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.e2e_pipeline import EdgeSignPipeline, postprocess_yolo, softmax

MODEL_DIR = ROOT / "model_space"
TEST_DIR  = ROOT / "data" / "aihub_traffic" / "test" / "images"

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _timeit(fn, warmup: int, runs: int) -> tuple[float, float, float]:
    """fn() 반복 실행 → (mean_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def _load_session(path: Path, providers=("CPUExecutionProvider",)):
    sess = ort.InferenceSession(str(path), providers=list(providers))
    return sess, sess.get_inputs()[0].name


# ─────────────────────────────────────────────────────────────────────────────
# 컴포넌트 벤치마크
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_components(warmup: int, runs: int):
    print("\n" + "="*60)
    print("Component-level Latency Benchmark  (CPU ONNX Runtime)")
    print("="*60)

    results = {}

    configs = [
        ("YOLOv8s FP32",        MODEL_DIR / "yolov8s_signs_fp32.onnx",       (1, 3, 640, 640), False),
        ("YOLOv8s W8A8",        MODEL_DIR / "yolov8s_signs_w8a8.onnx",        (1, 3, 640, 640), False),
        ("YOLOv8s W4A16",       MODEL_DIR / "yolov8s_signs_w4a16.onnx",       (1, 3, 640, 640), False),
        ("YOLOv8s SmoothQuant", MODEL_DIR / "yolov8s_signs_smoothquant.onnx", (1, 3, 640, 640), False),
        ("OCR FP32",            MODEL_DIR / "korean_ocr_net_fp32.onnx",        (1, 1, 64, 64),   False),
        ("OCR W8A8",            MODEL_DIR / "korean_ocr_net_w8a8.onnx",        (1, 1, 64, 64),   False),
        ("OCR W4A16",           MODEL_DIR / "korean_ocr_net_w4a16.onnx",       (1, 1, 64, 64),   False),
        ("OCR 1-Bit",           MODEL_DIR / "korean_ocr_net_1bit.onnx",        (1, 1, 64, 64),   False),
        ("TrafficSign FP32",    MODEL_DIR / "traffic_sign_net_fp32.onnx",      (1, 3, 32, 32),   False),
        ("TrafficSign W8A8",    MODEL_DIR / "traffic_sign_net_w8a8.onnx",      (1, 3, 32, 32),   False),
        ("TrafficSign W4A16",   MODEL_DIR / "traffic_sign_net_w4a16.onnx",     (1, 3, 32, 32),   False),
        ("TrafficSign 1-Bit",   MODEL_DIR / "traffic_sign_net_1bit.onnx",      (1, 3, 32, 32),   False),
        ("ReID W8A8",           MODEL_DIR / "reid_net_w8a8.onnx",              (1, 3, 64, 64),   False),
    ]

    print(f"\n  {'Model':<28} {'Mean(ms)':>9} {'Min(ms)':>8} {'Max(ms)':>8}  {'File(MB)':>8}")
    print("  " + "-"*70)

    for name, path, shape, _ in configs:
        if not path.exists():
            print(f"  {name:<28}  [not found: {path.name}]")
            continue
        try:
            sess, inp_name = _load_session(path)
            dummy = np.random.rand(*shape).astype(np.float32)
            mean_ms, min_ms, max_ms = _timeit(
                lambda: sess.run(None, {inp_name: dummy}),
                warmup=warmup, runs=runs
            )
            size_mb = path.stat().st_size / 1e6
            print(f"  {name:<28} {mean_ms:>9.2f} {min_ms:>8.2f} {max_ms:>8.2f}  {size_mb:>8.2f}")
            results[name] = {"mean_ms": mean_ms, "min_ms": min_ms, "max_ms": max_ms,
                             "size_mb": size_mb}
        except Exception as e:
            print(f"  {name:<28}  [ERROR: {e}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 FPS 벤치마크
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_pipeline_fps(warmup: int, runs: int):
    print("\n" + "="*60)
    print("Full Pipeline FPS Benchmark  (CPU ONNX Runtime)")
    print("="*60)

    # 테스트 프레임 로드
    frames = []
    if TEST_DIR.exists():
        seq_dirs = sorted(TEST_DIR.iterdir())
        for seq_dir in seq_dirs:
            imgs = sorted(seq_dir.glob("*.jpg"))[:runs + warmup]
            for img_path in imgs:
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append(img)
                if len(frames) >= runs + warmup:
                    break
            if len(frames) >= runs + warmup:
                break
    if not frames:
        print("  [INFO] 테스트 프레임 없음 - 640x480 더미 프레임 사용")
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                  for _ in range(runs + warmup)]

    print(f"  테스트 프레임: {len(frames)} ({frames[0].shape[1]}x{frames[0].shape[0]})")

    pipeline_configs = [
        {
            "name": "E0  FP32 All (fake)",
            "yolo":  MODEL_DIR / "yolov8s_signs_fp32.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_fp32.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_fp32.onnx",
        },
        {
            "name": "E1  W8A8 Det (fake)",
            "yolo":  MODEL_DIR / "yolov8s_signs_w8a8.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_fp32.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_fp32.onnx",
        },
        {
            "name": "E3  W8A8 All (fake)",
            "yolo":  MODEL_DIR / "yolov8s_signs_w8a8.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_w8a8.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_w8a8.onnx",
        },
        # ── Real INT8 Static QDQ ──────────────────────────────────────────────
        {
            "name": "E0  FP32→INT8 Static (real)",
            "yolo":  MODEL_DIR / "yolov8s_signs_int8_static.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_fp32.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_fp32.onnx",
        },
        {
            "name": "E3  INT8 Static All (real)",
            "yolo":  MODEL_DIR / "yolov8s_signs_int8_static.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_int8_static.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_int8_static.onnx",
        },
        {
            "name": "E5  SQ+W8A8 (fake)",
            "yolo":  MODEL_DIR / "yolov8s_signs_smoothquant.onnx",
            "ocr":   MODEL_DIR / "korean_ocr_net_w8a8.onnx",
            "tsign": MODEL_DIR / "traffic_sign_net_w8a8.onnx",
        },
    ]

    fps_results = {}
    print(f"\n  {'Config':<28} {'Latency/frame':>14} {'FPS':>8}  {'30fps?':>7}")
    print("  " + "-"*65)

    for cfg in pipeline_configs:
        yolo_path  = cfg["yolo"]
        ocr_path   = cfg["ocr"]
        tsign_path = cfg["tsign"]

        if not all(p.exists() for p in [yolo_path, ocr_path, tsign_path]):
            missing = [p.name for p in [yolo_path, ocr_path, tsign_path] if not p.exists()]
            print(f"  {cfg['name']:<28}  [missing: {missing}]")
            continue

        try:
            pipe = EdgeSignPipeline(
                yolo_onnx=str(yolo_path),
                ocr_onnx=str(ocr_path),
                tsign_onnx=str(tsign_path),
            )

            # warmup
            for i in range(min(warmup, len(frames))):
                pipe.process_frame(frames[i % len(frames)])

            # benchmark
            pipe.reset()
            t_start = time.perf_counter()
            n = min(runs, len(frames))
            for i in range(n):
                pipe.process_frame(frames[i])
            elapsed = time.perf_counter() - t_start

            fps = n / elapsed
            lat_ms = elapsed / n * 1000
            meet_30 = "YES" if fps >= 30 else "no"

            print(f"  {cfg['name']:<28} {lat_ms:>12.1f}ms {fps:>8.1f}  {meet_30:>7}")
            fps_results[cfg["name"]] = {"fps": fps, "latency_ms": lat_ms}

        except Exception as e:
            print(f"  {cfg['name']:<28}  [ERROR: {e}]")

    return fps_results


# ─────────────────────────────────────────────────────────────────────────────
# 요약 보고서
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(comp_results: dict, fps_results: dict):
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)

    print("\n[Detector Latency Comparison]")
    det_keys = [k for k in comp_results if k.startswith("YOLOv8s")]
    if det_keys:
        base = comp_results.get("YOLOv8s FP32", {}).get("mean_ms", 1.0)
        for k in det_keys:
            d = comp_results[k]
            speedup = base / d["mean_ms"] if d["mean_ms"] > 0 else 0
            print(f"  {k:<28} {d['mean_ms']:>7.2f}ms  {speedup:>5.2f}x speedup vs FP32")

    print("\n[OCR Latency Comparison]")
    ocr_keys = [k for k in comp_results if k.startswith("OCR")]
    if ocr_keys:
        base = comp_results.get("OCR FP32", {}).get("mean_ms", 1.0)
        for k in ocr_keys:
            d = comp_results[k]
            speedup = base / d["mean_ms"] if d["mean_ms"] > 0 else 0
            print(f"  {k:<28} {d['mean_ms']:>7.2f}ms  {speedup:>5.2f}x speedup vs FP32")

    print("\n[Pipeline FPS]")
    for name, r in fps_results.items():
        bar = "#" * int(r["fps"] / 2)
        print(f"  {name:<28} {r['fps']:>6.1f} FPS  {bar}")

    print("\n[Conclusion]")
    if fps_results:
        best = max(fps_results, key=lambda k: fps_results[k]["fps"])
        print(f"  Best FPS: {best}  ({fps_results[best]['fps']:.1f} FPS)")
        if any(r["fps"] >= 30 for r in fps_results.values()):
            print("  30+ FPS target: ACHIEVED on some configs")
        else:
            top_fps = max(r["fps"] for r in fps_results.values())
            print(f"  30+ FPS target: NOT MET (best: {top_fps:.1f} FPS)")
            print("  -> GPU/INT8 runtime deployment would close this gap")
            print("  -> CPU ORT fake-quant (FP32 weights) cannot show INT speedup")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Edge-Sign Pipeline Benchmark")
    parser.add_argument("--n_warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--n_runs",   type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--comp_only",  action="store_true", help="Component benchmark only")
    parser.add_argument("--pipe_only",  action="store_true", help="Pipeline benchmark only")
    args = parser.parse_args()

    print(f"\nEdge-Sign Phase 5 Benchmark")
    print(f"  Warmup: {args.n_warmup}  Runs: {args.n_runs}")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Providers: {ort.get_available_providers()}")

    comp_results = {}
    fps_results  = {}

    if not args.pipe_only:
        comp_results = benchmark_components(args.n_warmup, args.n_runs)
    if not args.comp_only:
        fps_results  = benchmark_pipeline_fps(args.n_warmup, args.n_runs)

    print_summary(comp_results, fps_results)
    print()


if __name__ == "__main__":
    main()
