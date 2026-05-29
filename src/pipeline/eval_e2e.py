"""
Edge-Sign v2  E2E Comprehensive Evaluation
E0~E7 전체 실험 구성을 종합 평가하고 Final Score를 산출한다.

각 실험의 검출/추적/인식 정확도는 이미 완료된 단계별 실험 결과를 활용하며,
파이프라인 FPS는 실제 추론으로 측정한다.

사용법:
  python src/pipeline/eval_e2e.py
  python src/pipeline/eval_e2e.py --fps_only
  python src/pipeline/eval_e2e.py --table_only
"""

import argparse
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "model_space"
TEST_DIR  = ROOT / "data" / "aihub_traffic" / "test" / "images"

# ─────────────────────────────────────────────────────────────────────────────
# 실험별 사전 측정된 정확도 결과 (단계별 실험에서 수집)
# ─────────────────────────────────────────────────────────────────────────────

# v2 Stratified Split (2026-05-30 재학습) 검출기 mAP@0.5
# E2 = E0 검출기, E3/E6 = E1 검출기, E7 = E4 검출기
DET_MAP50 = {
    "E0": 0.587, "E1": 0.587, "E2": 0.587,
    "E3": 0.587, "E4": 0.523, "E5": 0.587,
    "E6": 0.587, "E7": 0.523,
}
DET_MAP5095 = {
    "E0": 0.381, "E1": 0.381, "E2": 0.381,
    "E3": 0.381, "E4": 0.322, "E5": 0.381,
    "E6": 0.381, "E7": 0.322,
}

# v2 추적 MOTA (주야간 stratified test 2시퀀스)
TRACK_MOTA = {
    "E0": 0.295, "E1": 0.291, "E2": 0.295,
    "E3": 0.291, "E4": 0.176, "E5": 0.280,
    "E6": 0.068, "E7": 0.176,
}
TRACK_IDF1 = {
    "E0": 0.495, "E1": 0.491, "E2": 0.495,
    "E3": 0.491, "E4": 0.309, "E5": 0.479,
    "E6": 0.330, "E7": 0.309,
}
TRACK_HOTA = {
    "E0": 0.570, "E1": 0.565, "E2": 0.570,
    "E3": 0.565, "E4": 0.424, "E5": 0.558,
    "E6": 0.444, "E7": 0.424,
}
# v2 test (주간+야간 통합)는 GT 3,386 객체 평균으로 IDSW 절대값이 v1 대비 큼
TRACK_IDSW = {
    "E0": 28, "E1": 44, "E2": 28,
    "E3": 44, "E4": 21, "E5": 28,
    "E6":  2, "E7": 21,
}

# 인식 정확도 — OCR Top-1 (KoreanOCRNet)
OCR_TOP1 = {
    "E0": 98.5, "E1": 98.5, "E2": 98.4,
    "E3": 98.4, "E4": 54.6, "E5": 98.5,
    "E6": 98.4, "E7":  0.3,
}

# 인식 정확도 — 교통표지판 Top-1 (TrafficSignNet, GTSDB val)
TSIGN_TOP1 = {
    "E0": 62.8, "E1": 62.8, "E2": 63.2,
    "E3": 63.2, "E4": 49.2, "E5": 62.8,
    "E6": 63.2, "E7": 12.8,
}

# v2 이론적 배포 크기 (MB) — INT 포맷 환산
# YOLOv8s INT8 ≈ 5.4 MB (Static QDQ 실측 11.66 MB → INT 이론치 5.4 MB)
MODEL_SIZE_MB = {
    "E0": 22.3,   # YOLOv8s FP32(21.5) + OCR FP32(0.69) + TS FP32(0.12)
    "E1":  6.2,   # YOLOv8s W8A8(5.4) + OCR FP32(0.69) + TS FP32(0.12)
    "E2": 21.7,   # YOLOv8s FP32(21.5) + OCR W8A8(0.17) + TS W8A8(0.03)
    "E3":  5.6,   # YOLOv8s W8A8(5.4) + OCR W8A8(0.17) + TS W8A8(0.03)
    "E4":  2.8,   # YOLOv8s W4A16(2.7) + OCR W4A16(0.09) + TS W4A16(0.02)
    "E5":  5.6,   # YOLOv8s SQ(5.4) + OCR W8A8(0.17) + TS W8A8(0.03)
    "E6":  5.8,   # E3 + ReID W8A8(0.24)
    "E7":  2.7,   # YOLOv8s W4A16(2.7) + OCR 1-Bit(0.02) + TS 1-Bit(0.003)
}

# 실험별 모델 파일 구성
EXPERIMENT_CONFIGS = {
    "E0": {
        "yolo":  "yolov8s_signs_fp32.onnx",
        "ocr":   "korean_ocr_net_fp32.onnx",
        "tsign": "traffic_sign_net_fp32.onnx",
        "label": "FP32 All",
    },
    "E1": {
        "yolo":  "yolov8s_signs_w8a8.onnx",
        "ocr":   "korean_ocr_net_fp32.onnx",
        "tsign": "traffic_sign_net_fp32.onnx",
        "label": "W8A8 Det",
    },
    "E2": {
        "yolo":  "yolov8s_signs_fp32.onnx",
        "ocr":   "korean_ocr_net_w8a8.onnx",
        "tsign": "traffic_sign_net_w8a8.onnx",
        "label": "FP32 Det + W8A8 Rec",
    },
    "E3": {
        "yolo":  "yolov8s_signs_w8a8.onnx",
        "ocr":   "korean_ocr_net_w8a8.onnx",
        "tsign": "traffic_sign_net_w8a8.onnx",
        "label": "W8A8 All",
    },
    "E4": {
        "yolo":  "yolov8s_signs_w4a16.onnx",
        "ocr":   "korean_ocr_net_w4a16.onnx",
        "tsign": "traffic_sign_net_w4a16.onnx",
        "label": "W4A16 All",
    },
    "E5": {
        "yolo":  "yolov8s_signs_smoothquant.onnx",
        "ocr":   "korean_ocr_net_w8a8.onnx",
        "tsign": "traffic_sign_net_w8a8.onnx",
        "label": "SmoothQuant+W8A8",
    },
    "E7": {
        "yolo":  "yolov8s_signs_w4a16.onnx",
        "ocr":   "korean_ocr_net_1bit.onnx",
        "tsign": "traffic_sign_net_1bit.onnx",
        "label": "W4A16 Det + 1-Bit Rec",
    },
}

# E6는 BoT-SORT 사용 — FPS는 eval_botsort.py에서 측정된 20.4 FPS 사용
E6_FPS_MEASURED = 20.4


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 FPS 측정
# ─────────────────────────────────────────────────────────────────────────────

def load_test_frames(n: int) -> list:
    """AI Hub test 시퀀스에서 프레임 로드 (없으면 더미 사용)."""
    frames = []
    if TEST_DIR.exists():
        for seq_dir in sorted(TEST_DIR.iterdir()):
            for img_path in sorted(seq_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append(img)
                if len(frames) >= n:
                    break
            if len(frames) >= n:
                break
    if not frames:
        print("  [INFO] 테스트 프레임 없음 — 더미 프레임(640x480) 사용")
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n)]
    return frames[:n]


def measure_fps(exp_id: str, cfg: dict, frames: list, warmup: int = 3) -> float:
    """단일 실험 구성의 파이프라인 FPS를 측정한다."""
    from src.pipeline.e2e_pipeline import EdgeSignPipeline

    yolo_path  = MODEL_DIR / cfg["yolo"]
    ocr_path   = MODEL_DIR / cfg["ocr"]
    tsign_path = MODEL_DIR / cfg["tsign"]

    missing = [p.name for p in [yolo_path, ocr_path, tsign_path] if not p.exists()]
    if missing:
        print(f"  {exp_id}: 모델 파일 없음 {missing} — 건너뜀")
        return None

    try:
        pipe = EdgeSignPipeline(
            yolo_onnx=str(yolo_path),
            ocr_onnx=str(ocr_path),
            tsign_onnx=str(tsign_path),
        )
        # warmup
        for i in range(warmup):
            pipe.process_frame(frames[i % len(frames)])

        pipe.reset()
        t0 = time.perf_counter()
        for frame in frames:
            pipe.process_frame(frame)
        elapsed = time.perf_counter() - t0
        return len(frames) / elapsed
    except Exception as e:
        print(f"  {exp_id}: FPS 측정 오류 — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Final Score 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(exp_id: str, fps: float, fps_e0: float) -> dict:
    """
    Final Score = 0.6 * PerfNorm + 0.2 * SpeedNorm + 0.2 * MemNorm
      PerfNorm  = OCR_Top1_i / OCR_Top1_E0
      SpeedNorm = Latency_E0 / Latency_i  (= FPS_i / FPS_E0)
      MemNorm   = min(1, Size_E0 / Size_i)
    """
    ocr_e0   = OCR_TOP1["E0"]
    size_e0  = MODEL_SIZE_MB["E0"]

    perf_norm  = OCR_TOP1[exp_id] / ocr_e0
    speed_norm = (fps / fps_e0) if (fps and fps_e0) else 0.0
    mem_norm   = min(1.0, size_e0 / MODEL_SIZE_MB[exp_id])

    final = 0.6 * perf_norm + 0.2 * speed_norm + 0.2 * mem_norm
    return {
        "PerfNorm":  round(perf_norm, 4),
        "SpeedNorm": round(speed_norm, 4),
        "MemNorm":   round(mem_norm, 4),
        "FinalScore": round(final, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_results(fps_map: dict):
    fps_e0 = fps_map.get("E0") or 1.0

    print("\n" + "=" * 90)
    print(" Edge-Sign v2  E2E Comprehensive Evaluation Results")
    print("=" * 90)

    # ── 검출 + 추적 ─────────────────────────────────────────────────────────
    print("\n[Detection & Tracking Metrics]")
    print(f"  {'ID':<4} {'Label':<24} {'mAP@.5':>7} {'mAP@.5:.95':>11} {'MOTA':>7} {'IDF1':>7} {'HOTA':>7} {'IDSW':>5}")
    print("  " + "-" * 78)
    for eid in ["E0","E1","E2","E3","E4","E5","E6","E7"]:
        lbl = EXPERIMENT_CONFIGS.get(eid, {}).get("label", "BoT-SORT" if eid=="E6" else "—")
        print(f"  {eid:<4} {lbl:<24} "
              f"{DET_MAP50[eid]:>7.3f} {DET_MAP5095[eid]:>11.3f} "
              f"{TRACK_MOTA[eid]:>7.3f} {TRACK_IDF1[eid]:>7.3f} "
              f"{TRACK_HOTA[eid]:>7.3f} {TRACK_IDSW[eid]:>5}")

    # ── 인식 ────────────────────────────────────────────────────────────────
    print("\n[Recognition Accuracy]")
    print(f"  {'ID':<4} {'Label':<24} {'OCR Top-1':>10} {'OCR Delta':>10} {'TS Top-1':>9} {'TS Delta':>9}")
    print("  " + "-" * 72)
    for eid in ["E0","E1","E2","E3","E4","E5","E6","E7"]:
        lbl = EXPERIMENT_CONFIGS.get(eid, {}).get("label", "BoT-SORT")
        ocr_d = OCR_TOP1[eid]  - OCR_TOP1["E0"]
        ts_d  = TSIGN_TOP1[eid] - TSIGN_TOP1["E0"]
        print(f"  {eid:<4} {lbl:<24} "
              f"{OCR_TOP1[eid]:>10.1f}% "
              f"{ocr_d:>+9.1f}pp "
              f"{TSIGN_TOP1[eid]:>9.1f}% "
              f"{ts_d:>+8.1f}pp")

    # ── FPS + Final Score ────────────────────────────────────────────────────
    print("\n[Pipeline FPS & Final Score]")
    print(f"  {'ID':<4} {'Label':<24} {'FPS':>7} {'Size(MB)':>9} {'PerfN':>7} {'SpeedN':>7} {'MemN':>6} {'Score':>7}")
    print("  " + "-" * 78)
    for eid in ["E0","E1","E2","E3","E4","E5","E6","E7"]:
        fps = fps_map.get(eid)
        lbl = EXPERIMENT_CONFIGS.get(eid, {}).get("label", "BoT-SORT")
        if fps is None:
            print(f"  {eid:<4} {lbl:<24} {'—':>7} {MODEL_SIZE_MB[eid]:>9.1f}  — (FPS 측정 실패)")
            continue
        sc = compute_final_score(eid, fps, fps_e0)
        print(f"  {eid:<4} {lbl:<24} "
              f"{fps:>7.1f} {MODEL_SIZE_MB[eid]:>9.1f} "
              f"{sc['PerfNorm']:>7.4f} {sc['SpeedNorm']:>7.4f} "
              f"{sc['MemNorm']:>6.4f} {sc['FinalScore']:>7.4f}")

    # ── 최적 구성 ────────────────────────────────────────────────────────────
    print("\n[Key Findings]")
    scores = {}
    for eid in ["E0","E1","E2","E3","E4","E5","E6","E7"]:
        fps = fps_map.get(eid)
        if fps:
            scores[eid] = compute_final_score(eid, fps, fps_e0)["FinalScore"]

    if scores:
        best_id = max(scores, key=lambda k: scores[k])
        print(f"  Best Final Score : {best_id}  ({scores[best_id]:.4f})")
        print(f"  30+ FPS achieved : {[e for e,f in fps_map.items() if f and f >= 30]}")
        print(f"  Size < 15 MB     : {[e for e in MODEL_SIZE_MB if MODEL_SIZE_MB[e] < 15]}")
        print(f"  OCR > 95%        : {[e for e in OCR_TOP1 if OCR_TOP1[e] > 95]}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E Comprehensive Evaluation")
    parser.add_argument("--fps_only",   action="store_true", help="FPS 측정만 실행")
    parser.add_argument("--table_only", action="store_true", help="FPS 측정 없이 테이블만 출력")
    parser.add_argument("--warmup",     type=int, default=3,  help="Warmup 프레임 수")
    parser.add_argument("--n_frames",   type=int, default=50, help="FPS 측정 프레임 수")
    args = parser.parse_args()

    print(f"\nEdge-Sign v2  E2E Evaluation")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Providers: {ort.get_available_providers()}")

    fps_map: dict[str, float | None] = {}

    if not args.table_only:
        frames = load_test_frames(args.n_frames + args.warmup)
        print(f"  Frames loaded: {len(frames)}  ({frames[0].shape[1]}x{frames[0].shape[0]})")

        for eid, cfg in EXPERIMENT_CONFIGS.items():
            print(f"  Measuring FPS: {eid} ({cfg['label']}) ...", end=" ", flush=True)
            fps = measure_fps(eid, cfg, frames[:args.n_frames], warmup=args.warmup)
            fps_map[eid] = fps
            print(f"{fps:.1f} FPS" if fps else "FAILED")

        # E6: BoT-SORT FPS는 eval_botsort.py에서 측정된 값 사용
        fps_map["E6"] = E6_FPS_MEASURED
        print(f"  E6 (BoT-SORT): {E6_FPS_MEASURED} FPS  (eval_botsort.py 측정값 사용)")
    else:
        # table_only 모드: 임시 FPS 값으로 테이블 출력
        fps_defaults = {
            "E0": 22.4, "E1": 24.8, "E2": 21.8, "E3": 25.0,
            "E4": 25.7, "E5": 20.8, "E6": 20.4, "E7": 25.5,
        }
        fps_map = fps_defaults

    if not args.fps_only:
        print_results(fps_map)

    return fps_map


if __name__ == "__main__":
    main()
