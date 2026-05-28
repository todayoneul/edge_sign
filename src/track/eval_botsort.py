"""
E6 BoT-SORT 추적 평가 스크립트.

W8A8 YOLOv8s 검출기 + BoT-SORT (CMC + W8A8 ReID) 조합으로
ByteTrack E1 결과와 비교.

사용법:
  python src/track/eval_botsort.py
  python src/track/eval_botsort.py --no_reid   # CMC만, ReID 없음
  python src/track/eval_botsort.py --no_cmc    # ReID만, CMC 없음
"""
import argparse
import sys
import time
import io
from pathlib import Path

import cv2
import numpy as np

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DET_ONNX  = ROOT / "model_space" / "yolov8s_signs_w8a8.onnx"
DEFAULT_REID_ONNX = ROOT / "model_space" / "reid_net_w8a8.onnx"
TEST_BASE         = ROOT / "data" / "aihub_traffic" / "test"

DET_CONF = 0.25
NMS_IOU  = 0.45

from src.track.eval_tracking import (
    OnnxDetector, load_gt_sequence, assign_gt_track_ids, compute_mot_metrics
)
from src.track.botsort import BoTSORTTracker, OnnxReIDNet


def evaluate_sequence_botsort(seq_dir: Path, detector: OnnxDetector,
                               tracker: BoTSORTTracker,
                               verbose: bool = True) -> tuple:
    img_paths, gt_raw = load_gt_sequence(seq_dir)
    gt_tracks = assign_gt_track_ids(gt_raw)

    pred_tracks = {}
    total_time  = 0.0

    for fidx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.perf_counter()
        dets   = detector.detect(img)
        tracks = tracker.update(dets, img)   # BoTSORTTracker
        total_time += time.perf_counter() - t0

        frame_preds = []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            frame_preds.append([x1, y1, x2, y2, t.score, t.cls, t.track_id])
        pred_tracks[fidx] = frame_preds

        if verbose and fidx % 20 == 0:
            gt_n   = len(gt_tracks.get(fidx, []))
            pred_n = len(frame_preds)
            print(f"    frame {fidx:4d}: GT={gt_n}, Pred={pred_n}, tracks={len(tracks)}")

    fps     = len(img_paths) / max(total_time, 1e-6)
    metrics = compute_mot_metrics(gt_tracks, pred_tracks)
    return metrics, fps


def run_all_sequences(det_onnx: Path, reid_onnx: Path | None,
                      use_cmc: bool = True, verbose: bool = True):
    print(f"\n[BoT-SORT E6]")
    print(f"  Detector : {det_onnx.name}")
    print(f"  ReID     : {reid_onnx.name if reid_onnx else 'None'}")
    print(f"  CMC      : {use_cmc}")

    detector = OnnxDetector(str(det_onnx), conf=DET_CONF, iou=NMS_IOU)

    reid_model = None
    if reid_onnx and reid_onnx.exists():
        reid_model = OnnxReIDNet(str(reid_onnx))

    test_seq_dirs = sorted((TEST_BASE / "images").iterdir())
    if not test_seq_dirs:
        print("테스트 시퀀스 없음:", TEST_BASE / "images")
        return

    print(f"  Sequences: {[d.name for d in test_seq_dirs]}")

    all_metrics = []
    all_fps     = []

    for seq_dir in test_seq_dirs:
        n_frames = len(list(seq_dir.glob("*.jpg")))
        print(f"\n  [SEQ] {seq_dir.name} ({n_frames} frames)")

        tracker = BoTSORTTracker(
            reid_net    = reid_model,
            track_thresh= 0.5,
            match_thresh= 0.8,
            track_buffer= 30,
            frame_rate  = 5,
            lam         = 0.5 if reid_model else 1.0,  # ReID 없으면 IoU만 사용
            alpha       = 0.95,
            use_cmc     = use_cmc,
        )

        metrics, fps = evaluate_sequence_botsort(
            seq_dir, detector, tracker, verbose=verbose)
        all_metrics.append(metrics)
        all_fps.append(fps)

        print(f"  MOTA={metrics['MOTA']:.4f}  IDF1={metrics['IDF1']:.4f}  "
              f"HOTA={metrics['HOTA']:.4f}  FPS={fps:.1f}")
        print(f"  GT={metrics['GT']}  FP={metrics['FP']}  "
              f"FN={metrics['FN']}  IDSW={metrics['IDSW']}")

    # 전체 평균
    print(f"\n{'='*60}")
    print(f"BoT-SORT E6 전체 평균 결과")
    print(f"{'='*60}")
    avg = {k: round(sum(m[k] for m in all_metrics) / len(all_metrics), 4)
           for k in ("MOTA", "IDF1", "HOTA", "DetA", "GT", "FP", "FN", "IDSW")}
    avg_fps = sum(all_fps) / len(all_fps)

    print(f"  MOTA:  {avg['MOTA']}")
    print(f"  IDF1:  {avg['IDF1']}")
    print(f"  HOTA:  {avg['HOTA']}")
    print(f"  DetA:  {avg['DetA']}")
    print(f"  FPS:   {avg_fps:.1f}")
    print(f"  GT:    {avg['GT']}  FP: {avg['FP']}  "
          f"FN: {avg['FN']}  IDSW: {avg['IDSW']}")

    # ByteTrack E1과 비교
    print(f"\n[E6 vs E1 ByteTrack 비교]")
    e1 = {"MOTA": 0.221, "IDF1": 0.384, "HOTA": 0.487, "IDSW": 0}
    for k in ("MOTA", "IDF1", "HOTA"):
        d = avg[k] - e1[k]
        print(f"  {k}: E1={e1[k]:.3f} -> E6={avg[k]:.3f} ({d:+.3f})")

    return avg, avg_fps


def main():
    parser = argparse.ArgumentParser(description="E6 BoT-SORT 추적 평가")
    parser.add_argument("--det_onnx",  default=str(DEFAULT_DET_ONNX))
    parser.add_argument("--reid_onnx", default=str(DEFAULT_REID_ONNX))
    parser.add_argument("--no_reid",   action="store_true")
    parser.add_argument("--no_cmc",    action="store_true")
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    det_onnx  = Path(args.det_onnx)
    reid_onnx = None if args.no_reid else Path(args.reid_onnx)

    run_all_sequences(det_onnx, reid_onnx,
                      use_cmc=not args.no_cmc,
                      verbose=not args.quiet)


if __name__ == "__main__":
    main()
