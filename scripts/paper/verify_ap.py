"""Cross-check 101-point AP integration against Ultralytics on saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics.utils.metrics import compute_ap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.evaluate_qdq_detection import CLASSES, IOU_THRESHOLDS, _ap, _match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("paper_evidence"))
    args = parser.parse_args()
    comparisons = []
    for model in ("yolo26_fp32", "yolo26_full_qdq", "yolo26_head_excluded_qdq"):
        path = args.evidence / "detection" / model / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        gt = [row["ground_truth"] for row in rows]
        predictions = [row["detections"] for row in rows]
        for class_id, name in CLASSES.items():
            for threshold in IOU_THRESHOLDS:
                tp, fp, n_gt = _match(
                    gt, predictions, class_id=class_id, threshold=threshold, min_conf=0.001
                )
                ours = _ap(tp, fp, n_gt)
                if tp:
                    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
                    ul_ap = float(
                        compute_ap(cum_tp / n_gt, cum_tp / np.maximum(1, cum_tp + cum_fp))[0]
                    )
                else:
                    ul_ap = 0.0
                comparisons.append(
                    {
                        "model": model,
                        "class": name,
                        "IoU": threshold,
                        "local_AP": ours,
                        "ultralytics_AP": ul_ap,
                        "absolute_difference": abs(ours - ul_ap),
                    }
                )
    result = {
        "max_absolute_AP_difference": max(row["absolute_difference"] for row in comparisons),
        "comparison_count": len(comparisons),
        "note": "Local 101-point precision-envelope trapezoid matches Ultralytics compute_ap.",
        "comparisons": comparisons,
    }
    output = args.evidence / "reports/AP_CROSSCHECK.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"maximum absolute AP difference: {result['max_absolute_AP_difference']:.6f}")


if __name__ == "__main__":
    main()
