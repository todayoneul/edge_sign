"""Recompute final AP from saved frame predictions without running inference again."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.evaluate_qdq_detection import VARIANTS, compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("paper_evidence"))
    args = parser.parse_args()
    for name in VARIANTS:
        folder = args.evidence / "detection" / name
        metrics_path = folder / "metrics.json"
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        backup = folder / "metrics_initial_101mean.json"
        if backup.exists():
            raise FileExistsError(f"refusing to replace initial metrics backup: {backup}")
        rows = [
            json.loads(line)
            for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        result = compute_metrics(
            [row["ground_truth"] for row in rows], [row["detections"] for row in rows]
        )
        result.update(
            frames_evaluated=len(rows),
            dropped_invalid_predictions=previous["dropped_invalid_predictions"],
            complete=True,
            recomputed_from_saved_predictions=True,
        )
        backup.write_text(
            json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: mAP50={result['mAP50']:.6f} mAP50_95={result['mAP50_95']:.6f}")


if __name__ == "__main__":
    main()
