"""Audit independent-test object annotations for manual cross-frame identity labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

IDENTITY_KEYS = {"track_id", "object_id", "instance_id", "identity_id", "person_id", "vehicle_id"}


def audit(artifact_root: Path, manifest: Path, output: Path) -> dict:
    root = artifact_root.resolve(strict=True)
    frames = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    annotation_keys = Counter()
    identity_counts = Counter()
    object_count = 0
    for frame in frames:
        data = json.loads((root / frame["label_rel"]).read_text(encoding="utf-8"))
        for ann in data.get("annotation", []):
            if ann.get("class") not in ("traffic_sign", "traffic_light"):
                continue
            object_count += 1
            annotation_keys.update(ann.keys())
            for key in IDENTITY_KEYS & ann.keys():
                if ann[key] not in (None, ""):
                    identity_counts[key] += 1
    result = {
        "test_frames": len(frames),
        "test_objects": object_count,
        "manual_identity_key_nonempty_counts": dict(identity_counts),
        "annotation_key_counts": dict(sorted(annotation_keys.items())),
        "manual_cross_frame_identity_ground_truth_available": bool(identity_counts),
        "decision": "exclude MOTA/IDF1/HOTA from main quantitative claims"
        if not identity_counts
        else "inspect identity continuity before MOT evaluation",
        "historical_tracking_results_status": "pseudo-GT only; not manual identity ground truth",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper_evidence/tracking/annotation_audit.json")
    )
    args = parser.parse_args()
    print(audit(args.artifact_root, args.manifest, args.output)["decision"])


if __name__ == "__main__":
    main()
