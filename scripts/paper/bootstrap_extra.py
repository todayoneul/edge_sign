"""Retention CIs for the additional experiments, with the paper's bootstrap (bootstrap_retention.py).

Pairs: the paper's FP32 run (paper_evidence/runtime/matrix/cpu_<v3|v4>_fp32) against every INT8
model in paper_evidence/extra/{calibration,mitigation}/cpu_*/. Same frame bootstrap and 25-frame
moving-block bootstrap (within sequences) as Table 4, same seeds; each pair uses its own seeded
stream so adding a pair never changes another pair's interval.

Usage: python scripts/paper/bootstrap_extra.py --resamples 1000
Output: paper_evidence/extra/<folder>/bootstrap_retention.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.bootstrap_retention import Flags, block_counts, ci
from scripts.paper.evaluate_qdq_detection import compute_metrics

MATRIX = Path("paper_evidence/runtime/matrix")
FOLDERS = [Path("paper_evidence/extra/calibration"), Path("paper_evidence/extra/mitigation")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=25)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    args = parser.parse_args()
    sequences = [json.loads(line)["sequence"] for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    refs = {}
    for folder in FOLDERS:
        results = {"resamples": args.resamples, "seed": args.seed, "block_frames": args.block,
                   "note": "frame bootstrap is optimistic (frames within a sequence are correlated); see block", "pairs": {}}
        for variant in sorted(p for p in folder.glob("cpu_*") if "int8" in p.name and (p / "metrics.json").exists()):
            family = variant.name.split("_")[1]
            if family not in refs:
                refs[family] = Flags(MATRIX / f"cpu_{family}_fp32")
            ref, cand = refs[family], Flags(variant)
            ones = np.ones(len(ref.gt), dtype=np.int64)
            exact = compute_metrics(cand.gt, cand.pred)["mAP50_95"]
            if abs(exact - cand.map50_95(ones)) > 1e-12:
                raise SystemExit(f"{variant.name}: fast mAP differs from compute_metrics")
            point = cand.map50_95(ones) / ref.map50_95(ones)
            stream = zlib.crc32(variant.name.encode())  # stable per-pair stream
            rng, block_rng = np.random.default_rng([args.seed, stream]), np.random.default_rng([args.seed + 1, stream])
            n = len(ref.gt)
            frame = [cand.map50_95(c) / ref.map50_95(c) for c in (np.bincount(rng.integers(0, n, n), minlength=n)
                                                                   for _ in range(args.resamples))]
            block = [cand.map50_95(c) / ref.map50_95(c) for c in (block_counts(sequences, args.block, block_rng)
                                                                   for _ in range(args.resamples))]
            results["pairs"][variant.name[4:]] = {"reference": f"{family}_fp32", "retention": point,
                                                  "frame": ci(np.array(frame)), "block": ci(np.array(block))}
            r = results["pairs"][variant.name[4:]]
            print(f"{variant.name[4:]}: {point * 100:.2f}% frame [{r['frame']['ci95_low'] * 100:.1f}, {r['frame']['ci95_high'] * 100:.1f}] "
                  f"block [{r['block']['ci95_low'] * 100:.1f}, {r['block']['ci95_high'] * 100:.1f}]", flush=True)
        (folder / "bootstrap_retention.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
