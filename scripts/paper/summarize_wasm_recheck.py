"""Quiet re-check of WASM FP32 vs INT8 detector latency on one device (runtime_matrix.py output).

The first Mac run showed no INT8 speedup on WASM (paper 4.3), measured once per model while
other apps were busy. run_mac_wasm_recheck.sh repeats it: every launch is a fresh headless
Chrome, round 1 is untagged and rounds 2..5 carry a _r<k> suffix, and FP32 / INT8 alternate
their order between rounds. Per model pair and WASM thread count this reports the per-round
FP32/INT8 ratio of the launch means (> 1: INT8 faster) and the median [min-max] over rounds.
The pipeline rounds are summarized by summarize_pipeline_repeats.py.

Usage: python scripts/paper/summarize_wasm_recheck.py --matrix paper_evidence/runtime/matrix_mac_recheck
Output: <matrix>/wasm_recheck_summary.json and a Markdown table on stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PAIRS = [("v4_fp32", "v4_int8_head_excl"), ("v3_fp32", "v3_int8_head_excl")]
THREADS = [("wasmt4", 4), ("wasm", 1)]
ROUNDS = ["", "_r2", "_r3", "_r4", "_r5"]


def launch(matrix: Path, tag: str, key: str, suffix: str) -> dict | None:
    path = matrix / f"speed_{tag}_ort1300_{key}{suffix}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        return None
    m = result["metrics"]["inference_ms"]
    return {"mean_ms": m["mean_ms"], "p90_ms": m["p90_ms"], "threads": result.get("effective_wasm_threads"),
            "isolated": result.get("cross_origin_isolated")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    out: dict = {"pairs": []}
    for tag, threads in THREADS:
        for fp32, int8 in PAIRS:
            rounds = []
            for suffix in ROUNDS:
                a, b = launch(args.matrix, tag, fp32, suffix), launch(args.matrix, tag, int8, suffix)
                if a and b:
                    rounds.append({"round": suffix[1:] or "r1", "fp32": a, "int8": b, "fp32_over_int8": a["mean_ms"] / b["mean_ms"]})
            if not rounds:
                continue
            ratios = [r["fp32_over_int8"] for r in rounds]
            threads_seen = sorted({r[k]["threads"] for r in rounds for k in ("fp32", "int8")}, key=str)
            out["pairs"].append({
                "wasm_threads": threads, "fp32": fp32, "int8": int8, "rounds": rounds,
                "median_fp32_mean_ms": statistics.median(r["fp32"]["mean_ms"] for r in rounds),
                "median_int8_mean_ms": statistics.median(r["int8"]["mean_ms"] for r in rounds),
                "median_ratio": statistics.median(ratios), "min_ratio": min(ratios), "max_ratio": max(ratios),
                "rounds_int8_faster": sum(x > 1 for x in ratios), "effective_threads_seen": threads_seen,
            })
    (args.matrix / "wasm_recheck_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("## WASM FP32 vs INT8 head-excluded, repeated fresh launches\n")
    print("| models | WASM threads (seen) | rounds | FP32 mean: median | INT8 mean: median | "
          "FP32/INT8 per round | median [min-max] | rounds INT8 faster |")
    print("|---|---|---:|---:|---:|---|---:|---:|")
    for p in out["pairs"]:
        per_round = ", ".join(f"{r['fp32_over_int8']:.2f}" for r in p["rounds"])
        print(f"| {p['fp32']} vs {p['int8']} | {p['wasm_threads']} ({'/'.join(map(str, p['effective_threads_seen']))}) "
              f"| {len(p['rounds'])} | {p['median_fp32_mean_ms']:.1f} | {p['median_int8_mean_ms']:.1f} | {per_round} "
              f"| {p['median_ratio']:.2f} [{p['min_ratio']:.2f}-{p['max_ratio']:.2f}] "
              f"| {p['rounds_int8_faster']}/{len(p['rounds'])} |")


if __name__ == "__main__":
    main()
