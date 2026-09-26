"""Run-to-run variation of the browser pipeline (RQ3): five launches per configuration.

Each launch is a fresh headless Chrome with a new profile that processes the first 512
test frames (runtime_matrix.py pipeline). Launch r1 is the untagged run; r2..r5 carry a
_r<k> suffix. Per configuration this reports, at one aggregation level (the launch):
  - the median and range over launches of the per-launch mean and of the per-launch p90;
    the latency criteria are judged on the median launch p90,
  - as a check, the p90 of all launches' frames pooled (5 x 512 = 2,560 >= 1,024, the
    MLPerf single-stream minimum); the verdict must not change,
  - paired differences between configurations inside each launch round.

Usage: python scripts/paper/summarize_pipeline_repeats.py
Output: <matrix>/pipeline_repeats_summary.json and Markdown tables on stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

CONFIGS = [
    "v4_fp32@webgpu+rec_fp32@wasm",
    "v4_fp16@webgpu+rec_fp32@wasm",
    "v4_fp16@webgpu+rec_int8_full@wasm",
    "v4_fp32@webgpu+rec_fp32@webgpu",
    "v4_fp16@webgpu+rec_fp16@webgpu",
    "v4_int8_head_excl@wasm+rec_int8_full@wasm",
    "v4_fp32@wasm+rec_fp32@wasm",
]
ROUNDS = ["", "_r2", "_r3", "_r4", "_r5"]
STAGES = ["det_prepare_ms", "det_ms", "decode_ms", "rec_prepare_ms", "rec_ms", "total_ms"]
# (label, a, b): difference a - b of per-launch mean total, inside each round
PAIRS = [
    ("recognizer on WASM instead of WebGPU (FP32 detector on WebGPU)", CONFIGS[0], CONFIGS[3]),
    ("FP16 instead of FP32 detector on WebGPU (recognizer FP32 on WASM)", CONFIGS[1], CONFIGS[0]),
    ("INT8 instead of FP32 recognizer on WASM (FP16 detector on WebGPU)", CONFIGS[2], CONFIGS[1]),
    ("all FP16 instead of all FP32 on WebGPU", CONFIGS[4], CONFIGS[3]),
    ("INT8 head-excluded instead of FP32 detector on WASM", CONFIGS[5], CONFIGS[6]),
]
P90_A, P90_B = 1000 / 30, 1000 / 15


def launch(matrix: Path, config: str, tag: str) -> dict | None:
    stem = matrix / f"pipeline_t4_ort1300_{config}{tag}"
    result = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")) if stem.with_suffix(".json").exists() else None
    if not result or result.get("status") != "completed":
        return None
    with open(f"{stem}_trace.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trace = {k: np.array([float(r[k]) for r in rows]) for k in STAGES}
    return {"trace": trace, "mean_detections": result["mean_detections"], "fetch_retries": result.get("fetch_retries")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    args = parser.parse_args()
    data = {c: [launch(args.matrix, c, t) for t in ROUNDS] for c in CONFIGS}
    out: dict = {"rounds": ["r1"] + [t[1:] for t in ROUNDS[1:]], "frames_per_launch": 512, "configs": {}, "pairs": []}
    for c, runs in data.items():
        ok = [r for r in runs if r]
        means = {s: [float(r["trace"][s].mean()) for r in ok] for s in STAGES}
        pooled = np.concatenate([r["trace"]["total_ms"] for r in ok])
        p90s = [float(np.percentile(r["trace"]["total_ms"], 90)) for r in ok]
        p90 = float(np.percentile(pooled, 90))
        out["configs"][c] = {
            "launches": len(ok),
            "mean_total_per_launch": [None if r is None else float(r["trace"]["total_ms"].mean()) for r in runs],
            "p90_total_per_launch": [None if r is None else float(np.percentile(r["trace"]["total_ms"], 90)) for r in runs],
            "median_of_means": {s: statistics.median(v) for s, v in means.items()},
            "min_mean_total": min(means["total_ms"]), "max_mean_total": max(means["total_ms"]),
            "pooled_frames": int(pooled.size), "pooled_mean_total": float(pooled.mean()), "pooled_p90_total": p90,
            "median_p90_launch": statistics.median(p90s), "min_p90_launch": min(p90s), "max_p90_launch": max(p90s),
            "A_30fps": statistics.median(p90s) <= P90_A, "B_15fps": statistics.median(p90s) <= P90_B,
            "pooled_A_30fps": p90 <= P90_A, "pooled_B_15fps": p90 <= P90_B,
            "launches_passing_A": sum(x <= P90_A for x in p90s), "launches_passing_B": sum(x <= P90_B for x in p90s),
            "mean_detections": statistics.median(r["mean_detections"] for r in ok),
        }
    for label, a, b in PAIRS:
        diffs = [float(ra["trace"]["total_ms"].mean() - rb["trace"]["total_ms"].mean())
                 for ra, rb in zip(data[a], data[b], strict=True) if ra and rb]
        out["pairs"].append({"label": label, "a": a, "b": b, "rounds": len(diffs), "diff_ms": diffs,
                             "mean_diff_ms": statistics.mean(diffs), "min_diff_ms": min(diffs), "max_diff_ms": max(diffs),
                             "rounds_a_faster": sum(d < 0 for d in diffs)})
    (args.matrix / "pipeline_repeats_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("| configuration | launches | det | rec | mean: median [min-max] | p90: median [min-max] | "
          "A 30 FPS | B 15 FPS | pooled p90 (check) |")
    print("|---|---:|---:|---:|---:|---:|:-:|:-:|---:|")
    for c, s in out["configs"].items():
        m = s["median_of_means"]
        same = s["A_30fps"] == s["pooled_A_30fps"] and s["B_15fps"] == s["pooled_B_15fps"]
        print(f"| {c} | {s['launches']} | {m['det_ms']:.2f} | {m['rec_ms']:.2f} | {m['total_ms']:.2f} "
              f"[{s['min_mean_total']:.2f}-{s['max_mean_total']:.2f}] | {s['median_p90_launch']:.2f} "
              f"[{s['min_p90_launch']:.2f}-{s['max_p90_launch']:.2f}] | {'Y' if s['A_30fps'] else 'N'} "
              f"({s['launches_passing_A']}/{s['launches']}) | {'Y' if s['B_15fps'] else 'N'} "
              f"({s['launches_passing_B']}/{s['launches']}) | {s['pooled_p90_total']:.2f} "
              f"({'same verdict' if same else 'VERDICT DIFFERS'}) |")
    print("\n| paired comparison (a - b, per round) | rounds | mean diff ms | [min, max] | rounds a faster |")
    print("|---|---:|---:|---:|---:|")
    for p in out["pairs"]:
        print(f"| {p['label']} | {p['rounds']} | {p['mean_diff_ms']:+.2f} | [{p['min_diff_ms']:+.2f}, "
              f"{p['max_diff_ms']:+.2f}] | {p['rounds_a_faster']}/{p['rounds']} |")


if __name__ == "__main__":
    main()
