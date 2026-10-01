"""Where does ORT-Web WASM inference spend its time? Breakdown of wasm_kernel_profile.mjs CPU profiles.

Step 1 of the Mac WASM INT8 analysis (docs/ROADMAP.md). The ORT-Web build has no name section, so V8 reports
frames as `wasm-function[N]`; N is matched to the function index space of the same .wasm (wasm_simd_scan.py) and
each function is classed by the SIMD instructions it contains (first matching rule wins):

  int8_dot_gemm   uses i32x4.dot_i16x8_s (the MLAS U8X8 QGEMM of this build; a single, fully inlined function)
  quant_convert   uses int<->float conversion or narrowing (i32x4.trunc_sat_f32x4_s, f32x4.convert_i32x4_s,
                  i8x16/i16x8.narrow_*): QuantizeLinear, DequantizeLinear and requantization of INT8 outputs
  fp32_simd_mac   uses f32x4.mul and f32x4.add (SGEMM/convolution and other float SIMD arithmetic)
  other_simd      any other SIMD function
  scalar_wasm     wasm without SIMD
  non_wasm        JavaScript, garbage collection, idle and V8 bookkeeping

Self time per profile node comes from the sample time deltas. The report gives per-class ms per inference for
each run, the median over rounds, and the INT8 - FP32 difference per class.

Usage: python scripts/paper/wasm_kernel_breakdown.py --wasm <ort-wasm-simd-threaded.asyncify.wasm>
           --profiles <dir> [<dir> ...] --output <summary.json>
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.wasm_simd_scan import scan

CLASSES = ["int8_dot_gemm", "quant_convert", "fp32_simd_mac", "other_simd", "scalar_wasm", "non_wasm"]
CONVERT = (0xF8, 0xFA, 0x65, 0x66, 0x85, 0x86)
WASM_FRAME = re.compile(r"^wasm-function\[(\d+)\]$")


def classify(simd) -> str:
    if simd.get(0xBA):
        return "int8_dot_gemm"
    if any(simd.get(k) for k in CONVERT):
        return "quant_convert"
    if simd.get(0xE6) and simd.get(0xE4):
        return "fp32_simd_mac"
    return "other_simd" if simd else "scalar_wasm"


def self_times(profile: dict) -> dict[int, float]:
    """Self time (us) per node id; sample i lasts until sample i + 1 (the last one until endTime)."""
    samples, deltas = profile["samples"], profile["timeDeltas"]
    stamps, t = [], profile["startTime"]
    for d in deltas:
        t += d
        stamps.append(t)
    out: dict[int, float] = defaultdict(float)
    for i, node in enumerate(samples):
        end = stamps[i + 1] if i + 1 < len(stamps) else profile["endTime"]
        out[node] += max(0.0, end - stamps[i])
    return out


def breakdown(profile: dict, classes: dict[int, str], iterations: int) -> tuple[dict, list]:
    nodes = {n["id"]: n["callFrame"] for n in profile["nodes"]}
    per_class: dict[str, float] = defaultdict(float)
    per_func: dict[int, float] = defaultdict(float)
    for node_id, us in self_times(profile).items():
        m = WASM_FRAME.match(nodes[node_id]["functionName"])
        if m:
            index = int(m.group(1))
            per_class[classes.get(index, "scalar_wasm")] += us
            per_func[index] += us
        else:
            per_class["non_wasm"] += us
    ms = {c: per_class.get(c, 0.0) / 1000 / iterations for c in CLASSES}
    top = sorted(per_func.items(), key=lambda kv: -kv[1])[:8]
    return ms, [{"function": f"wasm-function[{i}]", "class": classes.get(i, "scalar_wasm"),
                 "ms_per_inference": us / 1000 / iterations} for i, us in top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wasm", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True, help="wasm_kernel_profile.mjs output dirs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    functions, bad = scan(args.wasm.read_bytes())
    if bad:
        raise SystemExit(f"{bad} function bodies failed the decoder self-check")
    classes = {f.index: classify(f.simd) for f in functions}
    class_sizes = {c: sum(1 for v in classes.values() if v == c) for c in CLASSES[:-1]}

    runs, grouped = [], defaultdict(list)
    for folder in args.profiles:
        for meta_path in sorted(folder.glob("profile_*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") != "completed":
                continue
            profile = json.loads(gzip.decompress(meta_path.with_suffix(".cpuprofile.gz").read_bytes()))
            ms, top = breakdown(profile, classes, meta["iterations"])
            device = folder.name
            runs.append({"device": device, "run": meta["run"], "model": meta["model"], "browser": meta.get("browser"),
                         "inference_mean_ms": meta["inference_ms"]["mean_ms"], "profiled_ms_per_inference": sum(ms.values()),
                         "class_ms_per_inference": ms, "top_functions": top})
            grouped[device, meta["model"]].append(runs[-1])

    medians = {}
    for (device, model), rs in sorted(grouped.items()):
        medians.setdefault(device, {})[model] = {
            "rounds": len(rs),
            "inference_mean_ms": statistics.median(r["inference_mean_ms"] for r in rs),
            "class_ms_per_inference": {c: statistics.median(r["class_ms_per_inference"][c] for r in rs) for c in CLASSES},
        }
    differences = {}
    for device, models in medians.items():
        for fp32 in [m for m in models if m.endswith("_fp32")]:
            int8 = fp32.replace("_fp32", "_int8_head_excl")
            if int8 in models:
                a, b = models[fp32], models[int8]
                differences.setdefault(device, {})[f"{int8} - {fp32}"] = {
                    "inference_mean_ms": b["inference_mean_ms"] - a["inference_mean_ms"],
                    "fp32_over_int8": a["inference_mean_ms"] / b["inference_mean_ms"],
                    "class_ms_per_inference": {c: b["class_ms_per_inference"][c] - a["class_ms_per_inference"][c]
                                               for c in CLASSES}}

    result = {"wasm": args.wasm.name, "decoder_mismatches": bad, "functions_per_class": class_sizes,
              "medians": medians, "int8_minus_fp32": differences, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for device, diffs in differences.items():
        for name, d in diffs.items():
            cls = ", ".join(f"{c} {v:+.1f}" for c, v in d["class_ms_per_inference"].items())
            print(f"{device} {name}: {d['inference_mean_ms']:+.1f} ms (FP32/INT8 {d['fp32_over_int8']:.2f}); {cls}")


if __name__ == "__main__":
    main()
