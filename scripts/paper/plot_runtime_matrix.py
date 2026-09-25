"""Figures for the component x precision x runtime matrix (reads summarize_runtime_matrix.py output).

  fig6_component_sensitivity.png  accuracy retention vs file size, per component, 99% line
  fig7_runtime_latency.png        batch-1 latency per variant x runtime (log scale), 30/15 FPS lines
  fig8_pipeline_assignment.png    browser detector+recognizer pipeline, per-stage mean ms per config

Usage: python scripts/paper/plot_runtime_matrix.py --matrix paper_evidence/runtime/matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = {"v4": ("YOLO26-n detector", "#1f77b4", "o"), "v3": ("YOLOv8s detector", "#d62728", "s"),
          "rec": ("KoreanSignNet recognizer", "#2ca02c", "^")}
SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_full": "INT8 full", "int8_full_fbias": "INT8 full (fp32 bias)",
         "int8_head_excl": "INT8 head-excl.", "int8_head_excl_fbias": "INT8 head-excl. (fp32 bias)"}
RUNTIMES = [("cpu_t1", "ORT CPU 1T"), ("cpu_t4", "ORT CPU 4T"), ("wasm_1300", "WASM 1T"),
            ("wasmt4_1300", "WASM 4T"), ("webgpu_1220", "WebGPU (ORT-Web 1.22)"), ("webgpu_1300", "WebGPU (ORT-Web 1.30)")]
LAT_ROWS = ["v4_fp32", "v4_fp16", "v4_int8_head_excl", "v4_int8_head_excl_fbias",
            "v3_fp32", "v3_fp16", "v3_int8_head_excl", "v3_int8_head_excl_fbias",
            "rec_fp32", "rec_fp16", "rec_int8_full", "rec_int8_full_fbias"]


def label(key: str) -> str:
    family, rest = key.split("_", 1)
    return f"{family} {SHORT.get(rest, rest)}"


SENS_ROWS = ["v4_fp16", "v4_int8_full", "v4_int8_head_excl", "v3_fp16", "v3_int8_full", "v3_int8_head_excl",
             "rec_fp16", "rec_int8_full", "rec_int8_head_excl"]


def fig_sensitivity(rows: list[dict], bootstrap: dict, out: Path) -> None:
    """Retention per component and variant, zoomed on 94-101% so the 99% line is readable."""
    by = {r["model"]: r for r in rows}
    keys = [k for k in SENS_ROWS if by.get(k, {}).get("retention") is not None]
    lo_x = 94.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(keys))[::-1]
    for yi, key in zip(y, keys):
        r = by[key]
        name, color, _ = FAMILY[r["family"]]
        value = r["retention"] * 100
        if value < lo_x:  # collapsed: draw a hatched stub and say so
            ax.barh(yi, 0.6, left=lo_x, color="white", edgecolor=color, hatch="////")
            ax.text(lo_x + 0.75, yi, f"{value:.0f}% (no detections)", va="center", fontsize=8, color=color)
        else:
            ax.barh(yi, value - lo_x, left=lo_x, color=color, alpha=0.85)
            ci = bootstrap.get(key)
            if ci:
                ax.errorbar(value, yi, xerr=[[value - ci["ci95_low"] * 100], [ci["ci95_high"] * 100 - value]],
                            fmt="none", ecolor="black", capsize=3, lw=1)
            ax.text(min(value, 100.3) + 0.12, yi + 0.28, f"{value:.1f}", fontsize=7.5)
    labels = [f"{FAMILY[by[k]['family']][0].split()[0]} {SHORT[k.split('_', 1)[1]]}  ({by[k]['bytes'] / 1e6:.2f} MB)"
              for k in keys]
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(99, color="black", ls="--", lw=1)
    ax.text(99.05, y[-1] - 0.62, "99% of FP32", fontsize=7.5)
    ax.set_xlim(lo_x, 101)
    ax.set_ylim(-0.7, len(keys) - 0.1)
    ax.set_xlabel("Retention vs FP32 (%); whisker = 95% bootstrap CI", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_latency(rows: list[dict], out: Path) -> None:
    by = {r["model"]: r for r in rows}
    keys = [k for k in LAT_ROWS if k in by]
    runtimes = [rt for rt in RUNTIMES if any(by[k].get(rt[0]) for k in keys)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.8 / max(1, len(runtimes))
    x = np.arange(len(keys))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(runtimes)))
    for i, (col, name) in enumerate(runtimes):
        vals = [by[k][col]["mean_ms"] if by[k].get(col) else np.nan for k in keys]
        errs = [max(0.0, by[k][col]["p90_ms"] - by[k][col]["mean_ms"]) if by[k].get(col) else 0 for k in keys]
        ax.bar(x + (i - (len(runtimes) - 1) / 2) * width, vals, width, yerr=[np.zeros(len(keys)), errs],
               color=colors[i], label=name, capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(1000 / 30, color="black", ls="--", lw=1, label="33.3 ms (30 FPS)")
    ax.axhline(1000 / 15, color="gray", ls=":", lw=1, label="66.7 ms (15 FPS)")
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log)\nbar = mean, whisker = p90")
    ax.set_xticks(x, [label(k) for k in keys], rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_pipeline(matrix: Path, out: Path) -> None:
    runs = []
    for p in sorted(matrix.glob("pipeline_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "completed":
            continue
        d, c = r["detector"], r["recognizer"]
        threads = r.get("effective_wasm_threads", 1)
        name = (f"det {SHORT.get(d['key'][3:], d['key'])} @{d['ep']}\n+ rec {SHORT.get(c['key'][4:], c['key'])} @{c['ep']}"
                + (f" (WASM {threads}T)" if "wasm" in (d["ep"], c["ep"]) else ""))
        runs.append((r["metrics"]["total_ms"]["mean_ms"], name, r["metrics"]))
    if not runs:
        return
    runs.sort()
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(runs) + 1.4))
    stages = [("det_prepare_ms", "normalize", "#bbbbbb"), ("det_ms", "detector", "#1f77b4"),
              ("decode_ms", "decode", "#9467bd"), ("rec_prepare_ms", "ROI crop", "#ff7f0e"), ("rec_ms", "recognizer", "#2ca02c")]
    y = np.arange(len(runs))
    left = np.zeros(len(runs))
    limit = 1000 / 15 * 1.5  # 100 ms; slower configs are clipped and labelled
    for key, name, color in stages:
        vals = np.array([m[key]["mean_ms"] for _, _, m in runs])
        ax.barh(y, np.clip(left + vals, None, limit) - np.clip(left, None, limit), left=np.clip(left, None, limit),
                color=color, label=name)
        left += vals
    for i, (_, _, m) in enumerate(runs):
        text = f"{m['total_ms']['mean_ms']:.1f} ms (p90 {m['total_ms']['p90_ms']:.1f})"
        if left[i] > limit:
            ax.text(limit * 0.98, i, "off scale: " + text, va="center", ha="right", fontsize=7, color="white")
        else:
            ax.text(left[i] + 1, i, text, va="center", fontsize=7)
    ax.axvline(1000 / 30, color="black", ls="--", lw=1)
    ax.axvline(1000 / 15, color="gray", ls=":", lw=1)
    ax.set_yticks(y, [n for _, n, _ in runs], fontsize=7)
    ax.set_xlim(0, limit)
    ax.set_xlabel("Per-frame latency (ms, mean); dashed = 33.3 ms (30 FPS), dotted = 66.7 ms (15 FPS)")
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--figures", type=Path, default=Path("paper_evidence/figures"))
    args = parser.parse_args()
    rows = json.loads((args.matrix / "summary.json").read_text(encoding="utf-8"))["rows"]
    args.figures.mkdir(parents=True, exist_ok=True)
    boot = args.matrix / "bootstrap_retention.json"
    pairs = json.loads(boot.read_text(encoding="utf-8"))["pairs"] if boot.exists() else {}
    fig_sensitivity(rows, pairs, args.figures / "fig6_component_sensitivity.png")
    fig_latency(rows, args.figures / "fig7_runtime_latency.png")
    fig_pipeline(args.matrix, args.figures / "fig8_pipeline_assignment.png")
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
