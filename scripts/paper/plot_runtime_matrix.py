"""Figures for the component x precision x runtime matrix (reads summarize_runtime_matrix.py output).

  fig6_component_sensitivity   accuracy retention per component and precision, 95% and 99% lines (paper Fig. 2)
  fig7_runtime_latency         batch-1 latency per variant x runtime on one device (log scale)
  fig8_pipeline_assignment     browser detector+recognizer pipeline stages on one device
  with --compare <second device matrix>:
  fig11_runtime_latency_devices  detector latency, representative conditions, one panel per device (paper Fig. 4)
  fig12_pipeline_devices         key pipeline placements, device A and B bars per placement (paper Fig. 5)

Every figure is written as <name>.pdf (vector, for the manuscript) and <name>.png (600 dpi) in the
shared paper style (scripts/paper/paper_style.py).

Usage: python scripts/paper/plot_runtime_matrix.py --matrix paper_evidence/runtime/matrix
       [--compare paper_evidence/runtime/matrix_mac]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.paper_style import (
    FULL_WIDTH,
    MODEL_COLOR,
    OKABE_ITO,
    RUNTIME_COLOR,
    STAGE_COLOR,
    apply,
    panel_label,
    save,
    value_grid,
)

MODEL_NAME = {"v4": "YOLO26-n", "v3": "YOLOv8s", "rec": "KoreanSignNet", "coco": "YOLO11l"}
SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_full": "INT8 full", "int8_full_fbias": "INT8 full (FP32 bias)",
         "int8_head_excl": "INT8 head-excl.", "int8_head_excl_fbias": "INT8 head-excl. (FP32 bias)"}
RUNTIMES = [("cpu_t1", "ORT CPU 1T", OKABE_ITO["gray"], 0.55), ("cpu_t4", "ORT CPU 4T", OKABE_ITO["gray"], 1.0),
            ("wasm_1300", "WASM 1T", OKABE_ITO["orange"], 0.55), ("wasmt4_1300", "WASM 4T", OKABE_ITO["orange"], 1.0),
            ("webgpu_1220", "WebGPU (ORT-Web 1.22)", OKABE_ITO["sky"], 1.0),
            ("webgpu_1300", "WebGPU (ORT-Web 1.30)", OKABE_ITO["blue"], 1.0)]
LAT_ROWS = ["v4_fp32", "v4_fp16", "v4_int8_head_excl", "v4_int8_head_excl_fbias",
            "v3_fp32", "v3_fp16", "v3_int8_head_excl", "v3_int8_head_excl_fbias",
            "rec_fp32", "rec_fp16", "rec_int8_full", "rec_int8_full_fbias"]
SENS_ROWS = ["v4_fp16", "v4_int8_full", "v4_int8_head_excl", "v3_fp16", "v3_int8_full", "v3_int8_head_excl",
             "rec_fp16", "rec_int8_full", "rec_int8_head_excl"]
FPS_LINES = [(1000 / 30, "--", OKABE_ITO["black"], "30 FPS (33.3 ms)"), (1000 / 15, ":", OKABE_ITO["black"], "15 FPS (66.7 ms)")]


def label(key: str) -> str:
    family, rest = key.split("_", 1)
    return f"{MODEL_NAME[family]} {SHORT.get(rest, rest)}"


def fig_sensitivity(rows: list[dict], bootstrap: dict, out: Path) -> None:
    """Retention per component and variant, zoomed on 94-101% so the 95% and 99% lines are readable."""
    by = {r["model"]: r for r in rows}
    keys = [k for k in SENS_ROWS if by.get(k, {}).get("retention") is not None]
    lo_x, hi_x = 94.0, 101.2
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.9))
    y = np.arange(len(keys))[::-1]
    for yi, key in zip(y, keys, strict=True):
        r = by[key]
        color = MODEL_COLOR[r["family"]]
        value = r["retention"] * 100
        if value < lo_x:  # collapsed: hatched stub and a direct label
            ax.barh(yi, 0.5, left=lo_x, height=0.62, color="white", edgecolor=color, hatch="////", linewidth=0.6)
            ax.text(lo_x + 0.62, yi, f"{value:.0f}% (no detections)", va="center", color=color)
            continue
        ax.barh(yi, value - lo_x, left=lo_x, height=0.62, color=color)
        ci = bootstrap.get(key)
        right = value
        if ci:
            ax.errorbar(value, yi, xerr=[[value - ci["ci95_low"] * 100], [ci["ci95_high"] * 100 - value]],
                        fmt="none", ecolor=OKABE_ITO["black"], elinewidth=0.7, capsize=1.8)
            right = ci["ci95_high"] * 100
        ax.text(min(right, hi_x - 0.45) + 0.08, yi, f"{value:.1f}", va="center")
    ax.set_yticks(y, [f"{MODEL_NAME[by[k]['family']]} {SHORT[k.split('_', 1)[1]]} ({by[k]['bytes'] / 1e6:.2f} MB)" for k in keys])
    ax.axvline(99, color=OKABE_ITO["black"], ls="--", lw=0.8)
    ax.axvline(95, color=OKABE_ITO["gray"], ls=":", lw=0.8)
    ax.text(99.05, len(keys) - 0.45, "99% criterion", va="bottom")
    ax.text(95.05, len(keys) - 0.45, "95% tier (reference)", va="bottom", color="#555555")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.6, len(keys) + 0.1)
    ax.set_xlabel("Retention vs FP32 (%); whisker = 95% bootstrap CI")
    value_grid(ax, "x")
    save(fig, out)
    plt.close(fig)


def fig_latency(rows: list[dict], out: Path) -> None:
    by = {r["model"]: r for r in rows}
    keys = [k for k in LAT_ROWS if k in by]
    runtimes = [rt for rt in RUNTIMES if any(by[k].get(rt[0]) for k in keys)]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.9))
    width = 0.84 / max(1, len(runtimes))
    x = np.arange(len(keys))
    for i, (col, name, color, alpha) in enumerate(runtimes):
        vals = [by[k][col]["mean_ms"] if by[k].get(col) else np.nan for k in keys]
        errs = [max(0.0, by[k][col]["p90_ms"] - by[k][col]["mean_ms"]) if by[k].get(col) else 0 for k in keys]
        ax.bar(x + (i - (len(runtimes) - 1) / 2) * width, vals, width, yerr=[np.zeros(len(keys)), errs],
               color=color, alpha=alpha, label=name, error_kw={"elinewidth": 0.5, "capsize": 1})
    for yv, ls, color, name in FPS_LINES:
        ax.axhline(yv, color=color, ls=ls, lw=0.8, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms), mean; whisker = p90")
    ax.set_xticks(x, [label(k) for k in keys], rotation=35, ha="right")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    value_grid(ax)
    save(fig, out)
    plt.close(fig)


# Browser pipeline placements: (run id without the _r<k> repeat suffix, label). Configurations with
# repeated launches are drawn from the median of the per-launch stage means.
PIPELINE_ROWS = [
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_fp32@wasm", "Det FP16 @WebGPU + Rec FP32 @WASM"),
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_int8_full@wasm", "Det FP16 @WebGPU + Rec INT8 @WASM"),
    ("pipeline_t4_ort1300_v4_fp32@webgpu+rec_fp32@wasm", "Det FP32 @WebGPU + Rec FP32 @WASM"),
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_fp16@webgpu", "Det FP16 @WebGPU + Rec FP16 @WebGPU"),
    ("pipeline_t4_ort1300_v4_fp32@webgpu+rec_fp32@webgpu", "Det FP32 @WebGPU + Rec FP32 @WebGPU"),
    ("pipeline_t4_ort1300_v4_int8_head_excl@wasm+rec_int8_full@wasm", "Det INT8 head-excl. @WASM + Rec INT8 @WASM"),
    ("pipeline_t4_ort1300_v4_fp32@wasm+rec_fp32@wasm", "Det FP32 @WASM + Rec FP32 @WASM"),
    ("pipeline_t1_ort1300_v4_int8_head_excl@wasm+rec_int8_full@wasm", "Det INT8 head-excl. @WASM 1T + Rec INT8 @WASM 1T"),
    ("pipeline_t1_ort1300_v4_fp32@wasm+rec_fp32@wasm", "Det FP32 @WASM 1T + Rec FP32 @WASM 1T"),
    ("pipeline_t4_ort1300_v4_int8_head_excl_fbias@webgpu+rec_int8_full@webgpu", "Det INT8 head-excl. @WebGPU + Rec INT8 @WebGPU"),
]
STAGES = [("det_prepare_ms", "normalize"), ("det_ms", "detector"), ("decode_ms", "decode"),
          ("rec_prepare_ms", "ROI crop"), ("rec_ms", "recognizer")]


def pipeline_launches(matrix: Path, run: str) -> list[dict]:
    """Completed launches of one configuration: r1 (no suffix) and r2..r5."""
    out = []
    for suffix in ["", "_r2", "_r3", "_r4", "_r5"]:
        p = matrix / f"{run}{suffix}.json"
        if p.exists() and (r := json.loads(p.read_text(encoding="utf-8"))).get("status") == "completed":
            # per-frame trace is stored next to the result as <run>_trace.csv
            with open(matrix / f"{run}{suffix}_trace.csv", encoding="utf-8") as f:
                total = np.array([float(row["total_ms"]) for row in csv.DictReader(f)])
            out.append({"metrics": r["metrics"], "total": total})
    return out


def launch_summary(matrix: Path, run: str) -> tuple | None:
    launches = pipeline_launches(matrix, run)
    if not launches:
        return None
    stage = {k: float(np.median([x["metrics"][k]["mean_ms"] for x in launches])) for k, _ in STAGES}
    means = [x["metrics"]["total_ms"]["mean_ms"] for x in launches]
    p90s = [float(np.percentile(x["total"], 90)) for x in launches]
    return stage, float(np.median(means)), min(means), max(means), float(np.median(p90s)), len(launches)


def stacked_bar(ax, y: float, stage: dict, height: float, limit: float, first: bool, edge: bool) -> None:
    left = 0.0
    for key, name in STAGES:
        value = stage[key]
        width = min(left + value, limit) - min(left, limit)
        ax.barh(y, width, height, left=min(left, limit), color=STAGE_COLOR[key],
                edgecolor=OKABE_ITO["black"] if edge else "none", linewidth=0.3, label=name if first else None)
        left += value


def fps_lines(ax) -> None:
    for xv, ls, color, _ in FPS_LINES:
        ax.axvline(xv, color=color, ls=ls, lw=0.8)


def fig_pipeline(matrix: Path, out: Path) -> None:
    rows = [(name, s) for run, name in PIPELINE_ROWS if (s := launch_summary(matrix, run))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.3 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    limit = 110.0
    for k, (yi, (_name, (stage, med, lo, hi, p90, n))) in enumerate(zip(y, rows, strict=True)):
        stacked_bar(ax, yi, stage, 0.62, limit, first=k == 0, edge=False)
        text = f"{med:.1f} ms · p90 {p90:.1f} · n={n}"
        if med > limit:
            ax.text(limit - 1, yi, "off scale: " + text, va="center", ha="right", color="white")
            continue
        if n > 1:
            ax.errorbar(med, yi, xerr=[[med - lo], [hi - med]], fmt="none", ecolor=OKABE_ITO["black"], elinewidth=0.6, capsize=1.5)
        ax.text(max(med, hi) + 1.2, yi, text, va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})
    fps_lines(ax)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, limit)
    ax.set_xlabel("Per-frame latency (ms); bar = median of launch means, whisker = range over launches")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    value_grid(ax, "x")
    save(fig, out)
    plt.close(fig)


# Cross-device figures (paper Fig. 4 and 5): device A = --matrix, device B = --compare.
# Representative conditions only; INT8 is the head-excluded variant with INT32 bias.
DEVICE_CONDS = [  # (legend, runtime column, variant suffix, color, hatch)
    ("ORT CPU 4T FP32", "cpu_t4", "fp32", RUNTIME_COLOR["cpu"], ""),
    ("ORT CPU 4T INT8", "cpu_t4", "int8_head_excl", RUNTIME_COLOR["cpu"], "////"),
    ("WASM 4T FP32", "wasmt4_1300", "fp32", RUNTIME_COLOR["wasm"], ""),
    ("WASM 4T INT8", "wasmt4_1300", "int8_head_excl", RUNTIME_COLOR["wasm"], "////"),
    ("WebGPU FP32", "webgpu_1300", "fp32", RUNTIME_COLOR["webgpu"], ""),
    ("WebGPU FP16", "webgpu_1300", "fp16", RUNTIME_COLOR["webgpu_fp16"], ""),
    ("WebGPU INT8", "webgpu_1300", "int8_head_excl", RUNTIME_COLOR["webgpu"], "////"),
]
DEVICE_MODELS = [("v4", "YOLO26-n"), ("v3", "YOLOv8s"), ("coco", "YOLO11l (COCO)")]
DEVICE_PIPELINE_ROWS = PIPELINE_ROWS[:1] + PIPELINE_ROWS[2:7]  # WebGPU and 4-thread WASM placements


def fig_latency_devices(rows_a: list[dict], rows_b: list[dict], names: tuple[str, str], out: Path) -> None:
    """Batch-1 latency of the three detectors on both devices, one panel per device (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.55), sharey=True)
    width = 0.84 / len(DEVICE_CONDS)
    x = np.arange(len(DEVICE_MODELS))
    for ax, rows, name, tag in zip(axes, (rows_a, rows_b), names, ("(a)", "(b)"), strict=True):
        by = {r["model"]: r for r in rows}
        for i, (legend, col, suffix, color, hatch) in enumerate(DEVICE_CONDS):
            cells = [by.get(f"{fam}_{suffix}", {}).get(col) for fam, _ in DEVICE_MODELS]
            vals = [c["mean_ms"] if c else np.nan for c in cells]
            errs = [max(0.0, c["p90_ms"] - c["mean_ms"]) if c else 0.0 for c in cells]
            ax.bar(x + (i - (len(DEVICE_CONDS) - 1) / 2) * width, vals, width, yerr=[np.zeros(len(cells)), errs],
                   color=color, hatch=hatch, edgecolor="white" if hatch else color, linewidth=0.0, label=legend,
                   error_kw={"elinewidth": 0.5, "capsize": 1})
        for yv, ls, color, fps in FPS_LINES:
            ax.axhline(yv, color=color, ls=ls, lw=0.8, label=fps)
        ax.set_yscale("log")
        ax.set_ylim(5, 4000)
        ax.set_xticks(x, [m for _, m in DEVICE_MODELS])
        panel_label(ax, f"{tag} {name}")
        value_grid(ax)
    axes[0].set_ylabel("Latency (ms), mean; whisker = p90")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(c[0]) for c in DEVICE_CONDS] + [labels.index(f[3]) for f in FPS_LINES]  # bars, then FPS lines
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, out)
    plt.close(fig)


def fig_pipeline_devices(matrix_a: Path, matrix_b: Path, names: tuple[str, str], out: Path) -> None:
    """Per-stage pipeline latency, device A (upper bar) and device B (lower bar) for each placement."""
    groups = [(name, [launch_summary(matrix_a, run), launch_summary(matrix_b, run)]) for run, name in DEVICE_PIPELINE_ROWS]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 0.52 * len(groups) + 0.8))
    height, limit = 0.34, 108.0
    yticks = []
    for g, (_, bars) in enumerate(groups):
        base = len(groups) - 1 - g
        yticks.append(base)
        for d, bar in enumerate(bars):
            if bar is None:
                continue
            stage, med, lo, hi, p90, n = bar
            y = base + (0.19 if d == 0 else -0.19)
            stacked_bar(ax, y, stage, height, limit, first=g == 0 and d == 0, edge=d == 1)
            if n > 1:
                ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="none", ecolor=OKABE_ITO["black"], elinewidth=0.6, capsize=1.5)
            ax.text(max(med, hi) + 1.2, y, f"{'AB'[d]}  {med:.1f} ms · p90 {p90:.1f} · n={n}", va="center", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})
    fps_lines(ax)
    ax.set_yticks(yticks, [name for name, _ in groups])
    ax.set_xlim(0, limit)
    ax.set_xlabel(f"Per-frame latency (ms). A (upper): {names[0].split(':')[0]}; B (lower, outlined): {names[1].split(':')[0]}; "
                  "whisker: range over launches")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=OKABE_ITO["black"], ls="--", lw=0.8), Line2D([], [], color=OKABE_ITO["black"], ls=":", lw=0.8)]
    labels += ["30 FPS", "15 FPS"]
    ax.legend(handles, labels, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    value_grid(ax, "x")
    save(fig, out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--compare", type=Path, default=None,
                        help="second device's matrix (e.g. paper_evidence/runtime/matrix_mac): adds fig11 and fig12")
    parser.add_argument("--names", nargs=2, default=["Windows: Ryzen 5 9600X + RTX 5070", "Mac: Apple M2 Pro"])
    parser.add_argument("--figures", type=Path, default=Path("paper_evidence/figures"))
    args = parser.parse_args()
    apply()
    rows = json.loads((args.matrix / "summary.json").read_text(encoding="utf-8"))["rows"]
    boot = args.matrix / "bootstrap_retention.json"
    pairs = json.loads(boot.read_text(encoding="utf-8"))["pairs"] if boot.exists() else {}
    fig_sensitivity(rows, pairs, args.figures / "fig6_component_sensitivity.png")
    fig_latency(rows, args.figures / "fig7_runtime_latency.png")
    fig_pipeline(args.matrix, args.figures / "fig8_pipeline_assignment.png")
    if args.compare:
        rows_b = json.loads((args.compare / "summary.json").read_text(encoding="utf-8"))["rows"]
        names = (args.names[0], args.names[1])
        fig_latency_devices(rows, rows_b, names, args.figures / "fig11_runtime_latency_devices.png")
        fig_pipeline_devices(args.matrix, args.compare, names, args.figures / "fig12_pipeline_devices.png")
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
