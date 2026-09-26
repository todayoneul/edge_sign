"""Figures for the component x precision x runtime matrix (reads summarize_runtime_matrix.py output).

  fig6_component_sensitivity.png  accuracy retention vs file size, per component, 95% and 99% tier lines
  fig7_runtime_latency.png        batch-1 latency per variant x runtime (log scale), 30/15 FPS lines
  fig8_pipeline_assignment.png    browser detector+recognizer pipeline, per-stage ms per config
                                  (median over repeated launches, range whisker, pooled p90)
  with --compare <second device matrix>:
  fig11_runtime_latency_devices.png  detector latency, representative conditions, one panel per device
  fig12_pipeline_devices.png         key pipeline placements, device A and B bars per placement

Usage: python scripts/paper/plot_runtime_matrix.py --matrix paper_evidence/runtime/matrix
       [--compare paper_evidence/runtime/matrix_mac]
"""

from __future__ import annotations

import argparse
import csv
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
    """Retention per component and variant, zoomed on 94-101% so the 95% and 99% lines are readable."""
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
    ax.text(99.05, y[-1] - 0.62, "99% criterion", fontsize=7.5)
    ax.axvline(95, color="gray", ls=":", lw=1)
    ax.text(95.05, y[-1] - 0.62, "95% tier (ref.)", fontsize=7.5, color="dimgray")
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


# Table 8 rows: (run id without the _r<k> repeat suffix, label). Configurations with
# repeated launches are drawn from the median of the per-launch stage means.
PIPELINE_ROWS = [
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_fp32@wasm", "det FP16 @WebGPU + rec FP32 @WASM"),
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_int8_full@wasm", "det FP16 @WebGPU + rec INT8 @WASM"),
    ("pipeline_t4_ort1300_v4_fp32@webgpu+rec_fp32@wasm", "det FP32 @WebGPU + rec FP32 @WASM"),
    ("pipeline_t4_ort1300_v4_fp16@webgpu+rec_fp16@webgpu", "det FP16 @WebGPU + rec FP16 @WebGPU"),
    ("pipeline_t4_ort1300_v4_fp32@webgpu+rec_fp32@webgpu", "det FP32 @WebGPU + rec FP32 @WebGPU"),
    ("pipeline_t4_ort1300_v4_int8_head_excl@wasm+rec_int8_full@wasm", "det INT8 head-excl. @WASM 4T + rec INT8 @WASM 4T"),
    ("pipeline_t4_ort1300_v4_fp32@wasm+rec_fp32@wasm", "det FP32 @WASM 4T + rec FP32 @WASM 4T"),
    ("pipeline_t1_ort1300_v4_int8_head_excl@wasm+rec_int8_full@wasm", "det INT8 head-excl. @WASM 1T + rec INT8 @WASM 1T"),
    ("pipeline_t1_ort1300_v4_fp32@wasm+rec_fp32@wasm", "det FP32 @WASM 1T + rec FP32 @WASM 1T"),
    ("pipeline_t4_ort1300_v4_int8_head_excl_fbias@webgpu+rec_int8_full@webgpu", "det INT8 head-excl. @WebGPU + rec INT8 @WebGPU"),
]
STAGES = [("det_prepare_ms", "normalize", "#bbbbbb"), ("det_ms", "detector", "#1f77b4"),
          ("decode_ms", "decode", "#9467bd"), ("rec_prepare_ms", "ROI crop", "#ff7f0e"), ("rec_ms", "recognizer", "#2ca02c")]


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


def fig_pipeline(matrix: Path, out: Path) -> None:
    rows = []
    for run, name in PIPELINE_ROWS:
        launches = pipeline_launches(matrix, run)
        if not launches:
            continue
        stage = {k: float(np.median([x["metrics"][k]["mean_ms"] for x in launches])) for k, _, _ in STAGES}
        means = [x["metrics"]["total_ms"]["mean_ms"] for x in launches]
        p90s = [float(np.percentile(x["total"], 90)) for x in launches]
        rows.append((name, stage, float(np.median(means)), min(means), max(means), float(np.median(p90s)), len(launches)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    limit = 1000 / 15 * 1.5  # 100 ms; slower configs are clipped and labelled
    for key, label_, color in STAGES:
        vals = np.array([r[1][key] for r in rows])
        ax.barh(y, np.clip(left + vals, None, limit) - np.clip(left, None, limit), left=np.clip(left, None, limit),
                color=color, label=label_)
        left += vals
    for yi, (_name, _, med, lo, hi, p90, n) in zip(y, rows, strict=True):
        spread = f", range {lo:.1f}-{hi:.1f}, n={n}" if n > 1 else ", n=1"
        text = f"{med:.1f} ms (p90 {p90:.1f}{spread})"
        if med > limit:
            ax.text(limit * 0.98, yi, "off scale: " + text, va="center", ha="right", fontsize=7, color="white")
        else:
            if n > 1:
                ax.errorbar(med, yi, xerr=[[med - lo], [hi - med]], fmt="none", ecolor="black", capsize=2, lw=0.8)
            ax.text(max(med, hi) + 1, yi, text, va="center", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})
    ax.axvline(1000 / 30, color="black", ls="--", lw=1)
    ax.axvline(1000 / 15, color="gray", ls=":", lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, limit)
    ax.set_xlabel("Per-frame latency (ms): median over browser launches of the launch mean (bar) and p90 (label);\n"
                  "whisker = range of launch means; dashed = 33.3 ms (30 FPS), dotted = 66.7 ms (15 FPS)", fontsize=8)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# Cross-device figures (paper Fig. 4 and 5): device A = --matrix, device B = --compare.
# Representative conditions only; INT8 is the head-excluded variant with INT32 bias.
DEVICE_CONDS = [  # (legend, runtime column, variant suffix, color, hatch)
    ("ORT CPU 4T FP32", "cpu_t4", "fp32", "#8c8c8c", ""),
    ("ORT CPU 4T INT8", "cpu_t4", "int8_head_excl", "#8c8c8c", "////"),
    ("WASM 4T FP32", "wasmt4_1300", "fp32", "#ff7f0e", ""),
    ("WASM 4T INT8", "wasmt4_1300", "int8_head_excl", "#ff7f0e", "////"),
    ("WebGPU FP32", "webgpu_1300", "fp32", "#1f77b4", ""),
    ("WebGPU FP16", "webgpu_1300", "fp16", "#9ecae1", ""),
    ("WebGPU INT8", "webgpu_1300", "int8_head_excl", "#1f77b4", "////"),
]
DEVICE_MODELS = [("v4", "YOLO26-n"), ("v3", "YOLOv8s"), ("coco", "YOLO11l (COCO)")]
DEVICE_PIPELINE_ROWS = PIPELINE_ROWS[:1] + PIPELINE_ROWS[2:7]  # WebGPU and 4-thread WASM placements


def fig_latency_devices(rows_a: list[dict], rows_b: list[dict], names: tuple[str, str], out: Path) -> None:
    """Batch-1 latency of the three detectors on both devices, one panel per device (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    width = 0.8 / len(DEVICE_CONDS)
    x = np.arange(len(DEVICE_MODELS))
    for ax, rows, name in zip(axes, (rows_a, rows_b), names, strict=True):
        by = {r["model"]: r for r in rows}
        for i, (legend, col, suffix, color, hatch) in enumerate(DEVICE_CONDS):
            cells = [by.get(f"{fam}_{suffix}", {}).get(col) for fam, _ in DEVICE_MODELS]
            vals = [c["mean_ms"] if c else np.nan for c in cells]
            errs = [max(0.0, c["p90_ms"] - c["mean_ms"]) if c else 0.0 for c in cells]
            ax.bar(x + (i - (len(DEVICE_CONDS) - 1) / 2) * width, vals, width, yerr=[np.zeros(len(cells)), errs],
                   color=color, hatch=hatch, edgecolor="black", linewidth=0.4, label=legend, capsize=1.5,
                   error_kw={"lw": 0.6})
        ax.axhline(1000 / 30, color="black", ls="--", lw=1, label="33.3 ms (30 FPS)")
        ax.axhline(1000 / 15, color="gray", ls=":", lw=1, label="66.7 ms (15 FPS)")
        ax.set_yscale("log")
        ax.set_xticks(x, [m for _, m in DEVICE_MODELS], fontsize=8.5)
        ax.set_title(name, fontsize=9)
        ax.grid(axis="y", alpha=0.3, which="both")
    axes[0].set_ylabel("Latency (ms, log)\nbar = mean, whisker = p90")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=5, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_pipeline_devices(matrix_a: Path, matrix_b: Path, names: tuple[str, str], out: Path) -> None:
    """Per-stage pipeline latency, device A (upper bar) and device B (lower bar) for each placement."""
    groups = []
    for run, label_ in DEVICE_PIPELINE_ROWS:
        bars = []
        for matrix in (matrix_a, matrix_b):
            launches = pipeline_launches(matrix, run)
            if not launches:
                bars.append(None)
                continue
            stage = {k: float(np.median([x["metrics"][k]["mean_ms"] for x in launches])) for k, _, _ in STAGES}
            means = [x["metrics"]["total_ms"]["mean_ms"] for x in launches]
            p90s = [float(np.percentile(x["total"], 90)) for x in launches]
            bars.append((stage, float(np.median(means)), min(means), max(means), float(np.median(p90s)), len(launches)))
        groups.append((label_, bars))
    fig, ax = plt.subplots(figsize=(8.8, 0.8 * len(groups) + 1.6))
    height, limit = 0.36, 115.0
    yticks = []
    for g, (_, bars) in enumerate(groups):
        base = len(groups) - 1 - g
        yticks.append(base)
        for d, bar in enumerate(bars):
            if bar is None:
                continue
            stage, med, lo, hi, p90, n = bar
            y = base + (0.2 if d == 0 else -0.2)
            left = 0.0
            for key, stage_label, color in STAGES:
                ax.barh(y, stage[key], height, left=left, color=color, edgecolor="black" if d else "none",
                        linewidth=0.4, label=stage_label if g == 0 and d == 0 else None)
                left += stage[key]
            if n > 1:
                ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="none", ecolor="black", capsize=2, lw=0.8)
            spread = f", range {lo:.1f}-{hi:.1f}, n={n}" if n > 1 else ", n=1"
            ax.text(max(med, hi) + 1, y, f"{'AB'[d]}: {med:.1f} ms (p90 {p90:.1f}{spread})", va="center", fontsize=6.8,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85})
    ax.axvline(1000 / 30, color="black", ls="--", lw=1)
    ax.axvline(1000 / 15, color="gray", ls=":", lw=1)
    ax.set_yticks(yticks, [label_ for label_, _ in groups], fontsize=7.5)
    ax.set_xlim(0, limit)
    ax.set_xlabel(f"Per-frame latency (ms). Upper bar = A ({names[0].split(':')[0]}), "
                  f"lower bar = B ({names[1].split(':')[0]}).\n"
                  "Bar = median over browser launches of the launch mean; whisker = range of launch means;\n"
                  "dashed = 33.3 ms (30 FPS), dotted = 66.7 ms (15 FPS)", fontsize=7.5)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--compare", type=Path, default=None,
                        help="second device's matrix (e.g. paper_evidence/runtime/matrix_mac): adds fig11 and fig12")
    parser.add_argument("--names", nargs=2, default=["Windows: Ryzen 5 9600X + RTX 5070", "Mac: Apple M2 Pro"])
    parser.add_argument("--figures", type=Path, default=Path("paper_evidence/figures"))
    args = parser.parse_args()
    rows = json.loads((args.matrix / "summary.json").read_text(encoding="utf-8"))["rows"]
    args.figures.mkdir(parents=True, exist_ok=True)
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
