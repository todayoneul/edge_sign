"""Figures for the component x precision x runtime matrix (reads summarize_runtime_matrix.py output).

  fig6_component_sensitivity   accuracy retention per component and precision, 95% and 99% lines (paper Fig. 2)
  fig7_runtime_latency         batch-1 latency per variant x runtime on one device (log scale; not in the draft)
  fig8_pipeline_assignment     browser detector+recognizer pipeline stages on one device (not in the draft)
  with --compare <second device matrix>:
  fig11_runtime_latency_devices  detector latency bars, representative conditions, one panel per device (paper Fig. 3)
  fig12_pipeline_devices         key pipeline placements, device A and B stage bars per placement (paper Fig. 4)

Paper figure numbers follow the current draft (Fig. 1 = fig9 schematic, Fig. 5 = fig10 qualitative).

Every figure is written as <name>.pdf (vector, for the manuscript) and <name>.png (600 dpi) in the
shared paper style (scripts/paper/paper_style.py).

--recheck <quiet re-check of device B> (needs --compare): the re-check applies the same CPU-quiet gate as the
device A runs, so wherever it repeated a measurement it replaces the --compare value. The WASM 4T detector cells
(YOLO26-n, YOLOv8s) become the median over its launches of the launch mean and p90, and the pipeline bars use its
launches. Everything it did not repeat (YOLO11l, 1-thread WASM, the FP32 @WebGPU pipelines) stays as measured in
the first run; both raw records stay in their own folders.

Usage: python scripts/paper/plot_runtime_matrix.py --matrix paper_evidence/runtime/matrix
       [--compare paper_evidence/runtime/matrix_mac [--recheck paper_evidence/runtime/matrix_mac_recheck]]
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.paper_style import (
    FONT_SIZE,
    FULL_WIDTH,
    GRAY,
    HAIR,
    INK,
    MODEL_COLOR,
    OKABE_ITO,
    STAGE_COLOR,
    apply,
    key_row,
    save,
    swatch,
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


# The table-style figure (Fig. 2 in the paper) shares the look of the Fig. 1 schematic: fills with a darker
# outline of the same hue, ink for primary text, gray for secondary text, hairline rules between groups.
DASHED, DOTTED = (0, (4, 2.5)), (0, (1, 1.6))  # 99% criterion / 30 FPS, and 95% reference / 15 FPS
CELL = FONT_SIZE - 1  # table cell text
NOTE = FONT_SIZE - 1.5  # headers and secondary text


class _Table:
    """Text columns placed at x inches from the figure's left edge and y in the data units of `ax`.

    Built on transFigure (not dpi_scale_trans) so the columns stay aligned with the axes under bbox='tight'.
    """

    def __init__(self, fig, ax, width_in: float):
        self.ax, self.width = ax, width_in
        self.tr = blended_transform_factory(fig.transFigure, ax.transData)

    def text(self, x_in: float, y: float, s: str, *, ha: str = "left", color: str = INK, size: float = CELL, **kw):
        return self.ax.text(x_in / self.width, y, s, transform=self.tr, ha=ha, va="center", color=color,
                            fontsize=size, clip_on=False, **kw)

    def rule(self, x0_in: float, x1_in: float, y: float, color: str = HAIR, lw: float = 0.5) -> None:
        self.ax.add_line(Line2D([x0_in / self.width, x1_in / self.width], [y, y], transform=self.tr, color=color,
                                linewidth=lw, clip_on=False))


def _axes_in(fig, width_in: float, height_in: float, left: float, right: float, bottom: float, top: float):
    """Axes whose edges are given in inches from the figure's left/bottom edge."""
    return fig.add_axes([left / width_in, bottom / height_in, (right - left) / width_in, (top - bottom) / height_in])


def _blend(hex_color: str, toward: str, amount: float) -> str:
    """Mix a color toward white/black by `amount` (0..1); used to soften the Okabe-Ito hues."""
    import matplotlib.colors as mcolors

    a, b = np.array(mcolors.to_rgb(hex_color)), np.array(mcolors.to_rgb(toward))
    return mcolors.to_hex(a + (b - a) * amount)


def fig_sensitivity(rows: list[dict], bootstrap: dict, out: Path) -> None:
    """Forest plot of retention per component and variant: point = retention vs FP32, whisker = 95% bootstrap CI,
    zoomed on 94-101% with the 95% reference and 99% criterion. Collapsed variants (no detections) cannot sit on the
    truncated axis, so they get no point, only a cross and a label at a fixed position inside the plot."""
    by = {r["model"]: r for r in rows}
    keys = [k for k in SENS_ROWS if by.get(k, {}).get("retention") is not None]
    families = list(dict.fromkeys(by[k]["family"] for k in keys))
    y_of, head_y, gaps, y = {}, {}, [], 0.0
    for i, fam in enumerate(families):
        if i:
            gaps.append(-(y - 0.1))
            y += 0.2
        head_y[fam] = -y  # model name on its own row, variants indented below it
        y += 1.0
        for k in (k for k in keys if by[k]["family"] == fam):
            y_of[k] = -y
            y += 1.0
    top, bottom = 0.45, -(y - 1.0) - 0.5
    lo_x, hi_x = 94.0, 101.0
    marker = {"v4": ("o", 3.9), "v3": ("s", 3.5), "rec": ("^", 4.3), "coco": ("D", 3.4)}  # shape survives grayscale
    role = {"v4": "detector · mAP", "v3": "detector · mAP", "rec": "recognizer · Top-1", "coco": "detector · mAP"}
    ci_color, ref95, ref99 = "#333333", "#9A9FA4", "#333333"

    width_in, height_in = FULL_WIDTH, 3.1
    x_model, x_var = 0.02, 0.16
    fig = plt.figure(figsize=(width_in, height_in))
    ax = _axes_in(fig, width_in, height_in, left=1.3, right=width_in - 0.16, bottom=0.42, top=height_in - 0.17)
    tab = _Table(fig, ax, width_in)
    for fam in families:
        name = tab.text(x_model, head_y[fam], MODEL_NAME[fam], weight="bold")
        ax.annotate(f"  {role[fam]}", xy=(1, 0.5), xycoords=name, va="center", color=GRAY, fontsize=NOTE - 0.5)
    for yg in gaps:
        tab.rule(x_model, width_in - 0.08, yg, color="#ECEEF0", lw=0.35)

    for key in keys:
        r, yi = by[key], y_of[key]
        base = MODEL_COLOR[r["family"]]  # Okabe-Ito hue, softened fill and slightly darker edge
        value = r["retention"] * 100
        tab.text(x_var, yi, SHORT[key.split("_", 1)[1]])
        if value < lo_x:  # collapsed: no point or CI, same cross position in every such row
            ax.text(lo_x + 0.12, yi, "×  No detections", va="center", ha="left", color=GRAY, fontsize=NOTE,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
            continue
        right = value
        if ci := bootstrap.get(key):
            lo, hi = ci["ci95_low"] * 100, ci["ci95_high"] * 100
            ax.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt="none", ecolor=ci_color, elinewidth=0.55,
                        capsize=1.3, capthick=0.55, zorder=3)
            right = hi
        shape, size = marker[r["family"]]
        ax.plot(value, yi, marker=shape, markersize=size, markerfacecolor=_blend(base, "#FFFFFF", 0.35),
                markeredgecolor=_blend(base, "#000000", 0.25), markeredgewidth=0.6, linestyle="none", zorder=4)
        # clear the whisker cap, or the marker itself when the CI is shorter than the marker
        pt_per_unit = ax.get_position().width * width_in * 72 / (hi_x - lo_x)
        gap = max(3.5, size / 2 + 2.5 - (right - value) * pt_per_unit)
        ax.annotate(f"{value:.1f}", xy=(right, yi), xytext=(gap, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=NOTE, color=INK)

    ax.axvline(95, color=ref95, ls=DOTTED, lw=0.7, zorder=1)
    ax.axvline(99, color=ref99, ls=DASHED, lw=0.7, zorder=1)
    for xv, s, color in ((95, "95% reference", GRAY), (99, "99% criterion", INK)):
        ax.annotate(s, xy=(xv, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=NOTE, color=color)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(int(lo_x), int(hi_x) + 1))
    ax.set_xlabel("Accuracy retention vs. FP32 (%)")
    ax.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.4)
    ax.set_axisbelow(True)
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


LAUNCH_SUFFIXES = ["", "_r2", "_r3", "_r4", "_r5"]  # launch 1 carries no suffix


def pipeline_launches(matrix: Path, run: str) -> list[dict]:
    """Completed launches of one configuration: r1 (no suffix) and r2..r5."""
    out = []
    for suffix in LAUNCH_SUFFIXES:
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


def first_summary(matrices: list[Path], run: str) -> tuple | None:
    """launch_summary from the first matrix that measured this configuration (the quiet re-check goes first)."""
    return next((s for m in matrices if (s := launch_summary(m, run))), None)


def wasm4t_launches(matrix: Path, model: str) -> list[dict]:
    """inference_ms metrics of the completed single-model WASM 4T launches (ORT-Web 1.30) of one model variant."""
    out = []
    for suffix in LAUNCH_SUFFIXES:
        p = matrix / f"speed_wasmt4_ort1300_{model}{suffix}.json"
        if p.exists() and (r := json.loads(p.read_text(encoding="utf-8"))).get("status") == "completed":
            out.append(r["metrics"]["inference_ms"])
    return out


def with_recheck(rows: list[dict], recheck: Path) -> list[dict]:
    """Rows with the WASM 4T cell of every re-checked model replaced by the median over launches of the launch mean
    and p90 (the aggregation used for the pipeline bars). Models the re-check did not run keep their row."""
    out = []
    for r in rows:
        launches = wasm4t_launches(recheck, r["model"])
        if not launches:
            out.append(r)
            continue
        p90 = float(np.median([m["p90_ms"] for m in launches]))
        cell = {**(r.get("wasmt4_1300") or {}),
                "mean_ms": float(np.median([m["mean_ms"] for m in launches])), "p90_ms": p90,
                "p99_ms": float(np.median([m["p99_ms"] for m in launches])), "n_launches": len(launches),
                "A_30fps": p90 <= FPS_LINES[0][0], "B_15fps": p90 <= FPS_LINES[1][0]}
        out.append({**r, "wasmt4_1300": cell})
    return out


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


# Cross-device figures (paper Figs. 3 and 4): device A = --matrix, device B = --compare (+ --recheck).
# Representative conditions only; INT8 is the head-excluded variant with INT32 bias.
DEVICE_RUNTIMES = [  # (label, summary column, FP32 fill, outline): color = where it runs
    ("ORT CPU 4T", "cpu_t4", "#8F8F8F", "#505050"),
    ("WASM 4T", "wasmt4_1300", "#E89F5D", "#AF641C"),
    ("WebGPU", "webgpu_1300", "#2B689E", "#1E4A73"),
]
DEVICE_PRECISIONS = {  # runtime column -> precisions measured there (INT8 = head-excluded, INT32 bias)
    "cpu_t4": ["fp32", "int8_head_excl"], "wasmt4_1300": ["fp32", "int8_head_excl"],
    "webgpu_1300": ["fp32", "fp16", "int8_head_excl"],
}
PRECISION_TINT = {"fp32": 0.0, "fp16": 0.5, "int8_head_excl": 0.9}  # runtime fill blended toward white by this much
PRECISION_KEY = [("FP32", "#5E5E5E"), ("FP16", "#AEAEAE"), ("INT8", "#EFEFEF")]
DEVICE_MODELS = [("v4", "YOLO26-n", None), ("v3", "YOLOv8s", None), ("coco", "YOLO11l", "COCO")]  # (family, name, note)
DEVICE_PIPELINE_ROWS = PIPELINE_ROWS[:1] + PIPELINE_ROWS[2:7]  # WebGPU and 4-thread WASM placements


def panel_title(ax, name: str, y_pt: float = 4.0) -> None:
    """Device in bold ink followed by the hardware in gray ("Windows: Ryzen ..."), `y_pt` points above the axes."""
    device, _, hardware = name.partition(":")
    title = ax.annotate(device, xy=(0, 1), xycoords="axes fraction", xytext=(0, y_pt), textcoords="offset points",
                        fontweight="bold", fontsize=FONT_SIZE, color=INK, va="bottom")
    if hardware:
        ax.annotate(f"  {hardware.strip()}", xy=(1, 0), xycoords=title, va="bottom", color=GRAY, fontsize=CELL)


def fig_latency_devices(rows_a: list[dict], rows_b: list[dict], names: tuple[str, str], out: Path) -> None:
    """Grouped bars of batch-1 detector latency, one panel per device on a shared log axis (paper Fig. 3).

    Groups = model; bar color = runtime, bar tint = precision (FP32 full, FP16 half, INT8 pale with the runtime
    outline). Bar = mean, whisker = p90.
    """
    bar_w, pitch, gap = 0.093, 0.114, 0.036  # in model-group units (one group = 1.0)
    offsets, bars = [], []
    x = 0.0
    for r, (_, col, fill, edge) in enumerate(DEVICE_RUNTIMES):
        x += gap if r else 0.0
        for precision in DEVICE_PRECISIONS[col]:
            offsets.append(x)
            bars.append((col, precision, _blend(fill, "#FFFFFF", PRECISION_TINT[precision]), edge))
            x += pitch
    center = (offsets[0] + offsets[-1]) / 2
    offsets = [o - center for o in offsets]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.11), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.925, bottom=0.12, top=0.875, wspace=0.06)
    for ax, rows, name in zip(axes, (rows_a, rows_b), names, strict=True):
        by = {r["model"]: r for r in rows}
        for g, (fam, _, _) in enumerate(DEVICE_MODELS):
            for dx, (col, precision, face, edge) in zip(offsets, bars, strict=True):
                cell = by[f"{fam}_{precision}"][col]
                mean, p90 = cell["mean_ms"], cell["p90_ms"]
                ax.bar(g + dx, mean, bar_w, color=face, edgecolor=edge, linewidth=0.6, zorder=2)
                ax.errorbar(g + dx, mean, yerr=[[0.0], [max(0.0, p90 - mean)]], fmt="none", ecolor=INK,
                            elinewidth=0.7, capsize=2.0, capthick=0.9, zorder=3)
        ax.axhline(1000 / 30, color="#333333", ls=DASHED, lw=1.0, zorder=1)
        ax.axhline(1000 / 15, color="#8C8C8C", ls=DOTTED, lw=1.1, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(5, 3000)
        ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.5, len(DEVICE_MODELS) - 0.5)
        ax.set_xticks(range(len(DEVICE_MODELS)), [m for _, m, _ in DEVICE_MODELS])
        ax.tick_params(axis="x", length=0, pad=3)
        for g, (_, _, note) in enumerate(DEVICE_MODELS):
            if note:
                ax.annotate(note, xy=(g, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -13),
                            textcoords="offset points", ha="center", va="top", color=GRAY, fontsize=NOTE)
        ax.grid(axis="y", which="major", color="#EBEBEB", linewidth=0.5)
        ax.set_axisbelow(True)
        panel_title(ax, name, y_pt=5.0)
    axes[0].set_ylabel("Latency (ms)")
    for yv, text, color in ((1000 / 15, "15 FPS", GRAY), (1000 / 30, "30 FPS", INK)):
        axes[1].annotate(text, xy=(1, yv), xycoords=axes[1].get_yaxis_transform(), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=NOTE, color=color)

    runtime_key = [("Runtime", {"color": GRAY})]
    for runtime, _, fill, edge in DEVICE_RUNTIMES:
        runtime_key += [swatch(fill, edge), runtime]
    precision_key = [("Precision", {"color": GRAY})]
    for text, fill in PRECISION_KEY:
        precision_key += [swatch(fill, INK), text]
    note_key = [("bar = mean; whisker = p90", {"color": GRAY})]
    key_row(fig, [runtime_key, precision_key, note_key], x=0.5, y=0.975, gap=16.0)
    save(fig, out)
    plt.close(fig)


def fig_pipeline_devices(matrix_a: Path, matrices_b: list[Path], names: tuple[str, str], out: Path) -> None:
    """Per-stage pipeline latency, device A (upper bar) and device B (lower, outlined bar) for each placement
    (paper Fig. 4). Device B reads `matrices_b` in priority order (the quiet re-check first)."""
    groups = [(name, [launch_summary(matrix_a, run), first_summary(matrices_b, run)])
              for run, name in DEVICE_PIPELINE_ROWS]
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
                ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="none", ecolor=OKABE_ITO["black"],
                            elinewidth=0.6, capsize=1.5)
            ax.text(max(med, hi) + 1.2, y, f"{'AB'[d]}  {med:.1f} ms · p90 {p90:.1f} · n={n}", va="center", fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})
    fps_lines(ax)
    ax.set_yticks(yticks, [name for name, _ in groups])
    ax.set_xlim(0, limit)
    ax.set_xlabel(f"Per-frame latency (ms). A (upper): {names[0].split(':')[0]}; "
                  f"B (lower, outlined): {names[1].split(':')[0]}; whisker: range over launches")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=OKABE_ITO["black"], ls="--", lw=0.8),
                Line2D([], [], color=OKABE_ITO["black"], ls=":", lw=0.8)]
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
    parser.add_argument("--recheck", type=Path, default=None,
                        help="quiet re-check of the second device (e.g. paper_evidence/runtime/matrix_mac_recheck): "
                             "replaces the --compare values it repeated")
    parser.add_argument("--names", nargs=2, default=["Windows: Ryzen 5 9600X + RTX 5070", "Mac: Apple M2 Pro"])
    parser.add_argument("--figures", type=Path, default=Path("paper_evidence/figures"))
    args = parser.parse_args()
    if args.recheck and not args.compare:
        parser.error("--recheck needs --compare")
    apply()
    rows = json.loads((args.matrix / "summary.json").read_text(encoding="utf-8"))["rows"]
    boot = args.matrix / "bootstrap_retention.json"
    pairs = json.loads(boot.read_text(encoding="utf-8"))["pairs"] if boot.exists() else {}
    fig_sensitivity(rows, pairs, args.figures / "fig6_component_sensitivity.png")
    fig_latency(rows, args.figures / "fig7_runtime_latency.png")
    fig_pipeline(args.matrix, args.figures / "fig8_pipeline_assignment.png")
    if args.compare:
        rows_b = json.loads((args.compare / "summary.json").read_text(encoding="utf-8"))["rows"]
        matrices_b = [args.compare]
        if args.recheck:
            rows_b = with_recheck(rows_b, args.recheck)
            matrices_b.insert(0, args.recheck)
        names = (args.names[0], args.names[1])
        fig_latency_devices(rows, rows_b, names, args.figures / "fig11_runtime_latency_devices.png")
        fig_pipeline_devices(args.matrix, matrices_b, names, args.figures / "fig12_pipeline_devices.png")
    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
