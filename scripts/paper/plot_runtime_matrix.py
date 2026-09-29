"""Figures for the component x precision x runtime matrix (reads summarize_runtime_matrix.py output).

  fig6_component_sensitivity   accuracy retention per component and precision, 95% and 99% lines (paper Fig. 2)
  fig7_runtime_latency         batch-1 latency per variant x runtime on one device (log scale)
  fig8_pipeline_assignment     browser detector+recognizer pipeline stages on one device
  with --compare <second device matrix>:
  fig11_runtime_latency_devices  detector latency, representative conditions, one panel per device (paper Fig. 5)
  fig12_pipeline_devices         key pipeline placements, device A and B bars per placement (paper Fig. 6)

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
    SOFT,
    SOFT_STAGE,
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


# Table-style figures (Figs. 2, 5, 6 in the paper) share the look of the Fig. 1 schematic: fills with a darker
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


# Cross-device figures (paper Fig. 5 and 6): device A = --matrix, device B = --compare (+ --recheck).
# Representative conditions only; INT8 is the head-excluded variant with INT32 bias.
DEVICE_RUNTIMES = [  # (label, summary column, base color): color = where it runs
    ("ORT CPU 4T", "cpu_t4", "#8C8C8C"),
    ("WASM 4T", "wasmt4_1300", OKABE_ITO["orange"]),
    ("WebGPU", "webgpu_1300", OKABE_ITO["blue"]),
]
DEVICE_PRECISIONS = {  # runtime column -> precisions measured there (INT8 = head-excluded, INT32 bias)
    "cpu_t4": ["fp32", "int8_head_excl"], "wasmt4_1300": ["fp32", "int8_head_excl"],
    "webgpu_1300": ["fp32", "fp16", "int8_head_excl"],
}
PRECISION_MARK = {"fp32": ("o", 5.6, "FP32"), "fp16": ("^", 6.2, "FP16"), "int8_head_excl": ("s", 5.2, "INT8")}
DEVICE_MODELS = [("v4", "YOLO26-n"), ("v3", "YOLOv8s"), ("coco", "YOLO11l (COCO)")]
DEVICE_PIPELINE_ROWS = PIPELINE_ROWS[:1] + PIPELINE_ROWS[2:7]  # WebGPU and 4-thread WASM placements


def panel_title(ax, tag: str, name: str, y_pt: float = 4.0) -> None:
    """Panel tag and device in bold ink followed by the hardware in gray, `y_pt` points above the axes."""
    device, _, hardware = name.partition(":")
    title = ax.annotate(f"{tag} {device}", xy=(0, 1), xycoords="axes fraction", xytext=(0, y_pt),
                        textcoords="offset points", fontweight="bold", fontsize=FONT_SIZE - 0.5, color=INK, va="bottom")
    if hardware:
        ax.annotate(f"  {hardware.strip()}", xy=(1, 0), xycoords=title, va="bottom", color=GRAY, fontsize=NOTE)


def _marker_key(marker: str, size: float, face: str, edge: str, tick: bool = False, width: float = 10, height: float = 8):
    """Legend glyph: one marker (or a mean-to-p90 range glyph when `tick`) in a DrawingArea."""
    from matplotlib.offsetbox import DrawingArea

    area = DrawingArea(width + (16 if tick else 0), height, 0, 0)
    y = height / 2
    if tick:
        area.add_artist(Line2D([5, width + 11], [y, y], color=face, linewidth=1.2))
        area.add_artist(Line2D([width + 11], [y], marker="|", markersize=7.5, markeredgewidth=1.2, color=edge))
    area.add_artist(Line2D([5], [y], marker=marker, markersize=size, markerfacecolor=face, markeredgecolor=edge,
                           markeredgewidth=1.0, linestyle="none"))
    return area


def fig_latency_devices(rows_a: list[dict], rows_b: list[dict], names: tuple[str, str], out: Path) -> None:
    """Point-range plot of batch-1 detector latency, one panel per device on a shared log axis.

    Rows: model > runtime. Color = runtime, marker shape = precision; filled marker = mean, tick = p90, joined by a
    line. Precisions sharing a runtime row sit in fixed vertical lanes so near-equal values stay visible.
    """
    lanes = {2: [0.2, -0.2], 3: [0.27, 0.0, -0.27]}
    y_row, head_y, gaps, y = {}, {}, [], 0.0
    for i, (fam, _) in enumerate(DEVICE_MODELS):
        if i:
            gaps.append(-(y - 0.15))
            y += 0.3
        head_y[fam] = -y
        y += 0.85
        for _, col, _ in DEVICE_RUNTIMES:
            y_row[fam, col] = -y
            y += 1.0
    top, bottom = 0.45, -(y - 1.0) - 0.55
    lo_x, hi_x = 8.0, 6000.0

    width_in, height_in = FULL_WIDTH, 3.45
    a_left, a_right, b_left, b_right = 1.2, 3.66, 3.8, width_in - 0.04
    plot_bottom, plot_top = 0.4, height_in - 0.5
    fig = plt.figure(figsize=(width_in, height_in))
    axes = [_axes_in(fig, width_in, height_in, left=a_left, right=a_right, bottom=plot_bottom, top=plot_top),
            _axes_in(fig, width_in, height_in, left=b_left, right=b_right, bottom=plot_bottom, top=plot_top)]
    tab = _Table(fig, axes[0], width_in)
    for fam, name in DEVICE_MODELS:
        tab.text(0.02, head_y[fam], name, weight="bold")
        for runtime, col, _ in DEVICE_RUNTIMES:
            tab.text(0.16, y_row[fam, col], runtime)
    for yg in gaps:
        tab.rule(0.02, b_right, yg, color="#E3E6E8", lw=0.6)

    for ax, rows, name, tag in zip(axes, (rows_a, rows_b), names, ("(a)", "(b)"), strict=True):
        by = {r["model"]: r for r in rows}
        for fam, _ in DEVICE_MODELS:
            for _, col, base in DEVICE_RUNTIMES:
                face, edge = _blend(base, "#FFFFFF", 0.15), _blend(base, "#000000", 0.35)
                precisions = DEVICE_PRECISIONS[col]
                for precision, dy in zip(precisions, lanes[len(precisions)], strict=True):
                    cell = by[f"{fam}_{precision}"][col]
                    mean, p90, yi = cell["mean_ms"], cell["p90_ms"], y_row[fam, col] + dy
                    marker, size, _ = PRECISION_MARK[precision]
                    ax.plot([mean, p90], [yi, yi], color=face, linewidth=1.2, solid_capstyle="butt", zorder=2)
                    ax.plot(mean, yi, marker=marker, markersize=size, markerfacecolor=face, markeredgecolor=edge,
                            markeredgewidth=1.0, linestyle="none", zorder=4)
                    # p90 tick drawn above the marker and slightly taller, so it stays visible when p90 ~ mean
                    ax.plot(p90, yi, marker="|", markersize=7.5, markeredgewidth=1.2, color=edge, zorder=5)
                    if col == "webgpu_1300" and precision == "int8_head_excl":  # the one outlier worth a number
                        text = f"{mean:.0f} ms" if mean < 1000 else f"{mean / 1000:.2f} s"
                        ax.annotate(text, xy=(max(mean, p90), yi), xytext=(5, 0), textcoords="offset points",
                                    va="center", fontsize=NOTE, color=INK)
        ax.axvline(1000 / 30, color="#333333", ls=DASHED, lw=1.0, zorder=1)
        ax.axvline(1000 / 15, color="#8C8C8C", ls=DOTTED, lw=1.1, zorder=1)
        ax.annotate("30 FPS", xy=(1000 / 30, 1), xycoords=ax.get_xaxis_transform(), xytext=(-2, 1.5),
                    textcoords="offset points", ha="right", va="bottom", fontsize=NOTE - 0.5, color=INK)
        ax.annotate("15 FPS", xy=(1000 / 15, 1), xycoords=ax.get_xaxis_transform(), xytext=(2, 1.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=NOTE - 0.5, color=GRAY)
        ax.set_xscale("log")
        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(bottom, top)
        ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", which="major", color="#EBEBEB", linewidth=0.5)
        ax.set_axisbelow(True)
        panel_title(ax, tag, name, y_pt=11.5)
    fig.text((a_left + b_right) / 2 / width_in, 0.07 / height_in, "Latency (ms, log scale)", ha="center", va="bottom",
             fontsize=FONT_SIZE, color=INK)

    runtime_key = [("Runtime", {"color": GRAY, "fontsize": NOTE})]
    for runtime, _, base in DEVICE_RUNTIMES:
        runtime_key += [_marker_key("o", 5.6, _blend(base, "#FFFFFF", 0.15), _blend(base, "#000000", 0.35)),
                        runtime.replace(" 4T", "")]
    precision_key = [("Precision", {"color": GRAY, "fontsize": NOTE})]
    for marker, size, label_text in PRECISION_MARK.values():
        precision_key += [_marker_key(marker, size, "#FFFFFF", INK), label_text]
    stat_key = [_marker_key("o", 5.6, "#9A9A9A", "#333333", tick=True),
                ("filled = mean,  tick = p90", {"color": GRAY, "fontsize": NOTE})]
    key_row(fig, [runtime_key, precision_key, stat_key], x=0.5, y=(height_in - 0.07) / height_in, gap=14.0)
    save(fig, out)
    plt.close(fig)


def fig_pipeline_devices(matrix_a: Path, matrices_b: list[Path], names: tuple[str, str], out: Path) -> None:
    """Per-stage pipeline latency on both devices, as a table: detector | recognizer | device, stacked stage bars,
    then the median launch mean, its p90 and the number of launches. Device B reads `matrices_b` in priority order."""
    groups = []
    for run, name in DEVICE_PIPELINE_ROWS:
        det, rec = (part.split(" ", 1)[1].replace(" @", " · ") for part in name.split(" + "))
        groups.append((det, rec, [launch_summary(matrix_a, run), first_summary(matrices_b, run)]))
    devices = [n.split(":")[0] for n in names]
    top, bottom = 0.55, -(len(groups) - 1) - 0.55
    axes_top = 0.42 + 0.36 * (top - bottom)  # 0.36 in per placement
    width_in, height_in = FULL_WIDTH, axes_top + 0.5
    x_det, x_rec, x_dev, x_mean, x_p90, x_n, x_end = 0.02, 1.3, 2.1, 5.6, 5.95, 6.24, 6.28
    fig = plt.figure(figsize=(width_in, height_in))
    ax = _axes_in(fig, width_in, height_in, left=2.56, right=5.2, bottom=0.42, top=axes_top)
    tab = _Table(fig, ax, width_in)
    limit, height = 72.0, 0.32

    header = top + 0.5
    for x, s, ha in ((x_det, "Detector", "left"), (x_rec, "Recognizer", "left"), (x_dev, "Device", "left"),
                     (x_mean, "Mean", "right"), (x_p90, "p90", "right"), (x_n, "n", "right")):
        tab.text(x, header, s, ha=ha, color=GRAY, size=NOTE)
    tab.rule(x_det, x_end, top + 0.1, color="#C9CED2")
    for g, (det, rec, bars) in enumerate(groups):
        base = -g
        tab.text(x_det, base, det)
        tab.text(x_rec, base, rec)
        if g:
            tab.rule(x_det, x_end, base + 0.5)
        for d, bar in enumerate(bars):
            if bar is None:
                continue
            stage, med, lo, hi, p90, n = bar
            y = base + (0.19 if d == 0 else -0.19)
            left = 0.0
            for key, _ in STAGES:
                fill, edge, _ = SOFT[SOFT_STAGE[key]]
                ax.barh(y, min(stage[key], limit - left), height, left=left, facecolor=fill, edgecolor=edge,
                        linewidth=0.45)
                left += stage[key]
            if n > 1:
                ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="none", ecolor=INK, elinewidth=0.6,
                            capsize=1.3, capthick=0.6)
            tab.text(x_dev, y, devices[d], color=GRAY, size=NOTE)
            tab.text(x_mean, y, f"{med:.1f}", ha="right")
            tab.text(x_p90, y, f"{p90:.1f}", ha="right", color=GRAY)
            tab.text(x_n, y, f"{n}", ha="right", color=GRAY)

    ax.axvline(1000 / 30, color=INK, ls=DASHED, lw=0.7)
    ax.axvline(1000 / 15, color=INK, ls=DOTTED, lw=0.8)
    for fps, xv in (("30 FPS", 1000 / 30), ("15 FPS", 1000 / 15)):
        ax.text(xv, header, fps, ha="center", va="center", color=GRAY, fontsize=NOTE, clip_on=False)
    ax.set_xlim(0, limit)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.set_xticks(range(0, int(limit) + 1, 10))
    ax.set_xlabel("Per-frame latency (ms)")
    ax.grid(axis="x", color=HAIR, linewidth=0.5)
    ax.set_axisbelow(True)
    key = [[swatch(SOFT[SOFT_STAGE[k]][0], SOFT[SOFT_STAGE[k]][1]), name] for k, name in STAGES]
    key.append([("bar = median of launch means;  whisker = range over launches", {"color": GRAY, "fontsize": NOTE})])
    key_row(fig, key, x=0.5, y=(height_in - 0.08) / height_in, gap=9.0)
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
