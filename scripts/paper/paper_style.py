"""Shared figure style for the paper: one clean, journal-standard look for every figure.

Conventions (cf. J.-B. Huang, "Deep Paper Gestalt", arXiv:1812.08775; standard IEEE/Springer practice):
- Arial 8 pt at final print size (Helvetica-like sans stays legible when a figure is scaled into the page);
- width = the TIIS single-column text width (6.3 in), heights chosen per figure; no rescaling in the document;
- Okabe-Ito colorblind-safe palette with one fixed color per model, per runtime and per pipeline stage;
- no top/right spines, a light grid on the value axis only, frameless legends, panel labels instead of titles;
- every figure is written as vector PDF (for the manuscript) and 600-dpi PNG (for the Markdown draft).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#8C8C8C",
}
MODEL_COLOR = {"v4": OKABE_ITO["blue"], "v3": OKABE_ITO["vermillion"], "rec": OKABE_ITO["green"],
               "coco": OKABE_ITO["purple"]}
RUNTIME_COLOR = {"cpu": OKABE_ITO["gray"], "wasm": OKABE_ITO["orange"], "webgpu": OKABE_ITO["blue"],
                 "webgpu_fp16": OKABE_ITO["sky"]}
STAGE_COLOR = {"det_prepare_ms": "#C8C8C8", "det_ms": OKABE_ITO["blue"], "decode_ms": OKABE_ITO["purple"],
               "rec_prepare_ms": OKABE_ITO["orange"], "rec_ms": OKABE_ITO["green"]}
FULL_WIDTH = 6.3  # in, TIIS single-column text width
FONT_SIZE = 8


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 0.5, "ytick.labelsize": FONT_SIZE - 0.5, "legend.fontsize": FONT_SIZE - 0.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.0,
        "hatch.linewidth": 0.6, "errorbar.capsize": 1.5, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType (editable text in the PDF)
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def value_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, which="major", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, text: str) -> None:
    """'(a) ...' at the top-left of a panel, used instead of a title."""
    ax.set_title(text, loc="left", fontsize=FONT_SIZE, fontweight="bold", pad=4)


def save(fig, png: Path, dpi: int = 600) -> None:
    """Vector PDF plus PNG (600 dpi for line art; photographs use 300 dpi, the usual journal minimum)."""
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(png.with_suffix(".pdf"))
