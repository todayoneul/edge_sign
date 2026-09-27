"""Shared figure style for the paper: one clean, journal-standard look for every figure.

Conventions (cf. J.-B. Huang, "Deep Paper Gestalt", arXiv:1812.08775; standard IEEE/Springer practice):
- Arial 8 pt at final print size (Helvetica-like sans stays legible when a figure is scaled into the page);
- width = the TIIS single-column text width (6.3 in), heights chosen per figure; no rescaling in the document;
- Okabe-Ito colorblind-safe palette with one fixed color per model, per runtime and per pipeline stage;
  the table-style paper figures (Figs. 2, 5, 6) use the SOFT palette below to match the Fig. 1 schematic;
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

# Soft palette shared with the schematic in Fig. 1 (plot_study_overview.py): a muted fill, a darker outline of the
# same hue and a pale tint per hue, with ink/gray text. Adjacent hues in each figure pass a protan/deutan check
# (OKLab dE >= 8); "slate" and "light" are deliberately achromatic (CPU baseline, minor stage).
INK = "#252525"
GRAY = "#62686D"
HAIR = "#E3E6E8"  # group separators
SOFT = {  # hue: (fill, edge, tint)
    "blue": ("#5A8DB5", "#2F5F85", "#DCE9F1"),
    "rust": ("#C46F4E", "#8E4529", "#F5E1D8"),
    "green": ("#6DB396", "#33775B", "#DDEFE6"),
    "ochre": ("#E0B75A", "#8A6A16", "#F7ECCD"),
    "slate": ("#8E979E", "#50585E", "#ECEEF0"),
    "light": ("#C9CED2", "#7D858B", "#F3F4F4"),
}
SOFT_MODEL = {"v4": "blue", "v3": "rust", "rec": "green", "coco": "ochre"}
SOFT_STAGE = {"det_prepare_ms": "light", "det_ms": "blue", "decode_ms": "rust", "rec_prepare_ms": "ochre",
              "rec_ms": "green"}


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


def swatch(face: str, edge: str, hatch: str | None = None, width: float = 9, height: float = 6.5):
    """Legend key patch drawn like the boxes in Fig. 1 (fill + thin outline)."""
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Rectangle

    area = DrawingArea(width, height, 0, 0)
    area.add_artist(Rectangle((0, 0), width, height, facecolor=face, edgecolor=edge, linewidth=0.6, hatch=hatch))
    return area


def line_swatch(linestyle, color: str = INK, width: float = 14, height: float = 6.5):
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import DrawingArea

    area = DrawingArea(width, height, 0, 0)
    area.add_artist(Line2D([0, width], [height / 2] * 2, color=color, linestyle=linestyle, linewidth=0.8))
    return area


def key_row(fig, groups: list[list], x: float, y: float, gap: float = 11.0):
    """One-line legend of labelled groups centred at figure fraction (x, y).

    A group entry is a str (ink text), a (str, text-props dict) pair, or a swatch()/line_swatch() area.
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    def item(entry):
        if isinstance(entry, str):
            return TextArea(entry, textprops={"color": INK, "fontsize": FONT_SIZE - 1})
        if isinstance(entry, tuple):
            return TextArea(entry[0], textprops={"color": INK, "fontsize": FONT_SIZE - 1, **entry[1]})
        return entry

    packed = [HPacker(children=[item(e) for e in g], sep=3.0, align="center") for g in groups]
    box = AnchoredOffsetbox(loc="center", child=HPacker(children=packed, sep=gap, align="center"), frameon=False,
                            bbox_to_anchor=(x, y), bbox_transform=fig.transFigure, borderpad=0, pad=0)
    fig.add_artist(box)
    return box


def save(fig, png: Path, dpi: int = 600) -> None:
    """Vector PDF plus PNG (600 dpi for line art; photographs use 300 dpi, the usual journal minimum)."""
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(png.with_suffix(".pdf"))
