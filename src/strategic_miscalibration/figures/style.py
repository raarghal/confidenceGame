"""One look for every figure: fonts, colors and a single save path."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["AMBER", "BLUE", "GREEN", "GRID", "INK", "INK_2", "INK_3", "ORANGE", "VIOLET", "finish", "save"]

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#009E73"
VIOLET, AMBER = "#7b3fa8", "#b07d00"
INK, INK_2, INK_3 = "#111111", "#4d4d4d", "#8a8880"
GRID = "#dedcd7"

mpl.rcParams.update(
    {
        "font.size": 7.4,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 7.6,
        "axes.labelsize": 7.4,
        "axes.edgecolor": INK_3,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 6.2,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
)


def finish(ax: plt.Axes, axis: Literal["both", "x", "y"] = "both") -> None:
    """The light grid every panel uses, behind the data."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path, tight: bool = True) -> Path:
    """Write ``fig`` (PDF, plus a PNG preview) and close it.

    ``tight=False`` keeps the canvas size exactly, for figures sized to the page with constrained layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(path, bbox_inches=bbox)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches=bbox)
    plt.close(fig)
    return path
