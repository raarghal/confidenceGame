"""Figures of Section 6 and its appendix: what the measured strategy costs the user."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.layout_engine import ConstrainedLayoutEngine

from ..analysis import welfare
from .style import BLUE, INK, INK_3, ORANGE, finish, save

__all__ = ["welfare_body", "welfare_full"]


def _breakeven_edge(ax: plt.Axes, grid: np.ndarray, level: float = 100.0) -> None:
    """The break-even frontier drawn on tile boundaries: it lies *between* measured states, nowhere more precisely."""
    above = grid > level
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            if not above[i, j]:
                continue
            if j + 1 >= cols or not above[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=INK, lw=1.5, solid_capstyle="butt", zorder=4)
            if i + 1 >= rows or not above[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=INK, lw=1.5, solid_capstyle="butt", zorder=4)


def _heatmap(
    ax: plt.Axes, t: pd.DataFrame, col: str, norm: mpl.colors.Normalize, fontsize: float
) -> mpl.image.AxesImage:
    piv = t.pivot_table(index="mu", columns="h", values=col, aggfunc="mean")
    im = ax.imshow(piv.values, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if (v > 210 or v < 30) else INK,
            )
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
    ax.set_xlabel(r"$h$   (share honest)", labelpad=1)
    if col == "naive_pct":
        _breakeven_edge(ax, piv.values)
    return im


def _composition(t: pd.DataFrame) -> pd.Series:
    d = t.groupby("h")[["destroyed", "total"]].sum()
    return 100 * d.destroyed / d.total


def welfare_body(path: Path) -> Path:
    """Figure 3: (a) the naive user's loss with the break-even frontier; (b) destruction against exploitation
    by trust; (c) two-period payoff of the three users against the true able share at ``h* = 0``."""
    t = welfare.state_table("ledger")
    fig = plt.figure(
        figsize=(6.5, 2.00), layout=ConstrainedLayoutEngine(w_pad=0.008, h_pad=0.008, wspace=0.05, hspace=0.06)
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 0.90, 1.14])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=100, vmax=max(float(t.naive_pct.max()), 101))
    im = _heatmap(a, t, "naive_pct", norm, 5.6)
    a.set_ylabel(r"$\mu$   (share able)", labelpad=1)
    a.set_title("(a)", loc="left", pad=2)
    a.set_title("Naive User's loss", loc="center", pad=2)
    bar = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    bar.set_label("% of gain from trade lost", fontsize=5.8)
    bar.ax.tick_params(labelsize=5.6)
    bar.ax.axhline(100, color=INK, lw=1.5)

    share = _composition(t)
    xs = np.arange(len(share))
    b.bar(xs, share, width=0.74, color=BLUE)
    b.bar(xs, 100 - share, width=0.74, bottom=share, color=ORANGE)
    for i, v in enumerate(share):
        b.text(i, v - 2.5, f"{v:.0f}", ha="center", va="top", fontsize=5.6, color="white")
    b.set_xticks(xs, [f"{h:g}" for h in share.index])
    b.set(xlabel=r"$h$   (share honest)", ylabel="share of total loss (%)", ylim=(0, 100))
    b.text(xs[0], 4.0, "destruction", ha="center", va="bottom", fontsize=5.5, color="white", rotation=90)
    b.text(
        xs[0],
        float(share.iloc[0]) + 4.0,
        "exploitation",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="white",
        rotation=90,
    )
    b.set_title("(b)", loc="left", pad=2)
    b.set_title("Composition", loc="center", pad=2)
    finish(b, "y")

    s = welfare.capability_sweep()
    outside = float(s.outside.iloc[0])
    finish(c)
    c.axhline(outside, color=INK_3, lw=0.9, ls=":", zorder=2)
    c.annotate(
        "never delegate",
        xy=(0.86, outside),
        xytext=(0, -3),
        textcoords="offset points",
        fontsize=5.5,
        color=INK_3,
        va="top",
        ha="center",
    )
    c.plot(s.mu, s.bench, color=INK_3, lw=1.0, zorder=3, label="honest agent (benchmark)")
    style: dict[str, Any] = {"lw": 1.4, "ms": 2.6, "mec": "white", "mew": 0.5, "zorder": 4}
    c.plot(s.mu, s.naive, color=ORANGE, marker="o", label="naive", **style)
    c.plot(s.mu, s.calib, color=BLUE, marker="s", label="knows the rule", **style)
    c.plot(s.mu, s.inf, color=INK, ls="-.", marker="^", label="knows rule + population", **{**style, "lw": 1.2})
    gap = 100 * float(s.bench.iloc[-1] - s.inf.iloc[-1]) / float(s.bench.iloc[-1] - outside)
    c.annotate(
        "",
        xy=(1.0, s.bench.iloc[-1]),
        xytext=(1.0, s.inf.iloc[-1]),
        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
    )
    c.text(
        0.74,
        0.5 * (s.bench.iloc[-1] + s.inf.iloc[-1]) - 0.07,
        f"{gap:.0f}% still lost",
        fontsize=5.5,
        color=INK,
        ha="center",
        va="center",
    )
    c.set(xlabel=r"$\mu^*$   (true share able, at $h^*=0$)", ylabel="user payoff, two rounds", xlim=(-0.04, 1.12))
    c.legend(loc="lower right", fontsize=5.2, handlelength=1.4, handletextpad=0.3, labelspacing=0.12, borderpad=0.12)
    c.set_title("(c)", loc="left", pad=2)
    c.set_title("Scaling Capability", loc="center", pad=2)
    return save(fig, path, tight=False)


def welfare_full(path: Path) -> Path:
    """Figure 5: the naive and sophisticated users' losses at each state (population = belief), and the
    destruction/exploitation split."""
    t = welfare.state_table("ledger")
    fig = plt.figure(
        figsize=(6.5, 2.42), layout=ConstrainedLayoutEngine(w_pad=0.008, h_pad=0.008, wspace=0.04, hspace=0.06)
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[2.05, 1.0])
    top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    bottom = fig.add_subplot(gs[1, :])
    vmax = max(float(t.naive_pct.max()), float(t.soph_pct.max()), 101)
    norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=100, vmax=vmax)
    panels = (("naive_pct", "(a)", "Naive User"), ("soph_pct", "(b)", "Sophisticated User"))
    for ax, (col, label, name) in zip(top, panels, strict=True):
        im = _heatmap(ax, t, col, norm, 6.2)
        ax.set_title(label, loc="left", pad=2)
        ax.set_title(name, loc="center", pad=2)
    top[0].set_ylabel(r"$\mu$   (share able)", labelpad=1)
    top[1].tick_params(labelleft=False)
    bar = fig.colorbar(im, ax=top, fraction=0.030, pad=0.02)
    bar.set_label("% of the gain from trade lost", fontsize=6.4)
    bar.ax.tick_params(labelsize=6.0)
    bar.ax.axhline(100, color=INK, lw=1.5)

    share = _composition(t)
    ys = np.arange(len(share))
    bottom.barh(ys, share, height=0.74, color=BLUE)
    bottom.barh(ys, 100 - share, height=0.74, left=share, color=ORANGE)
    for i, v in enumerate(share):
        bottom.text(v - 1.2, i, f"{v:.0f}", ha="right", va="center", fontsize=6.2, color="white")
    bottom.set_yticks(ys, [f"{h:g}" for h in share.index])
    bottom.set(ylabel=r"$h$", xlabel="share of the total loss (%)", xlim=(0, 100))
    bottom.text(2.0, 0, "information destruction", ha="left", va="center", fontsize=6.0, color="white")
    bottom.text(float(share.iloc[0]) + 2.0, 0, "exploitation", ha="left", va="center", fontsize=6.0, color="white")
    bottom.set_title("(c)", loc="left", pad=2)
    finish(bottom, "x")
    return save(fig, path, tight=False)
