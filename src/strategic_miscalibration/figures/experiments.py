"""Figures of Section 5 and its appendix: what the agent does, against the theory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ..analysis import theory_fit
from .style import AMBER, BLUE, GREEN, INK, INK_2, INK_3, ORANGE, VIOLET, finish, save

__all__ = ["coherence", "glyph"]


def coherence(path: Path) -> Path:
    """Figure 2. (a) the 100 states on the ``(σ⁺, σ⁻)`` square; (b) ``σ⁻`` against each parameter;
    (c) ``σ⁻`` against δ on ``τR`` and ``τF`` with the equilibrium of Thm 4.3."""
    g = theory_fit.state_table("ledger")
    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.55), gridspec_kw={"width_ratios": [1.0, 1.30, 1.10], "wspace": 0.34}
    )

    ax = axes[0]
    hs = sorted(g.h.unique())
    cmap = plt.get_cmap("viridis")
    for i, h in enumerate(hs):
        s = g[g.h == h]
        ax.scatter(
            s.sigma_plus,
            s.sigma_minus,
            color=cmap(i / max(len(hs) - 1, 1)),
            s=9,
            linewidth=0,
            alpha=0.95,
            zorder=3,
            label=f"{h:g}",
        )
    ax.scatter([1.0], [0.0], marker="*", s=60, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=5)
    ax.annotate("honest", (1.0, 0.0), textcoords="offset points", xytext=(-3, 6), fontsize=5.8, ha="right", color=INK)
    ax.set(xlabel=r"$\sigma^+$", ylabel=r"$\sigma^-$", xlim=(-0.05, 1.05), ylim=(-0.05, 1.13), xticks=[0, 0.5, 1.0])
    ax.legend(
        title=r"$h$",
        loc="lower left",
        fontsize=5.2,
        title_fontsize=5.6,
        handletextpad=0.1,
        labelspacing=0.1,
        borderpad=0.22,
        ncol=2,
        columnspacing=0.5,
    )
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    for coord, colour, marker, label in (
        ("delta", BLUE, "o", r"$\delta$  (myopia)"),
        ("h", ORANGE, "s", r"$h$  (trust)"),
        ("mu", GREEN, "^", r"$\mu$  (ability belief)"),
    ):
        m = g.groupby(coord).sigma_minus.agg(["mean", "sem"])
        ax.errorbar(
            m.index,
            m["mean"],
            yerr=m["sem"],
            marker=marker,
            markersize=4.0,
            color=colour,
            linewidth=1.4,
            capsize=2,
            label=label,
        )
    ax.set(xlabel="parameter value", ylabel=r"$\sigma^-$", xlim=(0, 1), ylim=(0.33, 0.76))
    ax.legend(loc="upper left", fontsize=5.8, handletextpad=0.4, labelspacing=0.22, borderpad=0.25)
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    d_star = theory_fit.DELTA_STAR
    ax.axvline(d_star, color=INK_3, linewidth=0.7, linestyle=":", zorder=1)
    ax.annotate(
        rf"$\delta^*\!={d_star:.2f}$",
        (d_star, 0.955),
        textcoords="offset points",
        xytext=(2, 0),
        fontsize=5.6,
        ha="left",
        va="top",
        color=INK_2,
    )
    fragile_eq = g[(g.region == "F") & (g.delta < d_star)].groupby(["h", "mu"]).sigma_eq.first().mean()
    ax.plot([0.0, 0.75], [1.0, 1.0], color=VIOLET, linewidth=1.0, linestyle="--", zorder=2)
    ax.plot(
        [0.0, d_star, d_star, 0.75],
        [fragile_eq, fragile_eq, 1.0, 1.0],
        color=AMBER,
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )
    for region, colour, marker, name in (
        ("R", VIOLET, "o", r"robust $\tau_C^R$"),
        ("F", AMBER, "s", r"fragile $\tau_C^F$"),
    ):
        per = g[g.region == region].pivot_table(index=["h", "mu"], columns="delta", values="sigma_minus")
        m, se = per.mean(), per.std(ddof=1) / np.sqrt(len(per))
        ax.errorbar(
            m.index,
            m.values,
            yerr=se.values,
            marker=marker,
            markersize=3.6,
            color=colour,
            linewidth=1.4,
            capsize=2,
            zorder=3,
        )
        ax.annotate(
            name,
            (m.index[-1], m.values[-1]),
            textcoords="offset points",
            xytext=(4, -2),
            fontsize=5.8,
            color=INK,
            va="center",
        )
    ax.plot([], [], color=INK_2, linewidth=1.0, linestyle="--", label="equilibrium")
    ax.plot([], [], color=INK_2, linewidth=1.4, marker="o", markersize=3, label="measured")
    ax.set(
        xlabel=r"$\delta$  (myopia)",
        ylabel=r"$\sigma^-$",
        xlim=(0.0, 0.86),
        xticks=[0.05, 0.25, 0.45, 0.65],
        ylim=(0.3, 1.06),
    )
    ax.legend(
        loc="lower left", fontsize=5.4, handletextpad=0.3, labelspacing=0.15, borderpad=0.2, bbox_to_anchor=(0.0, 0.08)
    )
    ax.set_title("(c)", loc="left")
    for ax in axes:
        finish(ax)
    return save(fig, path)


BARS = (
    ("sigma_plus", r"$\sigma^+$", "#1f4e79"),
    ("sigma_minus", r"$\sigma^-$", "#6baed6"),
    ("d_high", r"$d^+$", "#b35806"),
    ("d_low", r"$d^-$", "#fdae61"),
)


def glyph(path: Path, delta: float = 0.65) -> Path:
    """Figure 4: at each belief state, the measured ``σ±`` and the agent's ``d̂(±)`` as bars, against the values
    the joint-law equilibrium correspondence admits (ticks); a gray band where ``σ`` is unrestricted."""
    t = theory_fit.glyph_table(delta)
    hs, mus = sorted(t.h.unique()), sorted(t.mu.unique())
    bw, bh = 0.80 * (hs[1] - hs[0]), 0.80 * (mus[1] - mus[0])
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    for r in t.itertuples():
        left, bottom = r.h - bw / 2, r.mu - bh / 2
        ax.add_patch(Rectangle((left, bottom), bw, bh, facecolor="none", edgecolor="#cfcdc7", lw=0.5, zorder=0))
        for i, (col, _label, colour) in enumerate(BARS):
            cx = left + (i + 0.5) / len(BARS) * bw
            if r.sigma_free and col.startswith("sigma"):
                ax.add_patch(
                    Rectangle(
                        (cx - 0.095 * bw, bottom),
                        0.19 * bw,
                        bh,
                        facecolor="0.65",
                        alpha=0.25,
                        edgecolor="none",
                        zorder=1,
                    )
                )
            v = getattr(r, col)
            if not np.isnan(v):
                ax.add_patch(
                    Rectangle(
                        (cx - 0.055 * bw, bottom), 0.11 * bw, v * bh, facecolor=colour, edgecolor="none", zorder=3
                    )
                )
            for tick in r.ticks[col]:
                ax.plot(
                    [cx - 0.10 * bw, cx + 0.10 * bw],
                    [bottom + tick * bh] * 2,
                    color=INK,
                    lw=1.0,
                    zorder=5,
                    solid_capstyle="butt",
                )
    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(0, 1),
        xticks=hs,
        yticks=mus,
        xlabel=r"$h$   (honesty belief)",
        ylabel=r"$\mu$   (ability belief)",
    )
    legend: list[Artist] = [Patch(facecolor=c, label=label) for _col, label, c in BARS]
    legend += [Line2D([0], [0], color=INK, lw=1.2, label="equilibrium value")]
    if t.sigma_free.any():
        legend += [Patch(facecolor="0.65", alpha=0.25, label=r"$\sigma$ unrestricted")]
    ax.legend(
        handles=legend,
        fontsize=6.4,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        handlelength=1.1,
        handleheight=0.8,
        labelspacing=0.35,
        borderpad=0.3,
    )
    fig.subplots_adjust(left=0.075, right=0.845, top=0.97, bottom=0.15)
    return save(fig, path)
