"""Section 6 (Welfare Impact): the figure and every number in the text, from the shipped runs.

    uv run python scripts/section6.py [--out DIR]

Writes DIR/figure_3.pdf (default DIR: out/section6) and prints each number next to the value the paper
prints. Offline; takes a few seconds.

Data: data/runs/ledger_full, the reporting rule measured in Section 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from strategic_miscalibration.analysis import welfare
from strategic_miscalibration.core.env import output_dir
from strategic_miscalibration.figures import welfare as welfare_figures


def show(title: str, rows: list[tuple[str, float, float, int]]) -> None:
    """Print ``(what, paper value, regenerated value, decimal places)`` rows."""
    print(f"\n{title}\n{'':56s}{'paper':>12s}{'regenerated':>14s}")
    for what, paper, got, places in rows:
        print(f"  {what:54s}{float(paper):>12.{places}f}{float(got):>14.{places}f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: out/section6)")
    out = ap.parse_args().out or output_dir() / "section6"

    print("Figure 3 ->", welfare_figures.welfare_body(out / "figure_3.pdf"))

    summary = welfare.summary()
    show(
        "Measuring welfare harm",
        [("naive user's loss (% of the gains from trade)", 68, summary["naive_loss_pct"], 0)],
    )
    show(
        "Decomposing the loss",
        [
            ("information destruction (% of the loss)", 71, summary["destruction_share_pct"], 0),
            ("  at h = 0.1", 59, summary["destruction_share_by_h"][0.1], 0),
            ("  at h = 0.9", 95, summary["destruction_share_by_h"][0.9], 0),
        ],
    )

    capability = welfare.capability_readings()
    show(
        "The effect of scaling capability",
        [
            ("able share μ* below which the naive user loses", 0.64, capability["breakeven_mu"], 2),
            ("naive loss at μ* = 0 (%)", 562, capability["naive_loss_pct_mu0"], 0),
            ("naive loss at μ* = 1 (%)", 36, capability["naive_loss_pct_mu1"], 0),
            ("μ* above which naive beats sophisticated", 0.73, capability["rule_knowing_below_naive_from"], 2),
        ],
    )


if __name__ == "__main__":
    main()
