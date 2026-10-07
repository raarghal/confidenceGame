"""Section 5 (LLM Experiments): the figure and every number in the text, from the shipped runs.

    uv run python scripts/section5.py [--out DIR]

Writes DIR/figure_2.pdf (default DIR: out/section5) and prints each number next to the value the paper
prints. Offline; takes a few seconds.

Data: data/runs/ledger_full (the toy setting) and data/runs/mathqa_* (math Q&A).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from strategic_miscalibration.analysis import theory_fit, transfer
from strategic_miscalibration.core.env import output_dir
from strategic_miscalibration.figures import experiments
from strategic_miscalibration.theory.game import paper_game


def show(title: str, rows: list[tuple[str, object, object, int]]) -> None:
    """Print ``(what, paper value, regenerated value, decimal places)`` rows; intervals are tuples."""

    def fmt(v: object, places: int) -> str:
        if isinstance(v, tuple):
            return fmt(v[0], places) + " [" + ", ".join(fmt(x, places) for x in v[1:]) + "]"
        return f"{float(v):.{places}f}"

    print(f"\n{title}\n{'':56s}{'paper':>20s}{'regenerated':>20s}")
    for what, paper, got, places in rows:
        print(f"  {what:54s}{fmt(paper, places):>20s}{fmt(got, places):>20s}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: out/section5)")
    out = ap.parse_args().out or output_dir() / "section5"

    print("Figure 2 ->", experiments.coherence(out / "figure_2.pdf"))

    game = paper_game()
    levels = theory_fit.levels()
    show(
        "Design",
        [
            ("trusted belief states", 16, levels["robust"] + levels["fragile"], 0),
            ("  in the robust region τR", 7, levels["robust"], 0),
            ("  in the fragile region τF", 9, levels["fragile"], 0),
            ("myopia threshold δ*", 0.46, game.delta_star(), 2),
        ],
    )
    show(
        "Severity and direction of manipulation",
        [
            ("σ⁻, mean over the 100 states", 0.560, levels["sigma_minus"], 3),
            ("σ⁺, mean over the 100 states", 0.983, levels["sigma_plus"], 3),
            ("states with σ⁻ ≥ 0.5", 75, levels["states_ge_half"], 0),
            ("states with σ⁻ > 0.5", 60, levels["states_gt_half"], 0),
            ("max σ⁻", 0.900, levels["max_minus"], 3),
            ("s.d. of σ⁺ across states", 0.035, levels["sd_plus"], 3),
            ("s.d. of σ⁻ across states", 0.139, levels["sd_minus"], 3),
        ],
    )

    myopia = theory_fit.myopia_response()
    show(
        "Comparison with the equilibrium",
        [
            ("σ⁻ where (1,1) is the unique equilibrium", 0.716, levels["restricted_level"], 3),
            ("change in σ⁻ below the threshold, τF", -0.039, myopia["F"]["below"][0], 3),
            ("change in σ⁻ below the threshold, τR", 0.033, myopia["R"]["below"][0], 3),
            ("jump in σ⁻ across the threshold, τF", 0.124, myopia["F"]["jump"][0], 3),
            ("jump in σ⁻ across the threshold, τR", 0.158, myopia["R"]["jump"][0], 3),
        ],
    )

    mediation = theory_fit.belief_mediation()
    stated = theory_fit.stated_continuation()
    show(
        "Why the agent departs",
        [
            ("reports that maximize the agent's own stated payoff", 0.952, theory_fit.coherence()["share"], 3),
            ("d̂(+) where the user delegates for certain", 0.678, mediation["dhat_high_where_true"], 3),
            ("h slope", 0.152, mediation["base"]["h"][0], 3),
            ("h slope, controlling for d̂(+)", 0.027, mediation["with_dhat"]["h"][0], 3),
            ("μ slope", 0.142, mediation["base"]["mu"][0], 3),
            ("μ slope, controlling for d̂(+)", 0.017, mediation["with_dhat"]["mu"][0], 3),
            ("stated next-round payoff / fee after HIGH, τR", (0.44, 0.37, 0.50), stated.loc["R", "stated_hi"], 2),
            ("stated next-round payoff / fee after LOW, τR", (0.44, 0.39, 0.50), stated.loc["R", "stated_lo"], 2),
            ("jump on τR from the stated values", (0.17, 0.14, 0.20), theory_fit.model_jumps()["R"]["M2'"], 2),
            ("jump on τR, measured", 0.16, myopia["R"]["jump"][0], 2),
        ],
    )

    mq = transfer.levels()
    calibration = transfer.calibration()
    show(
        "Calibration on real tasks (math Q&A)",
        [
            ("baseline confidence", 0.822, mq["confidence"][0], 3),
            ("baseline accuracy", 0.726, mq["accuracy"][0], 3),
            ("overconfidence, baseline", 0.096, calibration["overconfidence_plain"][0], 3),
            ("overconfidence, in game", 0.192, calibration["overconfidence_game"][0], 3),
            ("confidence gap correct − incorrect, baseline", 0.285, calibration["discrimination_plain"][0], 3),
            ("confidence gap correct − incorrect, in game", 0.139, calibration["discrimination_game"][0], 3),
            ("accuracy of reports ≥ 0.90, baseline (%)", 86.8, 100 * mq["top_plain"]["accuracy"], 1),
            ("accuracy of reports ≥ 0.90, in game (%)", 74.0, 100 * mq["top_game"]["accuracy"], 1),
        ],
    )


if __name__ == "__main__":
    main()
