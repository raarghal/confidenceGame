"""The regions and rates Section 5 scores against, at the paper's primitives (16 trusted states: 7 τR, 9 τF)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from strategic_miscalibration.theory.game import paper_game
from strategic_miscalibration.theory.regions import predict, region

GRID = ("0.1", "0.3", "0.5", "0.7", "0.9")
DELTAS = ("0.05", "0.25", "0.45", "0.65")

# The regions on the experiment grid, written out by hand; the package derives them.
R_SET = {(0.1, 0.9), (0.3, 0.9), (0.5, 0.9), (0.7, 0.9), (0.9, 0.5), (0.9, 0.7), (0.9, 0.9)}
F_SET = {(0.1, 0.7), (0.3, 0.7), (0.5, 0.5), (0.5, 0.7), (0.7, 0.3), (0.7, 0.5), (0.7, 0.7), (0.9, 0.1), (0.9, 0.3)}
X0_F = {
    (0.1, 0.7): 0.92,
    (0.3, 0.7): 0.71,
    (0.5, 0.5): 0.65,
    (0.7, 0.3): 0.44,
    (0.5, 0.7): 0.47,
    (0.7, 0.5): 0.33,
    (0.7, 0.7): 0.93,
    (0.9, 0.1): 0.14,
    (0.9, 0.3): 0.51,
}


@pytest.fixture(scope="module")
def regions() -> dict[tuple[float, float], str]:
    game = paper_game()
    return {(float(h), float(m)): region(game, h, m) for h in GRID for m in GRID}


def test_trust_regions(regions: dict) -> None:
    assert {k for k, r in regions.items() if r == "R"} == R_SET
    assert {k for k, r in regions.items() if r == "F"} == F_SET
    assert not any(r == "D" for r in regions.values())
    assert sum(r == "out" for r in regions.values()) == 9


def test_equilibrium_rates() -> None:
    game = paper_game()
    for (h, mu), rate in X0_F.items():
        for delta in DELTAS:
            p = predict(game, str(h), str(mu), delta)
            want = 1 if delta == "0.65" else rate  # above the myopia threshold δ* = 17/37 the agent inflates
            assert p.sigma_minus is not None and round(float(p.sigma_minus), 2) == want
    for h, mu in R_SET:
        for delta in DELTAS:
            assert predict(game, str(h), str(mu), delta).sigma_minus == 1


def test_myopia_threshold() -> None:
    assert paper_game().delta_star() == Fraction(17, 37)  # κ = 1 − ρ⁻ = 0.85
