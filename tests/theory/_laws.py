"""Random exact instances shared by the theory audits."""

from __future__ import annotations

import random
from fractions import Fraction as F

from strategic_miscalibration.theory.game import Game

from ..conftest import game_from_prims

CELLS = ((1, "H"), (1, "L"), (0, "H"), (0, "L"))


def frac(rng: random.Random, den: int = 100) -> F:
    """A random fraction strictly inside (0, 1)."""
    return F(rng.randint(1, den - 1), den)


def random_game(rng: random.Random) -> Game:
    """A binary game with random primitives ``θ_L < θ_H`` and ``ρ⁻ < ρ* < ρ⁺``, ``ρ* >= c``."""
    while True:
        tL, tH = sorted(frac(rng) for _ in range(2))
        rm, rp = sorted(frac(rng) for _ in range(2))
        if tL == tH or rm == rp:
            continue
        rs = rm + (rp - rm) * frac(rng)
        if rs > F(1, 10):
            return game_from_prims(tL, tH, rm, rp, rs)


def random_law(rng: random.Random) -> dict:
    """A random joint law on the four cells (generally not a product)."""
    w = {c: frac(rng) for c in CELLS}
    z = sum(w.values())
    return {c: v / z for c, v in w.items()}


def od(p: F) -> F:
    return p / (1 - p)
