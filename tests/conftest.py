"""Shared test helpers."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from strategic_miscalibration.theory.game import Game, Payoffs, TypeSpace

FIXTURES = Path(__file__).parent / "fixtures"


def game_from_prims(theta_L, theta_H, rho_minus, rho_plus, rho_star, c=Fraction(1, 10)) -> Game:
    """A binary game with a given ``ρ*``: ``r = 1`` and ``e = 1 − ρ* + c`` (so ``ρ* = 1 − (e − c)/r``)."""
    payoffs = Payoffs(Fraction(1), Fraction(c), 1 - Fraction(rho_star) + Fraction(c))
    return Game(TypeSpace.binary(theta_H, theta_L, rho_plus, rho_minus), payoffs)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--theory-draws",
        type=int,
        default=None,
        help="random primitive draws for the exact theory audits (default: a quick setting; the paper quotes 600)",
    )


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def theory_draws(request: pytest.FixtureRequest) -> int | None:
    """``--theory-draws`` if given, else ``None`` (each audit picks its own quick default)."""
    return request.config.getoption("--theory-draws")
