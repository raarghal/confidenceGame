"""Equilibrium solvers. Importing this package registers every solver it ships."""

from . import binary_t2
from .base import SOLVERS, NoSolverError, Profile, Solver, solve

__all__ = ["SOLVERS", "NoSolverError", "Profile", "Solver", "binary_t2", "solve"]
