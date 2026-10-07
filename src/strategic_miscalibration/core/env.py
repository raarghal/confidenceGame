"""Where things live: the project root, the data directory, the output directory, secrets.

Nothing in the package resolves a path relative to its own source file, so the package works the
same whether it is run from a checkout or installed. The search order for each location is:
environment variable, then the nearest ancestor of the working directory that contains
``pyproject.toml`` with this project's name.

This module is the only reader of ``.env``.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

__all__ = ["data_dir", "load_secrets", "output_dir", "project_root"]

PROJECT_NAME = "strategic-miscalibration"


@cache
def project_root() -> Path:
    """The checkout this process is running in.

    Raises:
        FileNotFoundError: Outside a checkout and with no ``SMC_ROOT`` set.
    """
    if root := os.environ.get("SMC_ROOT"):
        return Path(root).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and f'name = "{PROJECT_NAME}"' in pyproject.read_text(encoding="utf-8"):
            return candidate
    raise FileNotFoundError(
        f"not inside a {PROJECT_NAME} checkout (searched upward from {here}); set SMC_ROOT to point at one"
    )


def data_dir() -> Path:
    """Directory of packed runs (``$SMC_DATA_DIR``, default ``<root>/data``)."""
    return Path(os.environ.get("SMC_DATA_DIR") or project_root() / "data")


def output_dir() -> Path:
    """Directory for regenerated artifacts and new runs (``$SMC_OUTPUT_DIR``, default ``<root>/out``)."""
    return Path(os.environ.get("SMC_OUTPUT_DIR") or project_root() / "out")


def load_secrets() -> None:
    """Load API keys from the nearest ``.env`` above the working directory, without overriding the shell."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
