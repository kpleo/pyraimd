"""Top-level pytest configuration: slow-test handling.

Slow tests are deselected by default via ``addopts = ["-m", "not slow"]`` in
pyproject.toml; ``--runslow`` clears that marker expression so they run.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow (deselected by default)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks slow tests (deselected by default)")
    if config.getoption("--runslow", default=False):
        # Override the default '-m "not slow"' from pyproject addopts.
        config.option.markexpr = ""
