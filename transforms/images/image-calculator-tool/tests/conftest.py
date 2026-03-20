"""Pytest configuration file."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to the pytest command line."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )
