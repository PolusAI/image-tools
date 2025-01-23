"""Test fixtures.
"""

import pytest

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--downloads",
        action="store_true",
        dest="downloads",
        default=False,
        help="Run tests that download large data files",
    )
