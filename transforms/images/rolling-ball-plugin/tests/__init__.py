"""Unit Tests."""

from unittest import TestSuite

from .correctness_test import CorrectnessTest
from .version_test import VersionTest

test_cases = (
    VersionTest,
    CorrectnessTest,
)


def load_tests(loader, tests, pattern):
    """Load tests."""
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
