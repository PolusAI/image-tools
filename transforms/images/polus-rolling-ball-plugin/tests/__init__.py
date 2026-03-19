"""Test suite loader for polus-rolling-ball-plugin."""
from unittest import TestLoader
from unittest import TestSuite

from .correctness_test import CorrectnessTest
from .version_test import VersionTest

test_cases = (
    VersionTest,
    CorrectnessTest,
)


def load_tests(
    loader: TestLoader,
    _standard_tests: TestSuite,
    _pattern: str | None,
) -> TestSuite:
    """Build a ``TestSuite`` from this package's test cases."""
    suite = TestSuite()
    for test_class in test_cases:
        loaded = loader.loadTestsFromTestCase(test_class)
        suite.addTests(loaded)
    return suite
