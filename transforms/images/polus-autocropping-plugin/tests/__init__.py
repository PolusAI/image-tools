"""Tests for the polus-autocropping-plugin."""
import unittest
from unittest import TestSuite

from .test_autocrop import CorrectnessTest
from .version_test import VersionTest

test_cases = (
    VersionTest,
    CorrectnessTest,
)


def load_tests(
    loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str,
) -> unittest.TestSuite:
    """Build a test suite from loader and test cases."""
    suite = TestSuite()
    for test_class in test_cases:
        loaded = loader.loadTestsFromTestCase(test_class)
        suite.addTests(loaded)
    return suite
