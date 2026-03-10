from unittest import TestSuite

from .test_autocrop import CorrectnessTest
from .version_test import VersionTest

test_cases = (
    VersionTest,
    CorrectnessTest,
)


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
