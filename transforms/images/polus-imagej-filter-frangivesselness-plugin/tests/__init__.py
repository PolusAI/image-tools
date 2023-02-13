from unittest import TestSuite
from .version_test import VersionTest

test_cases = (VersionTest,)


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
