# noqa

from unittest import TestSuite

from .plugin_test import PluginTest
from .version_test import VersionTest

test_cases = (VersionTest, PluginTest)


def load_tests(loader, tests, pattern):  # noqa
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
