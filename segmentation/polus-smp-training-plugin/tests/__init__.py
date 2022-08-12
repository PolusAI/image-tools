from unittest import TestSuite
from tests.model_test import ModelTest
from tests.tile_test import TileTest

test_cases = [TileTest, ModelTest]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
