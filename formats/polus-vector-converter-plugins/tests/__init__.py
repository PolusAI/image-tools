from unittest import TestSuite

from .label_to_vector_test import LabelToVectorTest
from .vector_to_label_test import VectorToLabelTest
from .version_test import VersionTest

test_cases = (
    VersionTest,
    LabelToVectorTest,
    VectorToLabelTest,
)


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite
