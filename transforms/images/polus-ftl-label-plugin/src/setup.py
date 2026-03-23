"""Build configuration for the ftl Cython extension."""
import os

import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import setup

Options.annotate = True

os.environ["CFLAGS"] = "-march=haswell -O3"
os.environ["CXXFLAGS"] = "-march=haswell -O3"
setup(
    ext_modules=cythonize("ftl.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
