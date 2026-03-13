"""FTL Label Tool."""
import logging
import platform
from pathlib import Path

from setuptools import Extension
from setuptools import setup
from setuptools_rust import Binding
from setuptools_rust import RustExtension

# Optional Cython imports
try:
    import numpy
    from Cython.Build import cythonize
    from Cython.Compiler import Options
except ImportError:
    numpy = None
    cythonize = None
    Options = None

logger = logging.getLogger(__name__)
#  Detect architecture
machine = platform.machine().lower()
IS_X86 = machine in ("x86_64", "amd64", "i686", "i386")

SRC = Path("src/polus/images/transforms/images/ftl_label")

ext_modules = []

# Cython extension (x86/x64 only)
if IS_X86:
    try:
        Options.annotate = True

        cython_exts = cythonize(
            Extension(
                name="ftl",
                sources=[str(SRC / "ftl.pyx")],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-march=native", "-O3"],
                extra_link_args=["-O3"],
                language="c++",
            ),
            compiler_directives={"language_level": 3},
        )
        ext_modules.extend(cython_exts)
        logger.info("[setup.py] Cython found - compiling ftl from ftl.pyx")

    except ImportError as exc:
        logger.info(f"[setup.py] WARNING: Cython build skipped ({exc}).")


#  Rust/PyO3 extension (all platforms)
rust_ext = RustExtension(
    target="ftl_rust.ftl_rust",
    path="Cargo.toml",
    binding=Binding.PyO3,
    debug=False,
)

# Setup
setup(
    rust_extensions=[rust_ext],
    ext_modules=ext_modules,
    zip_safe=False,
)
