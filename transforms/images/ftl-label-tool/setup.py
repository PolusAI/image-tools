"""Build script for ftl-label native extensions.

Architecture handling
---------------------
ftl (Cython)     – uses x86/x64 AVX2+BMI2 intrinsics (x86intrin.h).
                   Compiled ONLY on x86_64. Skipped silently on ARM64 / Apple Silicon.

ftl_rust (Rust)  – pure Rust with Rayon; compiles on all platforms including ARM64.

When ftl is unavailable (ARM64), main.py routes ALL images through the Rust path.
"""

import os
import platform
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# ── Detect architecture ────────────────────────────────────────────────────────
machine = platform.machine().lower()
IS_X86 = machine in ("x86_64", "amd64", "i686", "i386")

SRC = Path("src/polus/images/transforms/images/ftl_label")

ext_modules = []

# ── Cython extension (x86/x64 only) ───────────────────────────────────────────
if IS_X86:
    try:
        import numpy
        from Cython.Build import cythonize
        from Cython.Compiler import Options

        Options.annotate = True
        os.environ["CFLAGS"] = "-march=native -O3"
        os.environ["CXXFLAGS"] = "-march=native -O3"

        cython_exts = cythonize(
            str(SRC / "ftl.pyx"),
            compiler_directives={"language_level": 3},
        )
        # Override the deep dotted path Cython infers → plain "ftl"
        for ext in cython_exts:
            ext.name = "ftl"
            ext.include_dirs = [numpy.get_include()]

        ext_modules.extend(cython_exts)
        print(f"[setup.py] x86_64 detected – Cython extension will be compiled.")

    except Exception as exc:  # noqa: BLE001
        print(f"[setup.py] WARNING: Cython build skipped ({exc}).")
else:
    print(
        f"[setup.py] Non-x86 architecture detected ({machine}) – "
        "Cython AVX extension is not supported here.\n"
        "           All images will be processed via the Rust backend."
    )

# ── Rust/PyO3 extension (all platforms) ───────────────────────────────────────
rust_ext = RustExtension(
    target="ftl_rust.ftl_rust",
    path="Cargo.toml",
    binding=Binding.PyO3,
    debug=False,
)

# ── Setup ──────────────────────────────────────────────────────────────────────
setup(
    rust_extensions=[rust_ext],
    ext_modules=ext_modules,
    zip_safe=False,
)