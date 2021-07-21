from setuptools import setup
from setuptools_rust import RustExtension


setup(
    name="ftl-rust",
    version="0.1.0",
    packages=["ftl_rust"],
    rust_extensions=[RustExtension("ftl_rust.ftl_rust", "Cargo.toml", debug=False)],
    include_package_data=True,
    zip_safe=False,
)
