"""Pytest configuration for bbbc-download-plugin."""
import warnings

# Avoid DeprecationWarning from jgo/scyjava (bfio deps): maven_scijava_repository()
# is deprecated; the libraries will use the URL directly in jgo 3.0.
warnings.filterwarnings(
    "ignore",
    message=r".*maven_scijava_repository.*",
    category=DeprecationWarning,
)
