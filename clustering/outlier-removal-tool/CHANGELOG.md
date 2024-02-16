# [0.2.6-dev0] - 2024-01-12

## Added

- Pytests to test this plugin
- This plugin is now installable with pip.
- Added support for arrow file format in addition to csv

## Changed

- Updated dependencies (bfio, filepattern, preadator) to latest
- Argparse package is replaced with Typer package for command line arguments
- Replaced docker base image with latest container image with pre-installed bfio
- Replaced pandas with vaex
- Seperating descriptive from numerical features for outlier detection if present in the tabular data
