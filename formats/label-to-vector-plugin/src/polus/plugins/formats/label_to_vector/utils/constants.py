"""Constants for the label_to_vector plugin."""

import logging
import multiprocessing
import os
import typing

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
# TODO: Rename this to POLUS_IMG_EXT
POLUS_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

NUM_THREADS = max(1, int(multiprocessing.cpu_count() // 2))

TILE_SIZE = 2048
TILE_OVERLAP = 64
SUFFIX_LEN = 6

LOG_LEVELS = typing.Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
]
