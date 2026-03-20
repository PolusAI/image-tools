"""Constants used by theia_bleedthrough_estimation plugin."""

import logging
import multiprocessing
import os

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")

NUM_THREADS = max(1, int(multiprocessing.cpu_count() // 2))
TILE_SIZE_2D = 1024 * 2
TILE_SIZE_3D = 128
MIN_2D_TILES = 8
MAX_2D_TILES = 16
EPSILON = 1e-8  # To avoid divide-by-zero errors
