"""Benchmark script for Rust-backed PolygonSet read/write."""
import logging
import time
from pathlib import Path

from ftl_rust import PolygonSet

logger = logging.getLogger(__name__)


def bench_rust() -> None:
    """Time PolygonSet read, digest, and write on a fixed test image."""
    count = 2209
    infile = Path(f"../../data/input_array/test_infile_{count}.ome.tif").resolve()
    outfile = Path(f"../../data/input_array/test_outfile_{count}.ome.tif").resolve()
    polygon_set = PolygonSet(connectivity=1)

    start = time.time()
    polygon_set.read_from(infile)
    end = time.time()
    logger.info("took %.3f seconds to read and digest...", end - start)

    found = len(polygon_set)
    if count != found:
        msg = f"found {found} objects instead of {count}."
        raise ValueError(msg)

    polygon_set.write_to(outfile)
    logger.info("took %.3f seconds to write...", time.time() - end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bench_rust()
