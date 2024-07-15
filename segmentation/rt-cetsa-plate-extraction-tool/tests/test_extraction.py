"""Tests."""

from pathlib import Path
from polus.images.segmentation.rt_cetsa_plate_extraction import extract_plates


def test_extraction():
    """To pacify git actions.

    Original files to use for testing are too large to be shipped as testing data."""
    pass


INP_DIR = Path(
    "/Users/antoinegerardin/RT-CETSA-Analysis/dashboard/data/data_240700/240614_LDHA_01/preprocessed/images"
)
# INP_DIR = Path("/Users/antoinegerardin/RT-CETSA-Analysis/dashboard/data/data_240700/240614_NLuc vs ThermLuc/preprocessed/images")
PATTERN = "{index:d+}_{temp:f+}.tif"
OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)


def test_plate_extraction():
    extract_plates(INP_DIR, PATTERN, OUT_DIR)


test_plate_extraction()
