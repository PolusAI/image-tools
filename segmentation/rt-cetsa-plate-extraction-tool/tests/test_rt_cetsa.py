"""Tests on some real data."""

import os
import pathlib

from polus.images.segmentation.rt_cetsa_plate_extraction.__main__ import (
    main as extract_plates,
)

INP_DATA_DIR = os.environ.get(
    "TEST_DATA_DIR",
    "/home/nishaq/Documents/axle/data/Data for Nick/20210318 LDHA compound plates/20210318 LDHA compound plate 1 6K cells",
)


def get_inp_dir() -> pathlib.Path:
    """Return the path to the folder with the input data."""
    inp_dir = pathlib.Path(INP_DATA_DIR).resolve()
    assert inp_dir.is_dir(), f"Input directory not found: {inp_dir}"
    return inp_dir


def get_out_dir() -> pathlib.Path:
    """Return the path to the folder with the output data."""
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    assert data_dir.is_dir(), f"Data directory not found: {data_dir}"
    out_dir = data_dir / "test_1"
    out_dir.mkdir(parents=False, exist_ok=True)
    return out_dir


def test_tool():
    inp_dir = get_inp_dir()
    out_dir = inp_dir.parent / "out-plate-1"
    out_dir.mkdir(parents=False, exist_ok=True)

    # Extract the plate
    extract_plates(
        inp_dir=inp_dir,
        pattern="{id:d+}.tif",
        preview=False,
        out_dir=out_dir,
    )
