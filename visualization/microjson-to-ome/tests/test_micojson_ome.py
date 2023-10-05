"""Test Micojson to Ome."""
from pathlib import Path

import numpy as np
from bfio import BioReader
from polus.plugins.visualization.microjson_to_ome.microjson_ome import MicrojsonOmeModel

from tests.fixture import *  # noqa: F403
from tests.fixture import clean_directories


def test_microjsonomemodel(generate_jsondata: Path, output_directory: Path) -> None:
    """Testing of object boundries (rectangle vertices)."""
    inp_dir = generate_jsondata
    for file in Path(inp_dir).iterdir():
        model = MicrojsonOmeModel(
            out_dir=output_directory,
            file_path=file,
        )
        model.convert_microjson_to_ome()
    for outfile in output_directory.iterdir():
        br = BioReader(outfile)
        image = br.read()
        unique_labels = 2
        assert len(np.unique(image)) == unique_labels

    clean_directories()
