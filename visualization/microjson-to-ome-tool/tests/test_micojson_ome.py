"""Test Micojson to Ome."""
import shutil
from pathlib import Path

import numpy as np
from bfio import BioReader
from polus.images.visualization.microjson_to_ome.microjson_ome import MicrojsonOmeModel


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


def test_microjsonomemodel(generate_jsondata: Path, output_directory: Path) -> None:
    """Testing of object boundries (rectangle vertices)."""
    inp_dir = generate_jsondata
    for file in Path(inp_dir).iterdir():
        model = MicrojsonOmeModel(
            out_dir=output_directory,
            file_path=file,
        )
        model.microjson_to_ome()
    for outfile in output_directory.iterdir():
        br = BioReader(outfile)
        image = br.read()
        unique_labels = 2
        assert len(np.unique(image)) == unique_labels

    clean_directories()
