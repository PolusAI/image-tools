"""Testing of image dimension stacking."""
from pathlib import Path
from typing import Union

import polus.images.formats.image_dimension_stacking.dimension_stacking as ds
from bfio import BioReader
from bfio import BioWriter

from .conftest import *  # noqa:F403
from .conftest import clean_directories


def test_dimension_stacking(
    synthetic_images: tuple[Union[str, Path], str, str],
    output_directory: Path,
) -> None:
    """Test dimension stacking."""
    inp_dir, variable, pattern = synthetic_images

    ds.dimension_stacking(
        inp_dir=inp_dir,
        file_pattern=pattern,
        group_by=variable,
        out_dir=output_directory,
    )

    outfile = [
        f for f in output_directory.iterdir() if f"{variable}0(0-9).ome.tif" in f.name
    ]
    assert all(outfile) is True
    assert len(outfile) == 1

    total_dimensions = 10

    br = BioReader(outfile[0])
    if variable == "c":
        assert total_dimensions == br.C
    if variable == "z":
        assert total_dimensions == br.Z
    if variable == "t":
        assert total_dimensions == br.T


def test_write_image_stack(
    synthetic_images: tuple[Union[str, Path], str, str],
    output_directory: Path,
) -> None:
    """Test writing stacked images."""
    inp_dir, variable, _ = synthetic_images

    for file in Path(inp_dir).iterdir():
        if file.name.endswith(".ome.tif"):
            with BioReader(file) as br:
                metadata = br.metadata

            with BioWriter(
                output_directory.joinpath(file.name),
                metadata=metadata,
            ) as bw:
                ds.write_image_stack(file=file, di=0, group_by=variable, bw=bw)
    total_dimensions = 10
    assert len(list(output_directory.iterdir())) == total_dimensions


def test_z_distance(synthetic_images: tuple[Union[str, Path], str, str]) -> None:
    """Test estimating z-distance."""
    inp_dir, _, _ = synthetic_images
    distances = []
    for file in Path(inp_dir).iterdir():
        ps_z = ds.z_distance(file=file)
        distances.append(ps_z)
    assert all(distances) is not None
    clean_directories()
