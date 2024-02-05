"""Testing of Ome Converter."""

import pathlib

import numpy as np
import pytest
from bfio import BioReader
from polus.plugins.formats.ome_converter.__main__ import app
from polus.plugins.formats.ome_converter.image_converter import batch_convert
from polus.plugins.formats.ome_converter.image_converter import convert_image
from typer.testing import CliRunner

runner = CliRunner()


def test_batch_converter(
    synthetic_images: tuple[list[np.ndarray], pathlib.Path],
    file_extension: str,
    output_directory: pathlib.Path,
) -> None:
    """Create synthetic image.

    This unit test runs the batch_converter and validates that the converted data is
    the same as the input data.
    """
    image, inp_dir = synthetic_images
    batch_convert(inp_dir, output_directory, ".+", file_extension)
    input_stems = {f.stem for f in inp_dir.iterdir()}
    output_stems = {f.name.split(".")[0] for f in output_directory.iterdir()}

    # Simple check to make sure all input files were converted
    assert input_stems == output_stems
    # # Check to make sure that output files are identical to original data
    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])


@pytest.mark.skipif("not config.getoption('downloads')")
def test_image_converter_omezarr(
    download_images: pathlib.Path,
    file_extension: str,
    output_directory: pathlib.Path,
) -> None:
    """Testing of bioformats supported image datatypes conversion.

    This test will convert the downloaded images to the specified file extension
    and validate that the converted data is the same as the input data.
    """
    br_img = BioReader(download_images)
    image = br_img.read()
    convert_image(download_images, file_extension, output_directory)

    for f in output_directory.iterdir():
        with BioReader(f) as br:
            assert np.all(image) == np.all(br[:])


def test_cli(
    synthetic_images: tuple[list[np.ndarray], pathlib.Path],
    output_directory: pathlib.Path,
    file_extension: str,
) -> None:
    """Test Cli."""
    _, inp_dir = synthetic_images

    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            ".+",
            "--fileExtension",
            file_extension,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
