"""Test Command line Tool."""
from typer.testing import CliRunner
from polus.plugins.formats.image_dimension_stacking.__main__ import app
from tests.fixture import *


def test_cli(synthetic_images: tuple[Union[str, Path]], output_directory: Path) -> None:
    """Test the command line."""
    inp_dir, _, pattern = synthetic_images

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            pattern,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


def test_multipattern_cli(
    synthetic_multi_images: Union[str, Path], output_directory: Path
) -> None:
    """Test the command line."""
    inp_dir = synthetic_multi_images
    pattern = "tubhiswt_z{z:d+}_c{c:d+}_t{t:d+}.ome.tif"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--inpDir",
            inp_dir,
            "--filePattern",
            pattern,
            "--outDir",
            output_directory,
        ],
    )

    assert result.exit_code == 0
    clean_directories()


def test_short_cli(
    synthetic_images: tuple[Union[str, Path]], output_directory: Path
) -> None:
    """Test the short cli command line."""
    inp_dir, _, pattern = synthetic_images
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "-i",
            inp_dir,
            "-f",
            pattern,
            "-o",
            output_directory,
        ],
    )

    assert result.exit_code == 0


clean_directories()
