"""FTL Label Tool."""
import pathlib
import pytest
from typer.testing import CliRunner
from polus.images.transforms.images.ftl_label.__main__ import app
from ftl_rust import PolygonSet

runner = CliRunner()


def run_cli_test(dataset_dir: pathlib.Path, output_dir: pathlib.Path):
    """Helper to run CLI and check outputs."""
    result = runner.invoke(
        app,
        [
            "--inpDir",
            str(dataset_dir),
            "--outDir",
            str(output_dir),
            "--connectivity",
            "1",
        ],
    )
    assert result.exit_code == 0
    # Ensure output directory has results
    assert len(list(output_dir.iterdir())) > 0


def test_cli_ftl_dataset(download_ftl_dataset, output_directory):
    """Run CLI on Hugging Face FTL dataset."""
    run_cli_test(download_ftl_dataset, output_directory)


def test_cli_dsb2018_dataset(download_dsb2018_dataset, output_directory):
    """Run CLI on DSB2018 test masks dataset."""
    run_cli_test(download_dsb2018_dataset, output_directory)


@pytest.mark.skipif(PolygonSet is None, reason="ftl_rust not installed")
def test_bench_rust_dsb2018(
    download_dsb2018_dataset: pathlib.Path, tmp_path: pathlib.Path
):
    """Run PolygonSet read/write on one real .ome.tif from DSB2018 dataset."""
    # Pick the first .ome.tif file from the downloaded dataset
    ome_files = list(download_dsb2018_dataset.glob("*.ome.tif"))
    assert ome_files, "No .ome.tif files found in DSB2018 dataset"
    input_file = ome_files[0]

    # Output file in temporary directory
    output_file = tmp_path / f"{input_file.stem}_out.ome.tif"

    polygon_set = PolygonSet(connectivity=1)

    # Read from real dataset
    polygon_set.read_from(input_file)

    # Write to temp directory
    polygon_set.write_to(output_file)

    # Check that output file exists
    assert output_file.exists()
