"""Test Command line Tool."""

from pathlib import Path
import polus.images.utils.rxiv_download.fetch as ft
from .conftest import clean_directories
import time
import pytest
from datetime import datetime


def test_fetch_and_save_records(
    output_directory: Path, get_params: pytest.FixtureRequest
) -> None:
    """Test record fetching and saving."""

    start = datetime.strptime(get_params, "%Y-%m-%d").date()

    model = ft.ArxivDownload(path=output_directory, rxiv="arXiv", start=start)
    model.fetch_and_save_records()

    out_ext = all([Path(f.name).suffix for f in output_directory.iterdir()])

    assert out_ext == True

    out_date = [Path(f.name).stem.split("_")[1] for f in output_directory.iterdir()][0]
    assert out_date == "".join(get_params.split("-"))
    clean_directories()
    time.sleep(5)


def test_fetch_records(
    output_directory: Path, get_params: pytest.FixtureRequest
) -> None:
    """Test fetch records."""

    start = datetime.strptime(get_params, "%Y-%m-%d").date()

    model = ft.ArxivDownload(path=output_directory, rxiv="arXiv", start=start)
    response = model.fetch_records()

    assert response != 0
    clean_directories()
    time.sleep(5)
