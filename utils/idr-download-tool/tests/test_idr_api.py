"""Test Idr download Tool."""

from pathlib import Path

import polus.images.utils.idr_download.idr_api as od
import pytest

from .conftest import clean_directories


@pytest.mark.skipif("not config.getoption('slow')")
def test_idr_download(
    output_directory: Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test data from Omero Server."""
    data_type, name, object_id = get_params
    model = od.IdrDownload(
        data_type=data_type,
        name=name,
        object_id=object_id,
        out_dir=output_directory,
    )
    model.get_data()
    assert any(output_directory.iterdir()) is True

    clean_directories()
