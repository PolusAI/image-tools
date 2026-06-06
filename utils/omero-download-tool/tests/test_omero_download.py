"""Test Omero download Tool."""

from pathlib import Path

import polus.images.utils.omero_download.omero_download as od
import pytest

from .conftest import clean_directories


@pytest.mark.skipif("not config.getoption('slow')")
def test_omero_download(
    output_directory: Path,
    get_params: pytest.FixtureRequest,
) -> None:
    """Test data from Omero Server."""
    data_type, name, object_id = get_params
    model = od.OmeroDwonload(
        data_type=data_type,
        name=name,
        object_id=object_id,
        out_dir=output_directory,
    )
    model.get_data()
    assert any(output_directory.iterdir()) is True

    clean_directories()
