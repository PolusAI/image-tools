"""Tests for midrc_download."""

from pathlib import Path

import polus.images.utils.midrc_download.midrc_download as md
import pytest

from .conftest import clean_directories


def test_midrc_download_get_query(
    genenerate_dict_params: pytest.FixtureRequest,
) -> None:
    """Test of transforming dictionary parameters into a GraphiQL query."""
    model = md.MIDRIC(**genenerate_dict_params)
    filter_obj = model.get_query(genenerate_dict_params)
    assert len(filter_obj.keys()) == 1
    clean_directories()


def test_midrc_download_query_data(
    genenerate_dict_params: pytest.FixtureRequest,
) -> None:
    """Perform a data query against a MIDRC Data Commons."""
    model = md.MIDRIC(**genenerate_dict_params)
    filter_obj = model.get_query(genenerate_dict_params)

    data = model.query_data(
        midrc_type="imaging_study",
        fields=None,
        filter_object=filter_obj,
        first=1,
    )
    assert len(data) != 0
    assert isinstance(data, list)
    clean_directories()


def test_midrc_download_download_data(
    genenerate_dict_params: pytest.FixtureRequest,
) -> None:
    """Test data download against a Data Commons."""
    model = md.MIDRIC(**genenerate_dict_params)
    filter_obj = model.get_query(genenerate_dict_params)

    data = model.query_data(
        midrc_type="imaging_study",
        fields=None,
        filter_object=filter_obj,
        first=1,
    )
    model.download_data(data=data)
    assert list(Path(genenerate_dict_params["out_dir"]).rglob("*")) != 0
    clean_directories()
