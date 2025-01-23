"""Tests for pyramid_generator_3d."""

import pytest

from polus.images.formats.pyramid_generator_3d.pyramid_generator_3d import pyramid_generator_3d


def test_pyramid_generator_3d():
    """Test pyramid_generator_3d."""
    # TODO: Add tests
    pass


@pytest.mark.skipif("not config.getoption('slow')")
def test_slow_pyramid_generator_3d():
    """Test that can take a long time to run."""
    # TODO: Add optional tests
    pass


@pytest.mark.skipif("not config.getoption('downloads')")
def test_download_pyramid_generator_3d():
    """Test thatdownload data from."""
    # TODO: Add optional tests
    pass
