"""Test plate coordinates."""

from polus.images.features.rt_cetsa_intensity_extraction import PlateSize
from polus.images.features.rt_cetsa_intensity_extraction import alphanumeric_row


def test_alphanumeric_row_48():
    """Test plate coordinates for plate size 48."""
    assert alphanumeric_row(0, 0, PlateSize(48)) == "A1"
    assert alphanumeric_row(5, 7, PlateSize(48)) == "F8"


def test_alphanumeric_row_96():
    """Test plate coordinates for plate size 96."""
    assert alphanumeric_row(0, 0, PlateSize(96)) == "A01"
    assert alphanumeric_row(7, 11, PlateSize(96)) == "H12"


def test_alphanumeric_row_384():
    """Test plate coordinates for plate size 384."""
    assert alphanumeric_row(0, 0, PlateSize(384)) == "A01"
    assert alphanumeric_row(15, 23, PlateSize(384)) == "P24"


def test_alphanumeric_row_1536():
    """Test plate coordinates for plate size 1536."""
    assert alphanumeric_row(0, 0, PlateSize(1536)) == "A01"
    assert alphanumeric_row(31, 47, PlateSize(1536)) == "AF48"
