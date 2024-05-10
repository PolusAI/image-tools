from polus.images.features.rt_cetsa_intensity_extraction import PlateSize
from polus.images.features.rt_cetsa_intensity_extraction import index_to_battleship


def test_battleship_coordinates():
    assert index_to_battleship(0, 0, PlateSize(96)) == "A01"
    assert index_to_battleship(12, 8, PlateSize(96)) == "H12"
