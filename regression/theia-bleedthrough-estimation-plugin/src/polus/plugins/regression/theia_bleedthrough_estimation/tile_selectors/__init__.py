"""Automated tile-selectors for Theia Bleedthrough Estimation plugin."""

import enum
import pathlib
import typing

import numpy
import scipy.stats

from .selector import Selector
from .selector import TileIndices


class Entropy(Selector):
    """Select tiles with the highest entropy."""

    def _score_tile(self, tile: numpy.ndarray) -> float:
        counts, _ = numpy.histogram(tile.flat, bins=128, density=True)
        return float(scipy.stats.entropy(counts))

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes an Entropy tile selector."""
        super().__init__(files, num_tiles_per_channel)


class MeanIntensity(Selector):
    """Select tiles with the highest mean intensity."""

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.mean(tile))

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes a MeanIntensity tile selector."""
        super().__init__(files, num_tiles_per_channel)


class MedianIntensity(Selector):
    """Select tiles with the highest median intensity."""

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.median(tile))

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes a MedianIntensity tile selector."""
        super().__init__(files, num_tiles_per_channel)


class IntensityRange(Selector):
    """Select tiles with the largest 90-10 percentile intensity difference."""

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.percentile(tile, 90) - numpy.percentile(tile, 10))

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes an IntensityRange tile selector."""
        super().__init__(files, num_tiles_per_channel)


SELECTORS: dict[str, type[Selector]] = {
    "Entropy": Entropy,
    "MeanIntensity": MeanIntensity,
    "MedianIntensity": MedianIntensity,
    "IntensityRange": IntensityRange,
}
"""A Dictionary to let us use a Selector by name."""


class Selectors(str, enum.Enum):
    """Enum of selectors for the Theia Bleedthrough Estimation plugin."""

    Entropy = "Entropy"
    MeanIntensity = "MeanIntensity"
    MedianIntensity = "MedianIntensity"
    IntensityRange = "IntensityRange"

    def __call__(self) -> type[Selector]:
        """Returns the selector class for this enum value."""
        return SELECTORS[self.value]


__all__ = [
    "TileIndices",
    "Entropy",
    "IntensityRange",
    "MeanIntensity",
    "MedianIntensity",
    "Selectors",
    "SELECTORS",
]
