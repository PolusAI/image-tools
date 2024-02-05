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

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes an Entropy tile selector."""
        super().__init__(files, num_tiles_per_channel)

    def _score_tile(self, tile: numpy.ndarray) -> float:
        counts, _ = numpy.histogram(tile.flat, bins=128, density=True)
        return float(scipy.stats.entropy(counts))


class MeanIntensity(Selector):
    """Select tiles with the highest mean intensity."""

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes a MeanIntensity tile selector."""
        super().__init__(files, num_tiles_per_channel)

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.mean(tile))


class MedianIntensity(Selector):
    """Select tiles with the highest median intensity."""

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes a MedianIntensity tile selector."""
        super().__init__(files, num_tiles_per_channel)

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.median(tile))


class IntensityRange(Selector):
    """Select tiles with the largest 90-10 percentile intensity difference."""

    def __init__(self, files: list[pathlib.Path], num_tiles_per_channel: int) -> None:
        """Initializes an IntensityRange tile selector."""
        super().__init__(files, num_tiles_per_channel)

    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.percentile(tile, 90) - numpy.percentile(tile, 10))


class Selectors(str, enum.Enum):
    """Enum of selectors for the Theia Bleedthrough Estimation plugin."""

    Entropy = "Entropy"
    MeanIntensity = "MeanIntensity"
    MedianIntensity = "MedianIntensity"
    IntensityRange = "IntensityRange"

    def __call__(self) -> type[Selector]:
        """Returns the selector class for this enum value."""
        s: type[Selector]
        if self.value == "Entropy":
            s = Entropy
        elif self.value == "MeanIntensity":
            s = MeanIntensity
        elif self.value == "MedianIntensity":
            s = MedianIntensity
        else:  # self.value == "IntensityRange"
            s = IntensityRange
        return s

    @classmethod
    def variants(cls) -> list["Selectors"]:
        """Returns the list of available selectors."""
        return [cls.Entropy, cls.MeanIntensity, cls.MedianIntensity, cls.IntensityRange]


__all__ = [
    "TileIndices",
    "Entropy",
    "IntensityRange",
    "MeanIntensity",
    "MedianIntensity",
    "Selectors",
]
