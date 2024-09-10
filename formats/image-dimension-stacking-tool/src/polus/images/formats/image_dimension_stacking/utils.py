"""Helpers for the tool."""

import enum
import logging
import os

import bfio
import numpy

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.zarr")
TILE_SIZE = 1024


class StackableAxis(str, enum.Enum):
    """Axis along which images can be stacked."""

    Z = "z"
    C = "c"
    T = "t"

    def read_tile(
        self,
        reader: bfio.BioReader,
        x: tuple[int, int],
        y: tuple[int, int],
    ) -> numpy.ndarray:
        """Read a tile from the reader."""
        if self == StackableAxis.Z:
            return reader[y[0] : y[1], x[0] : x[1], 0, :, :]
        if self == StackableAxis.C:
            return reader[y[0] : y[1], x[0] : x[1], :, 0, :]
        if self == StackableAxis.T:
            return reader[y[0] : y[1], x[0] : x[1], :, :, 0]
        return None

    def write_tile(  # noqa: PLR0913
        self,
        writer: bfio.BioWriter,
        x: tuple[int, int],
        y: tuple[int, int],
        tile: numpy.ndarray,
        index: int,
    ) -> None:
        """Write a tile to the writer."""
        if self == StackableAxis.Z:
            writer[y[0] : y[1], x[0] : x[1], index, :, :] = tile
        if self == StackableAxis.C:
            writer[y[0] : y[1], x[0] : x[1], :, index, :] = tile
        if self == StackableAxis.T:
            writer[y[0] : y[1], x[0] : x[1], :, :, index] = tile


def make_logger(name: str) -> logging.Logger:
    """Create a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(POLUS_LOG)
    return logger
