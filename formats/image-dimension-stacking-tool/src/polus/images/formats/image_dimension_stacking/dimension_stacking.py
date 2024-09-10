"""Stacking images along a given dimension."""

import pathlib

import bfio

from . import utils

logger = utils.make_logger(__name__)


def write_stack(
    inp_paths: list[pathlib.Path],
    axis: utils.StackableAxis,
    out_path: pathlib.Path,
) -> None:
    """Stack the input images along the given axis.

    This will read all the images from the input directory and stack them along
    the given axis. The output will be written to the output directory.

    Args:
        inp_paths: List of paths to input images. Should be sorted by filepattern.
        axis: Axis to stack images along.
        out_path: Path to output directory.
    """
    logger.info(f"Stacking images along {axis} axis.")
    logger.info(f"Input: {inp_paths}")
    logger.info(f"Output: {out_path}")

    # Get the metadata from the first image
    with bfio.BioReader(inp_paths[0]) as reader:
        metadata = reader.metadata

    # Open all the input images
    readers = [bfio.BioReader(p) for p in inp_paths]

    # Create the output writer
    with bfio.BioWriter(out_path, metadata=metadata) as writer:
        if axis.value == "z":
            writer.Z = len(readers)
        elif axis.value == "c":
            writer.C = len(readers)
        elif axis.value == "t":
            writer.T = len(readers)

        for y_min in range(0, writer.Y, utils.TILE_SIZE):
            y_max = min(writer.Y, y_min + utils.TILE_SIZE)

            for x_min in range(0, writer.X, utils.TILE_SIZE):
                x_max = min(writer.X, x_min + utils.TILE_SIZE)

                # Read the tiles from the input images
                tiles = [
                    axis.read_tile(r, (x_min, x_max), (y_min, y_max)) for r in readers
                ]

                # Write the tiles to the output image
                for i, tile in enumerate(tiles):
                    axis.write_tile(writer, (x_min, x_max), (y_min, y_max), tile, i)

    # Close the readers
    [r.close() for r in readers]
