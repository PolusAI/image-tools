"""Tiled rolling-ball background subtraction using scikit-image."""
from __future__ import annotations

import numpy as np
from bfio.bfio import BioReader
from bfio.bfio import BioWriter
from skimage import restoration
from skimage import util

# The number of pixels to be saved at a time must be a multiple of 1024.
TILE_SIZE = 1024


def _rolling_ball(
    tile: np.ndarray,
    ball_radius: int,
    light_background: bool,
) -> np.ndarray:
    """Apply rolling-ball to a single tile.

    Args:
        tile: A tile, usually from an ome.tif file.
        ball_radius: Radius of the ball for background estimation.
        light_background: Whether the image has a light background.

    Returns:
        Image with background subtracted.
    """
    # Get the shape of the original image, so we can reshape the result at the end.
    shape = np.shape(tile)

    # squeeze the image into a 2-d array
    tile = np.squeeze(tile)

    # invert the image if it has a light background
    if light_background:
        tile = util.invert(tile)

    # rolling ball background, then subtract from the image
    background = restoration.rolling_ball(tile, radius=ball_radius)
    tile = tile - background

    # if the image had a light background, invert the result.
    result = util.invert(tile) if light_background else tile

    return np.reshape(result, shape)


def _bounds(
    x: int,
    x_max: int,
    ball_radius: int,
) -> tuple[int, int, int, int, int]:
    """Compute tile and padding indices along one axis.

    Each tile is padded with up to ``ball_radius`` pixels from the full image along
    the edges.

    Args:
        x: Start index along the axis.
        x_max: Image extent along the axis.
        ball_radius: Ball radius used for padding.

    Returns:
        ``row_max, pad_left, pad_right, tile_left, tile_right``
    """
    row_max = min(x_max, x + TILE_SIZE)
    pad_left = max(0, x - ball_radius)
    pad_right = min(x_max, row_max + ball_radius)

    tile_left = 0 if x == 0 else ball_radius
    tile_right = min(x_max, tile_left + TILE_SIZE)
    return row_max, pad_left, pad_right, tile_left, tile_right


def rolling_ball(
    reader: BioReader,
    writer: BioWriter,
    ball_radius: int,
    light_background: bool,
) -> None:
    """Apply rolling-ball per Z slice, tiled in XY, and write to ``writer``.

    Processes the image in tiles so it scales to large images.

    Args:
        reader: Source image reader.
        writer: Destination writer (metadata should match reader).
        ball_radius: Rolling-ball radius; should exceed object radii of interest.
        light_background: Whether the scene has a light background.
    """
    for z in range(reader.Z):
        for y in range(0, reader.Y, TILE_SIZE):
            y_max, pad_top, pad_bottom, tile_top, tile_bottom = _bounds(
                y,
                reader.Y,
                ball_radius,
            )

            for x in range(0, reader.X, TILE_SIZE):
                x_max, pad_left, pad_right, tile_left, tile_right = _bounds(
                    x,
                    reader.X,
                    ball_radius,
                )

                tile = reader[pad_top:pad_bottom, pad_left:pad_right, z : z + 1, 0, 0]
                result = _rolling_ball(tile, ball_radius, light_background)
                writer[y:y_max, x:x_max, z : z + 1, 0, 0] = result[
                    tile_top:tile_bottom,
                    tile_left:tile_right,
                ]
