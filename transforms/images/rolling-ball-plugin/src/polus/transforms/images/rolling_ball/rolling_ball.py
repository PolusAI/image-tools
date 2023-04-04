import numpy
from bfio.bfio import BioReader
from bfio.bfio import BioWriter
from skimage import restoration
from skimage import util

# The number of pixels to be saved at a time must be a multiple of 1024.
TILE_SIZE = 1024


def _rolling_ball(tile, ball_radius: int, light_background: bool):
    """ Applies the rolling-ball algorithm to a single tile.

    Args:
        tile: A tile, usually from an ome.tif file.
        ball_radius: The radius of the ball to use for calculating the background.
        light_background: Whether the image has a light background.

    Returns:
        An image with its background subtracted away.
    """
    # Get the shape of the original image, so we can reshape the result at the end.
    shape = numpy.shape(tile)

    # squeeze the image into a 2-d array
    tile = numpy.squeeze(tile)

    # invert the image if it has a light background
    if light_background:
        tile = util.invert(tile)

    # use the rolling ball algorithm to calculate the background and subtract it from the image.
    background = restoration.rolling_ball(tile, radius=ball_radius)
    tile = tile - background

    # if the image had a light backend, invert the result.
    result = util.invert(tile) if light_background else tile

    result = numpy.reshape(result, shape)
    return result


def _bounds(x, x_max, ball_radius):
    """ Calculates the indices for handling the edges of tiles.

    We pad each tile with 'ball_radius' pixels from the full image along the
     top, bottom, left, and right edges of each tile.
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
):
    """ Applies the rolling-ball algorithm from skimage to perform background subtraction.

    This function processes the image in tiles and, therefore, scales to images of any size.
    It writes the resulting image to the given BioWriter object.

    Args:
        reader: BioReader object from which to read the image.
        writer: BioWriter object to which to write the image.
        ball_radius: The radius of the ball to use for calculating the background.
                     This should be greater than the radii of relevant objects in the image.
        light_background: Whether the image has a light background.

    """
    for z in range(reader.Z):

        for y in range(0, reader.Y, TILE_SIZE):
            y_max, pad_top, pad_bottom, tile_top, tile_bottom = _bounds(y, reader.Y, ball_radius)

            for x in range(0, reader.X, TILE_SIZE):
                x_max, pad_left, pad_right, tile_left, tile_right = _bounds(x, reader.X, ball_radius)

                tile = reader[pad_top:pad_bottom, pad_left:pad_right, z:z + 1, 0, 0]
                result = _rolling_ball(tile, ball_radius, light_background)
                writer[y:y_max, x:x_max, z:z + 1, 0, 0] = result[tile_top:tile_bottom, tile_left:tile_right]
    return
