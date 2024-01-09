"""Cell border segmentation package."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from sys import platform
from typing import Any

import numpy as np
import tensorflow as tf
from bfio import BioReader
from bfio import BioWriter

# Set the environment variable to prevent odd warnings from tensorflow


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

if platform == "linux" or platform == "linux2":
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    NUM_THREADS = max(cpu_count() // 2, 2)


# Define preprocessing and neural network params
WINDOW_SIZE = 127
MAX_NORM = 6
TILE_SIZE = 1024
OUTPUT_TILE = 1200


class ReflectionPadding2D(tf.keras.layers.Layer):
    """ReflectionPadding2D Custom class to handle matconvnet padding.

    This class is a Keras layer that does reflection padding, which
    is the default method of padding in matconvnet pooling operations.

    Modified from the following:
    https://stackoverflow.com/a/60116269

    """

    def __init__(self, padding: tuple[tuple[int, int], ...], **kwargs: int) -> None:
        """Define class instance attributes."""
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super().__init__(**kwargs)

    def compute_output_shape(self, s: np.ndarray) -> tuple[None, None, None, Any]:
        """Compute shape of padded image."""
        if s[1] is None:
            return (None, None, None, s[3])
        return (
            s[0],
            s[1] + self.padding[0][0] + self.padding[0][1],
            s[2] + self.padding[1][0] + self.padding[1][1],
            s[3],
        )

    def call(self, x: np.ndarray) -> np.ndarray:
        """Padding of an image."""
        (top_pad, bottom_pad), (left_pad, right_pad) = self.padding
        return tf.pad(
            x,
            [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]],
            "REFLECT",
        )

    def get_config(self) -> None:
        """Retrieve values for configuration."""
        return super().get_config()


def imboxfilt(image: np.ndarray, window_size: int) -> np.ndarray:
    """Imboxfilt Use a box filter on a stack of images.

    This method applies a box filter to an image. The input is assumed
    to be a 4D array, and should be pre-padded. The output will be smaller
    by window_size-1 pixels in both width and height since this filter does
    not pad the input to account for filtering.

    Args:
        image: A 4d array of images
        window_size: An odd integer window size

    Returns:
        A filtered array of images.
    """
    if window_size % 2 == 0:
        msg = "window_size must be an odd integer"
        raise ValueError(msg)
    # Generate an integral image
    image_ii = image.cumsum(1).cumsum(2)

    # Create the output
    return (
        image_ii[:, 0:-window_size, 0:-window_size, :]
        + image_ii[:, window_size:, window_size:, :]
        - image_ii[:, window_size:, 0:-window_size, :]
        - image_ii[:, 0:-window_size, window_size:, :]
    )


def local_response(image: np.ndarray, window_size: int) -> np.ndarray:
    """local_response Regional normalization.

    This method normalizes each pixel using the mean and standard
    deviation of all pixels within the window_size. The window_size
    parameter should be 2*radius+1 of the desired region of pixels
    to normalize by. The image should be padded by window_size//2
    on each side.

    Args:
        image ([numpy.ndarray]): 4d array of image tiles
        window_size ([int]): Size of region to normalize

    Returns:
        4d array of image tiles
    """
    image = image.astype(np.float64)
    local_mean = imboxfilt(image, window_size) / (window_size**2)
    local_mean_square = imboxfilt(image**2, window_size) / (window_size**2)
    local_std = np.sqrt(local_mean_square - (local_mean**2))
    local_std[local_std < 10**-3] = 10**-3
    return (
        image[
            :,
            window_size // 2 : -window_size // 2,
            window_size // 2 : -window_size // 2,
            :,
        ]
        - local_mean
    ) / local_std


def segment_patch(
    model: tf.keras.Model,
    xt: int,
    yt: int,
    br: BioReader,
    bw: BioWriter,
) -> None:
    """Pretrained model prediction of cell border.

    This function predicts cell border of padded and normalized image tile.
    """
    # Load the image
    x_min = max(0, xt - 91)
    x_max = min(br.X, xt + OUTPUT_TILE + 92)
    y_min = max(0, yt - 91)
    y_max = min(br.Y, yt + OUTPUT_TILE + 92)

    image: np.ndarray = br[y_min:y_max, x_min:x_max]

    # Add reflective padding if needed
    padding = [[0, 0], [0, 0]]
    if x_min == 0:
        padding[1][0] = 91
    if x_max == br.X:
        padding[1][1] = OUTPUT_TILE + 92 - br.X + xt
    if y_min == 0:
        padding[0][0] = 91
    if y_max == br.Y:
        padding[0][1] = OUTPUT_TILE + 92 - br.Y + yt

    image = np.pad(image, padding, "symmetric")

    # Preprocess the image before segmentation
    image = image[None, :, :, None]
    norm = local_response(image, WINDOW_SIZE)
    norm[norm > MAX_NORM] = MAX_NORM
    norm[norm < -MAX_NORM] = -MAX_NORM

    norm_tile = np.zeros((36, 256, 256, 1))
    for i, (x, y) in enumerate(product(range(0, 1200, 200), range(0, 1200, 200))):
        norm_tile[i, :, :, 0] = norm[0, y : y + 256, x : x + 256, 0]

    # Segment the images
    seg = model.predict(norm_tile, verbose=0)
    seg = (seg > 0).astype(np.uint8).squeeze()

    output = np.zeros((OUTPUT_TILE, OUTPUT_TILE), dtype=np.uint8)
    for i, (x, y) in enumerate(
        product(range(0, 1200, 200), range(0, OUTPUT_TILE, 200)),
    ):
        output[y : y + 200, x : x + 200] = seg[i].squeeze()

    bw[
        yt : yt + min(TILE_SIZE, br.Y - yt),
        xt : xt + min(TILE_SIZE, br.X - xt),
    ] = output[: min(br.Y - yt, TILE_SIZE), : min(br.X - xt, TILE_SIZE)]


def segment_image(model: tf.keras.Model, im_path: Path, out_dir: Path) -> None:
    """Applying segment_patch function on input images."""
    logger.info(f"Segmenting: {im_path.name}")
    # Loop through files in inpDir image collection and process
    with BioReader(im_path) as br, BioWriter(
        out_dir.joinpath(im_path.name),
        metadata=br.metadata,
    ) as bw:
        bw.dtype = np.uint8

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            executor.map(
                lambda x: segment_patch(model, x[0], x[1], br, bw),
                product(range(0, br.X, TILE_SIZE), range(0, br.Y, TILE_SIZE)),
            )
