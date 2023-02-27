"""Primary functions for performing binary operations."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
from bfio import BioReader, BioWriter
from filepattern import FilePattern
from preadator import ProcessManager

from polus.plugins.transforms.images.binary_operations.utils import (
    TileTuple,
    blackhat,
    close_,
    dilate,
    erode,
    fill_holes,
    invert,
    iterate_tiles,
    morphgradient,
    open_,
    remove_large,
    remove_small,
    skeletonize,
    tophat,
)

logger = logging.getLogger(__name__)


class Operation(str, Enum):
    """Available binary operations."""

    INVERT = "invert"
    OPEN = "open"
    CLOSE = "close"
    MORPHOLOGICAL_GRADIENT = "morphologicalGradient"
    DILATE = "dilate"
    ERODE = "erode"
    FILL_HOLES = "fillHoles"
    SKELETON = "skeleton"
    TOP_HAT = "topHat"
    BLACK_HAT = "blackHat"
    REMOVE_LARGE = "removeLarge"
    REMOVE_SMALL = "removeSmall"


class StructuringShape(str, Enum):
    """The structuring element used for binary operations."""

    ELLIPSE = "ellipse"
    RECT = "rect"
    CROSS = "cross"


def _process_tile(
    filepath: Path,
    window_slice: TileTuple,
    step_slice: TileTuple,
    step_size: int,
    writer: BioWriter,
    operation: Operation,
    se: int = 3,
    iterations: int = 1,
    threshold: Optional[int] = None,
):
    # The function that will be run based on user input, and an additional parameter
    dispatch = {
        "invert": (invert, None),
        "open": (open_, None),
        "close": (close_, None),
        "morphologicalGradient": (morphgradient, None),
        "dilate": (dilate, iterations),
        "erode": (erode, iterations),
        "fillHoles": (fill_holes, None),
        "skeleton": (skeletonize, None),
        "topHat": (tophat, None),
        "blackHat": (blackhat, None),
        "removeLarge": (
            remove_large,
            threshold,
        ),
        "removeSmall": (
            remove_small,
            threshold,
        ),
    }

    # Need extra padding when doing operations so it does not skew results
    # Initialize variables based on operation
    function = dispatch[operation][0]
    extra_arguments = dispatch[operation][1]

    with ProcessManager.thread():
        with BioReader(filepath) as br:
            # read a tile of BioReader
            tile_readarray = br[window_slice]

            tile_writearray = function(tile_readarray, kernel=se, n=extra_arguments)

            # finalize the output
            writer[step_slice] = tile_writearray[0:step_size, 0:step_size]


def scalable_binary_op(
    filepath: Path,
    out_dir: Path,
    operation: Operation,
    kernel: int = 3,
    structuring_shape: StructuringShape = StructuringShape.ELLIPSE,
    iterations: int = 1,
    threshold: Optional[int] = None,
):
    """Run a binary operation on an arbitrarily sized image.

    Use this to run a binary operation on an image that is too large to fit into RAM.
    Instead of passing in an in memory array, pass in the path to the file to process
    and this function will perform tiled processing on it.

    Args:
        filepath: Path to image file to process.
        out_dir: Output path to put processed data.
        operation: The operation to perform.
        kernel: Size of the kernel. Defaults to 3.
        structuring_shape: The shape of the structuring element used to perform the
            operation. This is only used for some operations, such as dilation and
            erosion. Defaults to StructuringShape.ELLIPSE.
        iterations: Number of iterations to apply the operation. This is only used for
            some binary operations, such as dilation and erosion. Defaults to 1.
        threshold: Object size threshold. Only used for `remove_large` and
            `remove_small`. When `remove_large`, objects above the threshold are
            removed.Defaults to None.
    """
    with ProcessManager.process(filepath.name):
        if not isinstance(structuring_shape, StructuringShape):
            raise TypeError(
                "structuring_shape must be a str or StructuringShape value."
            )
        cv2_struct = getattr(cv2, f"MORPH_{structuring_shape.value.upper()}")

        if threshold is None:
            assert kernel is not None, "The kernel size must be a positive number."
            extra_padding = kernel
            se = cv2.getStructuringElement(cv2_struct, (kernel, kernel))
        else:
            extra_padding = 512
            se = None

        if operation in [Operation.REMOVE_LARGE, Operation.REMOVE_SMALL]:
            assert (
                threshold is not None
            ), "If removing large or small objects, the threshold value must be set."

        # Create the output file path
        out_path = out_dir.joinpath(filepath.name)

        with BioReader(filepath) as br:
            metadata = br.metadata

        with BioWriter(out_path, metadata=metadata) as bw:
            assert br.shape == bw.shape
            bfio_shape: tuple = br.shape

            step_size: int = 8 * br._TILE_SIZE
            window_size: int = step_size + (2 * extra_padding)

            for window_slice, step_slice in iterate_tiles(
                shape=bfio_shape, window_size=window_size, step_size=step_size
            ):
                # info on the Slices for debugging
                ProcessManager.submit_thread(
                    _process_tile,
                    filepath=filepath,
                    window_slice=window_slice,
                    step_slice=step_slice,
                    step_size=step_size,
                    writer=bw,
                    operation=operation,
                    se=se,
                    iterations=iterations,
                    threshold=threshold,
                )

            ProcessManager.join_threads()


def batch_binary_ops(
    inp_dir: Path,
    out_dir: Path,
    operation: str,
    file_pattern: str = ".+",
    kernel: int = 3,
    structuring_shape: Union[StructuringShape, int] = StructuringShape.ELLIPSE,
    iterations: int = 1,
    threshold: Optional[int] = None,
):
    """Run binary operations on a batch of images.

    The inputs are mostly consistent with the `scalable_binary_op` function, except the
    file_pattern input is needed to select a subset of the data.

    Args:
        inp_dir: Path to image files to process.
        out_dir: Output path to put processed data.
        operation: The operation to perform.
        file_pattern: The filepattern used to select a subset of the data.
        kernel: Size of the kernel. Defaults to 3.
        structuring_shape: The shape of the structuring element used to perform the
            operation. This is only used for some operations, such as dilation and
            erosion. Defaults to StructuringShape.ELLIPSE.
        iterations: Number of iterations to apply the operation. This is only used for
            some binary operations, such as dilation and erosion. Defaults to 1.
        threshold: Object size threshold. Only used for `remove_large` and
            `remove_small`. When `remove_large`, objects above the threshold are
            removed.Defaults to None.
    """
    ProcessManager.init_processes()

    # Convert string inputs to proper enum
    if isinstance(structuring_shape, str):
        structuring_shape = StructuringShape(structuring_shape)
    if isinstance(operation, str):
        operation = Operation(operation)

    fp = FilePattern(inp_dir, pattern=file_pattern)
    for _, files in fp():
        for file in files:
            ProcessManager.submit_process(
                scalable_binary_op,
                Path(file),
                out_dir,
                operation,
                kernel,
                structuring_shape,
                iterations,
                threshold,
            )

    if len(ProcessManager._processes) > 0:
        ProcessManager.join_processes()
    else:
        raise RuntimeError(
            "No data to process. Make sure the input directory is correct "
            + "and that the filepattern matches files in the input directory."
        )
