"""Primary functions for performing binary operations."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy
from bfio import BioReader
from bfio import BioWriter
from filepattern import FilePattern
from polus.plugins.transforms.images.binary_operations.utils import TileTuple
from polus.plugins.transforms.images.binary_operations.utils import blackhat
from polus.plugins.transforms.images.binary_operations.utils import close_
from polus.plugins.transforms.images.binary_operations.utils import dilate
from polus.plugins.transforms.images.binary_operations.utils import erode
from polus.plugins.transforms.images.binary_operations.utils import fill_holes
from polus.plugins.transforms.images.binary_operations.utils import invert
from polus.plugins.transforms.images.binary_operations.utils import iterate_tiles
from polus.plugins.transforms.images.binary_operations.utils import morphgradient
from polus.plugins.transforms.images.binary_operations.utils import open_
from polus.plugins.transforms.images.binary_operations.utils import remove_large
from polus.plugins.transforms.images.binary_operations.utils import remove_small
from polus.plugins.transforms.images.binary_operations.utils import skeletonize
from polus.plugins.transforms.images.binary_operations.utils import tophat
from preadator import ProcessManager

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


REQUIRE_STRUCTURING_SHAPE = [
    Operation.BLACK_HAT,
    Operation.TOP_HAT,
    Operation.ERODE,
    Operation.DILATE,
    Operation.OPEN,
    Operation.CLOSE,
    Operation.MORPHOLOGICAL_GRADIENT,
    Operation.SKELETON,
]

OPERATION_DICT = {
    Operation.INVERT: invert,
    Operation.OPEN: open_,
    Operation.CLOSE: close_,
    Operation.MORPHOLOGICAL_GRADIENT: morphgradient,
    Operation.DILATE: dilate,
    Operation.ERODE: erode,
    Operation.FILL_HOLES: fill_holes,
    Operation.SKELETON: skeletonize,
    Operation.TOP_HAT: tophat,
    Operation.BLACK_HAT: blackhat,
    Operation.REMOVE_LARGE: remove_large,
    Operation.REMOVE_SMALL: remove_small,
}


def binary_op(  # noqa: PLR0913
    image: numpy.ndarray,
    operation: Union[Operation, str],
    structuring_shape: Union[str, StructuringShape] = StructuringShape.ELLIPSE,
    kernel: int = 3,
    iterations: int = 1,
    threshold: Optional[int] = None,
) -> numpy.ndarray:
    """Run a binary operation on in-memory data.

    This function executes a binary operation on a numpy array.

    Args:
        image: The image to perform the operation on.
        operation: The operation to perform.
        structuring_shape: The structuring shape. Not required for all operations.
            Defaults to StructuringShape.ELLIPSE.
        kernel: The kernel size for the structuring shape. Defaults to 3.
        iterations: Number of iterations to perform the operation. Only applies to erode
            and dilate. Defaults to 1.
        threshold: Size thresholding for removing large and small objects. Defaults to
            None.

    Returns:
        The image after performing the binary operation.
    """
    # Parse inputs
    if isinstance(structuring_shape, str):
        structuring_shape = StructuringShape(structuring_shape)
    if isinstance(operation, str):
        operation = Operation(operation)

    # Make sure the structuring element is valid
    cv2_struct = getattr(cv2, f"MORPH_{structuring_shape.value.upper()}")

    # Get the structuring element if it's required
    if operation in REQUIRE_STRUCTURING_SHAPE:
        se = cv2.getStructuringElement(cv2_struct, (kernel, kernel))
    else:
        se = None

    # Add extra arguments for specific operations
    if operation in [Operation.REMOVE_LARGE, Operation.REMOVE_SMALL]:
        extra_arguments = threshold
    elif operation in [Operation.ERODE, Operation.DILATE]:
        extra_arguments = iterations
    else:
        extra_arguments = None

    return OPERATION_DICT[operation](image, kernel=se, n=extra_arguments)


def _tile_thread(  # noqa: PLR0913
    filepath: Path,
    window_slice: TileTuple,
    step_slice: TileTuple,
    step_size: int,
    writer: BioWriter,
    operation: Union[Operation, str],
    structuring_shape: Union[str, StructuringShape] = StructuringShape.ELLIPSE,
    kernel: int = 3,
    iterations: int = 1,
    threshold: Optional[int] = None,
) -> None:
    with ProcessManager.thread(), BioReader(filepath) as br:
        # read a tile of BioReader
        tile = br[window_slice]

        out_tile = binary_op(
            image=tile,
            operation=operation,
            structuring_shape=structuring_shape,
            kernel=kernel,
            iterations=iterations,
            threshold=threshold,
        )

        # finalize the output
        writer[step_slice] = out_tile[0:step_size, 0:step_size]


def scalable_binary_op(  # noqa: PLR0913
    filepath: Path,
    out_dir: Path,
    operation: Union[Operation, str],
    structuring_shape: Union[str, StructuringShape] = StructuringShape.ELLIPSE,
    kernel: int = 3,
    iterations: int = 1,
    threshold: Optional[int] = None,
) -> None:
    """Run a binary operation on an arbitrarily sized image.

    Use this to run a binary operation on an image that is too large to fit into RAM.
    Instead of passing in an in memory array, pass in the path to the file to process
    and this function will perform tiled processing on it.

    Args:
        filepath: Path to image file to process.
        out_dir: Output path to put processed data.
        operation: The operation to perform.
        structuring_shape: The shape of the structuring element used to perform the
            operation. This is only used for some operations, such as dilation and
            erosion. Defaults to StructuringShape.ELLIPSE.
        kernel: Size of the kernel. Defaults to 3.
        iterations: Number of iterations to apply the operation. This is only used for
            some binary operations, such as dilation and erosion. Defaults to 1.
        threshold: Object size threshold. Only used for `remove_large` and
            `remove_small`. When `remove_large`, objects above the threshold are
            removed.Defaults to None.
    """
    if ProcessManager._thread_executor is None:
        ProcessManager.init_threads()

    if threshold is None:
        if kernel is None:
            msg = "The kernel size must be a positive number."
            logger.error(msg)
            raise ValueError(msg)
        extra_padding = kernel
    else:
        extra_padding = 512

    if (
        operation in [Operation.REMOVE_LARGE, Operation.REMOVE_SMALL]
        and threshold is None
    ):
        msg = "If removing large or small objects, the threshold value must be set."  # noqa: E501
        logger.error(msg)
        raise ValueError(msg)

    # Create the output file path
    out_path = out_dir.joinpath(filepath.name)

    with BioReader(filepath) as br:
        metadata = br.metadata

    with BioWriter(out_path, metadata=metadata) as bw:
        if br.shape != bw.shape:
            msg = (
                "The input and output images must have the same shape. "
                + f"Input shape: {br.shape}, Output shape: {bw.shape}"
            )
            logger.error(msg)
            raise ValueError(msg)

        bfio_shape: tuple = br.shape

        step_size: int = 8 * br._TILE_SIZE
        window_size: int = step_size + (2 * extra_padding)

        for window_slice, step_slice in iterate_tiles(
            shape=bfio_shape,
            window_size=window_size,
            step_size=step_size,
        ):
            # info on the Slices for debugging
            ProcessManager.submit_thread(
                _tile_thread,
                filepath=filepath,
                window_slice=window_slice,
                step_slice=step_slice,
                step_size=step_size,
                writer=bw,
                operation=operation,
                structuring_shape=structuring_shape,
                kernel=kernel,
                iterations=iterations,
                threshold=threshold,
            )

        ProcessManager.join_threads()


def _batch_process(  # noqa: PLR0913
    filepath: Path,
    out_dir: Path,
    operation: Union[Operation, str],
    structuring_shape: Union[str, StructuringShape] = StructuringShape.ELLIPSE,
    kernel: int = 3,
    iterations: int = 1,
    threshold: Optional[int] = None,
) -> None:
    with ProcessManager.process(filepath.name):
        scalable_binary_op(
            filepath=filepath,
            out_dir=out_dir,
            operation=operation,
            structuring_shape=structuring_shape,
            kernel=kernel,
            iterations=iterations,
            threshold=threshold,
        )


def batch_binary_ops(  # noqa: PLR0913
    inp_dir: Path,
    out_dir: Path,
    operation: str,
    file_pattern: str = ".+",
    kernel: int = 3,
    structuring_shape: Union[StructuringShape, int] = StructuringShape.ELLIPSE,
    iterations: int = 1,
    threshold: Optional[int] = None,
) -> None:
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

    fp = FilePattern(inp_dir, pattern=file_pattern)
    for _, files in fp():
        for file in files:
            ProcessManager.submit_process(
                _batch_process,
                Path(file),
                out_dir,
                operation,
                structuring_shape,
                kernel,
                iterations,
                threshold,
            )

    if len(ProcessManager._processes) > 0:
        ProcessManager.join_processes()
    else:
        raise RuntimeError(
            "No data to process. Make sure the input directory is correct "
            + "and that the filepattern matches files in the input directory.",
        )
