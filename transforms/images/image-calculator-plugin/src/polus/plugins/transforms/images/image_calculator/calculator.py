"""Functions for performing image calculations."""

import enum
import logging
import multiprocessing
import os
import pathlib
import typing

import bfio
import numpy

# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
MAX_WORKERS = max(1, multiprocessing.cpu_count() // 2)

CHUNK_SIZE = 4_096
SUFFIX_LEN = 6

OPERATIONS: dict[str, typing.Callable] = {
    "multiply": numpy.multiply,
    "divide": numpy.divide,
    "add": numpy.add,
    "subtract": numpy.subtract,
    "and": numpy.bitwise_and,
    "or": numpy.bitwise_or,
    "xor": numpy.bitwise_xor,
    "min": numpy.minimum,
    "max": numpy.maximum,
    "absdiff": lambda x, y: numpy.abs(numpy.subtract(x, y)),
}


class Operation(str, enum.Enum):
    """The operations that can be performed on the images."""

    Multiply = "multiply"
    Divide = "divide"
    Add = "add"
    Subtract = "subtract"
    And = "and"
    Or = "or"
    Xor = "xor"
    Min = "min"
    Max = "max"
    AbsDiff = "absdiff"


def _process_chunk(  # noqa: PLR0913
    primary_reader: bfio.BioReader,
    secondary_reader: bfio.BioReader,
    x: int,
    x_max: int,
    y: int,
    y_max: int,
    z: int,
    writer: bfio.BioWriter,
    operation: Operation,
) -> None:
    """Process on chunk of the images in a thread/process."""
    writer[y:y_max, x:x_max, z] = OPERATIONS[operation.value](
        primary_reader[y:y_max, x:x_max, z],
        secondary_reader[y:y_max, x:x_max, z],
    )


def process_image(
    primary_image: pathlib.Path,
    secondary_image: pathlib.Path,
    output_dir: pathlib.Path,
    operation: Operation,
) -> None:
    """Applies the specified operation on the two images and saves the output image.

    Args:
        primary_image: The primary image file.
        secondary_image: The secondary image file.
        output_dir: The output directory.
        operation: The operation to perform on the images.
    """
    with (
        bfio.BioReader(primary_image, max_workers=1) as primary_reader,
        bfio.BioReader(secondary_image, max_workers=1) as secondary_reader,
    ):
        input_extension = "".join(
            [s for s in primary_image.suffixes[-2:] if len(s) < SUFFIX_LEN],
        )
        out_name = primary_image.name.replace(input_extension, POLUS_IMG_EXT)
        out_path = output_dir.joinpath(out_name)

        # Initialize the output image
        with bfio.BioWriter(out_path, metadata=primary_reader.metadata) as writer:
            for z in range(primary_reader.Z):
                for x in range(0, primary_reader.X, CHUNK_SIZE):
                    x_max = min(x + CHUNK_SIZE, primary_reader.X)
                    for y in range(0, primary_reader.Y, CHUNK_SIZE):
                        y_max = min(y + CHUNK_SIZE, primary_reader.Y)

                        _process_chunk(
                            primary_reader,
                            secondary_reader,
                            x,
                            x_max,
                            y,
                            y_max,
                            z,
                            writer,
                            operation,
                        )
