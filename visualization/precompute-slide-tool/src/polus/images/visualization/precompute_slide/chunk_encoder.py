"""Provides the ChunkEncoder classes."""

import abc
import logging

import numpy

from . import utils

logger = logging.getLogger(__file__)
logger.setLevel(utils.POLUS_LOG)


class ChunkEncoder(abc.ABC):
    """Encoder for pyramid chunks.

    Modified and condensed from multiple functions and classes:
    https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/chunk_encoding.py
    """

    # Data types used by Neuroglancer
    DATA_TYPES = (
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
        "float32",
    )

    def __init__(self, info: dict) -> None:
        """Initialize the encoder.

        Inputs:
            info - dictionary containing the information in the info file
        """
        try:
            data_type = info["data_type"]
            num_channels = info["num_channels"]
        except KeyError as e:
            msg = f"The info dict is missing an essential key {e}"
            raise KeyError(msg) from e

        if not isinstance(num_channels, int) or not num_channels > 0:
            msg = f"Invalid value {num_channels} for num_channels"
            raise KeyError(msg)

        if data_type not in ChunkEncoder.DATA_TYPES:
            msg = f"Invalid data_type {data_type} (options: {ChunkEncoder.DATA_TYPES})"
            raise KeyError(msg)

        self.info = info
        self.num_channels = num_channels
        self.dtype = numpy.dtype(data_type).newbyteorder("<")

    @abc.abstractmethod
    def encode(self, chunk: numpy.ndarray) -> bytes:
        """Encode a chunk from a Numpy array into bytes."""
        pass


class NeuroglancerChunkEncoder(ChunkEncoder):
    """Encoder for Neuroglancer."""

    def encode(self, chunk: numpy.ndarray) -> bytes:
        """Encode a chunk from a Numpy array into bytes.

        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream).
        """
        # Rearrange the image for Neuroglancer
        chunk = numpy.moveaxis(
            chunk.reshape(chunk.shape[0], chunk.shape[1], 1, 1),
            (0, 1, 2, 3),
            (2, 3, 1, 0),
        )
        chunk = numpy.asarray(chunk).astype(self.dtype)

        if chunk.ndim != 4:  # noqa: PLR2004
            msg = "Chunk must be 4 dimensional."
            raise ValueError(msg)

        if chunk.shape[0] != self.num_channels:
            msg = f"Chunk must have {self.num_channels} channels."
            raise ValueError(msg)

        return chunk.tobytes()


class ZarrChunkEncoder(ChunkEncoder):
    """Encoder for Zarr."""

    def encode(self, chunk: numpy.ndarray) -> numpy.ndarray:
        """Encode a chunk from a Numpy array into bytes.

        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream).
        """
        # Rearrange the image for Neuroglancer
        chunk = chunk.reshape(chunk.shape[0], chunk.shape[1], 1, 1, 1).transpose(
            4,
            2,
            3,
            0,
            1,
        )
        return numpy.asarray(chunk).astype(self.dtype)


class DeepZoomChunkEncoder(ChunkEncoder):
    """Encoder for DeepZoom."""

    def encode(self, chunk: numpy.ndarray) -> numpy.ndarray:
        """Encode a chunk for DeepZoom.

        Nothing special to do for encoding except checking the number of
        dimensions.

        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Check to make sure the data is formatted properly
        if chunk.ndim != 2:  # noqa: PLR2004
            msg = "Chunk must be 2 dimensional."
            raise ValueError(msg)
        return chunk
