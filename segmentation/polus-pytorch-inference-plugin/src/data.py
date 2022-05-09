import logging
import pathlib
import typing

import bfio
import numpy
import torch

import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("data")
logger.setLevel(utils.POLUS_LOG)


TILE_INDICES = typing.Tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max)
TILE_ITEM = typing.Tuple[
    pathlib.Path,  # path to image
    TILE_INDICES,  # tile indices in the image
    torch.Tensor  # the tile itself
]
SCALABLE_BATCH = typing.Tuple[
    typing.List[pathlib.Path],  # list of image paths
    typing.List[TILE_INDICES],  # list of indices in the corresponding image.
    torch.Tensor  # stacked batch of tiles shaped (B, 1, Y, X)
]


class ScalableDataset:

    def __init__(
            self,
            image_paths: typing.List[pathlib.Path],
            tile_stride: int,
            batch_size: int,
            fill_value: float = 0,  # TODO: handle filling with mean of tile
    ):
        """ Creates a scalable data loader for use in running inference on
         arbitrarily large ome.tif images.

        The images must be two-dimensional, single-channel ome.tif images. If
         the image has multiple channels, we perform inference on the 0th
         channel. The image may be arbitrarily large in the X and X dimensions.
         This class will load tiles from the image to fill a batch.

        Args:
            image_paths: list of paths to ome.tif images.
            tile_stride: size of the x and y strides to use for tiled loading.
            batch_size: number of tiles in a batch of images.
            fill_value: If the tile is too small, use this value to fill in the
                         right and bottom edges to make up the shape.
        """
        self.image_paths = image_paths
        self.tile_stride = tile_stride
        self.tile_shape = (tile_stride, tile_stride)
        self.batch_size = batch_size
        self.fill_value = fill_value

        self.paths_tile_indices = [
            (path, indices)
            for path in self.image_paths
            for indices in self.__get_tile_indices(path)
        ]

        self.batch_indices = [
            [start + i for i in range(len(self.paths_tile_indices[start:start + self.batch_size]))]
            for start in range(0, len(self.paths_tile_indices), self.batch_size)
        ]

    def num_batches(self) -> int:
        """ Returns the number of batches in the dataset.
        """
        return len(self.batch_indices)

    def load_batch(self, index: int) -> SCALABLE_BATCH:
        """ Load a batch of tiles into memory.

        Args:
            index: index of the batch to load.

        Returns:
            A 3-tuple of:
                - list of paths of images from which the tiles were loaded.
                - list of tile indices from inside the corresponding image.
                - A stacked batch of tiles. Shape is (B, 1, Y, X)
        """
        # TODO: Optimize this to open each image once per batch
        batch = list(map(self.__load_tile, self.batch_indices[index]))

        # Unpack the batch
        paths = [path for path, _, _ in batch]
        tile_indices = [indices for _, indices, _ in batch]
        tiles = [tile for _, _, tile in batch]

        # stack the tiles into a batch and add axis for channel
        tiles = torch.stack(tiles, dim=0)[:, None, :, :]

        return paths, tile_indices, tiles

    def __load_tile(self, index: int) -> TILE_ITEM:
        """ Returns an indexed tile from the set of input images.

        Args:
            index: Index of the tile.

        Returns:
            A 3-tuple of:
                - path to input image
                - indices of the tile in that image,
                - the tile itself
        """

        path, indices = self.paths_tile_indices[index]
        y_min, y_max, x_min, x_max = indices

        with bfio.BioReader(path) as reader:
            tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]

        tile = self.__pad_tile(tile.astype(numpy.float32))
        return path, indices, torch.from_numpy(tile)

    def __get_tile_indices(self, image_path: pathlib.Path) -> typing.List[TILE_INDICES]:
        """ Given a path to an ome.tif image, returns the tile indices needed to perform scalable inference

        Args:
            image_path: Path to an ome.tif image. It must be a single-channel image but can be arbitrarily large.

        Returns:
            A list of tuples of tile indices.
        """
        with bfio.BioReader(image_path) as reader:
            y_end, x_end = reader.Y, reader.X

        tile_indices = [
            (y_min, min(y_end, y_min + self.tile_stride), x_min, min(x_end, x_min + self.tile_stride))
            for y_min in range(0, y_end, self.tile_stride)
            for x_min in range(0, x_end, self.tile_stride)
        ]
        return tile_indices

    def __pad_tile(self, tile: numpy.ndarray) -> numpy.ndarray:
        """ Pads a 2d tile that is too small with zeros on the right and bottom.
        """

        padded = numpy.zeros(shape=self.tile_shape, dtype=tile.dtype)
        if tile.shape != self.tile_shape:
            x, y = tile.shape
            padded[:x, :y] = tile[:]
            padded[x:, y:] = self.fill_value

        tile = padded.astype(tile.dtype)
        return tile
