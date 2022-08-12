import logging
from pathlib import Path
from typing import List
from typing import Union

import albumentations
import numpy
import torch.nn
import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from . import augmentations
from . import helpers

__all__ = [
    "Dataset",
    "MultiEpochsDataLoader",
    "Tile",
    "UnTile",
]

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("dataset")
logger.setLevel(helpers.POLUS_LOG)


class Dataset(TorchDataset):
    preprocessing = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            augmentations.LocalNorm(
                radius=128
            ),  # TODO(Najib): Replace with Global Norm
            torch.nn.Sigmoid(),
        ]
    )

    def __init__(
        self,
        images: numpy.ndarray,
        labels: Union[numpy.ndarray, List[Path]],
        augs=None,
        preprocessing=None,
    ):

        self.inference_mode = isinstance(labels, list)
        self.images, self.labels = images, labels

        self.augmentations = augs

        if preprocessing:
            self.preprocessing = preprocessing

    def __getitem__(self, index: int):

        if self.inference_mode:
            return str(self.labels[index])

        image_tile = self.images[index].astype(numpy.float32)
        label_tile = self.labels[index].astype(numpy.float32)

        if self.preprocessing is not None:
            image_tile = self.preprocessing(image_tile).numpy().squeeze()

        if self.augmentations is not None:
            transform = albumentations.Compose(self.augmentations)

            sample = transform(image=image_tile, mask=label_tile)
            image_tile = sample["image"]
            label_tile = sample["mask"]

        image_tile = image_tile[None, ...]
        label_tile = label_tile[None, ...]

        if image_tile.shape != label_tile.shape:
            raise ValueError(
                f"Image Tile {image_tile.shape} and Label Tile {label_tile.shape} do not have matching shapes."
            )

        return image_tile, label_tile

    def __len__(self):
        return len(self.images)


class MultiEpochsDataLoader(TorchDataLoader):
    """TODO(Madhuri): Docs and type hints.
    Explain what this is supposed to do and why have it at all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class Tile(torch.nn.Module):
    """Tile an input"""

    def __init__(self, tile_size=(512, 512)):
        super(Tile, self).__init__()

        self.tile_size = tile_size

    def forward(self, x):

        n, c, h, w = x.shape

        # Calculate number of tiles for each image
        h_tiles = (h - 1) // self.tile_size[0] + 1
        w_tiles = (w - 1) // self.tile_size[1] + 1

        # Pad the image if needed
        pad_size = (
            0,
            w_tiles * self.tile_size[1] - w,
            0,
            h_tiles * self.tile_size[0] - h,
        )
        x = torch.nn.functional.pad(x, pad_size, mode="reflect")

        # Reshape the data into tiles
        x = x.reshape(n, c, h_tiles, self.tile_size[0], w_tiles, self.tile_size[1])

        # Reshape the data into proper torch format
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(
            -1, c, self.tile_size[0], self.tile_size[1]
        )

        # Return both the tiled input and the shape of the original tensor
        return x, (n, c, h, w)


class UnTile(Tile):
    """Untile an input"""

    def forward(self, x, output_shape):

        n, c, h, w = x.shape

        # Get untiling shapes
        n_images = output_shape[0]
        h_tiles = output_shape[2] // (x.shape[2] - 1) + 1
        w_tiles = output_shape[3] // (x.shape[3] - 1) + 1

        # Reshape the data into tiles
        x = x.reshape(
            n_images, c, h_tiles, w_tiles, self.tile_size[0], self.tile_size[1]
        )

        # Reconstruct original image size
        x = x.permute(0, 1, 2, 4, 3, 5)

        x = x.reshape(
            n_images, c, h_tiles * self.tile_size[0], w_tiles * self.tile_size[1]
        )
        x = x[:, :, : output_shape[2], : output_shape[3]]

        return x
