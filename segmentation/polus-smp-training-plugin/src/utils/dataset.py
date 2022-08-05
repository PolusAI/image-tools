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
    'Dataset',
    'MultiEpochsDataLoader',
]

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("dataset")
logger.setLevel(helpers.POLUS_LOG)


class Dataset(TorchDataset):
    preprocessing = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        augmentations.LocalNorm(radius=128),  # TODO(Najib): Replace with Global Norm
        torch.nn.Sigmoid(),
    ])

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
            image_tile = sample['image']
            label_tile = sample['mask']

        image_tile = image_tile[None, ...]
        label_tile = label_tile[None, ...]

        if image_tile.shape != label_tile.shape:
            raise ValueError(f'Image Tile {image_tile.shape} and Label Tile {label_tile.shape} do not have matching shapes.')

        return image_tile, label_tile

    def __len__(self):
        return len(self.images)


class MultiEpochsDataLoader(TorchDataLoader):
    """ TODO(Madhuri): Docs and type hints.
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
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
