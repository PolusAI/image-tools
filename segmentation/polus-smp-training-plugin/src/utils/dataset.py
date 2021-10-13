import logging
from pathlib import Path
from typing import Dict
from typing import Union

import numpy
import torch
import torchvision
from bfio import BioReader
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform

from . import helpers

__all__ = [
    'Dataset',
]

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("dataset")
logger.setLevel(helpers.POLUS_LOG)


class LocalNorm(object):
    def __init__(
            self,
            window_size: int = 129,
            max_response: Union[int, float] = 6,
    ):
        assert window_size % 2 == 1, 'window_size must be an odd integer'

        self.window_size: int = window_size
        self.max_response: float = float(max_response)
        self.pad = torchvision.transforms.Pad(window_size // 2 + 1, padding_mode='reflect')
        # Mode can be 'test', 'train' or 'eval'.
        self.mode: str = 'eval'

    def __call__(self, x: Tensor):
        return torch.clip(
            self.local_response(self.pad(x)),
            min=-self.max_response,
            max=self.max_response,
        )

    def image_filter(self, image: Tensor) -> Tensor:
        """ Use a box filter on a stack of images
        This method applies a box filter to an image. The input is assumed to be a
        4D array, and should be pre-padded. The output will be smaller by
        window_size - 1 pixels in both width and height since this filter does not pad
        the input to account for filtering.
        """
        integral_image: Tensor = image.cumsum(dim=-1).cumsum(dim=-2)
        return (
                integral_image[..., :-self.window_size - 1, :-self.window_size - 1]
                + integral_image[..., self.window_size:-1, self.window_size:-1]
                - integral_image[..., self.window_size:-1, :-self.window_size - 1]
                - integral_image[..., :-self.window_size - 1, self.window_size:-1]
        )

    def local_response(self, image: Tensor):
        """ Regional normalization.
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2 * radius + 1 of the desired region of pixels to normalize by. The image should
        be padded by window_size // 2 on each side.
        """
        local_mean: Tensor = self.image_filter(image) / (self.window_size ** 2)
        local_mean_square: Tensor = self.image_filter(image.pow(2)) / (self.window_size ** 2)

        # Use absolute difference because sometimes error causes negative values
        local_std = torch.clip(
            (local_mean_square - local_mean.pow(2)).abs().sqrt(),
            min=1e-3,
        )

        min_i, max_i = self.window_size // 2, -self.window_size // 2 - 1
        response = image[..., min_i:max_i, min_i:max_i]

        return (response - local_mean) / local_std

class Poisson_Transform(BasicTransform):
    
    """ Apply poisson noise.
    Args:
        peak (int): [1-10] high values introduces more noise in the image
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        float32 """
    
    def __init__(self, peak, always_apply=False, p=0.5):        
        super(Poisson_Transform, self).__init__(always_apply, p)      
        self.peak = peak        
    def apply(self,img, peak):  
        if peak > 10:
            raise ValueError('Peak values range is 1-10')

        value = np.exp(10 - peak)
        noisy_image = np.random.poisson(img * value ).astype(np.float32) / value       
        return noisy_image
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "peak"):
            params["peak"] = self.peak
        return params 
    @property
    def targets(self):
        return {"image": self.apply}


class Dataset(TorchDataset):
    def __init__(self, labels_map: Dict[Path, Path], tile_map: helpers.Tiles):
        self.labels_paths: Dict[Path, Path] = labels_map
        self.tiles: helpers.Tiles = tile_map
        self.preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            LocalNorm(),
        ])

    def __getitem__(self, index: int):
        image_path, y_min, y_max, x_min, x_max = self.tiles[index]
        label_path = self.labels_paths[image_path]

        # read and preprocess image
        with BioReader(image_path) as reader:
            image_tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]
        image_tile = numpy.asarray(image_tile, dtype=numpy.float32)

               # read and preprocess label
        with BioReader(label_path) as reader:
            label_tile = reader[y_min:y_max, x_min:x_max, 0, 0, 0]
        label_tile = numpy.asarray(label_tile, dtype=numpy.float32)
        # label_tile = numpy.reshape(label_tile, (1, y_max - y_min, x_max - x_min))

        transform = A.Compose([A.RandomCrop(width=256, height=256), 
                                A.GridDistortion(num_steps=5, distort_limit=0.3,p=0.1),
                                A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.4, p=0.2),
                                Poisson_Transform(peak=10,p=0.3),   
                                A.OneOf([A.MotionBlur(blur_limit=15, p=0.1),
                                        A.Blur(blur_limit=15, p=0.1), 
                                        A.MedianBlur(blur_limit=3, p=.1)]
                                        , p=0.2)
                                        
                                ])

        sample = transform(image=image_tile, mask=label_tile)
        image_tile, label_tile = sample['image'], sample['mask']  

        image_tile = self.preprocessing(image_tile).numpy()

        return image_tile, label_tile

    def __len__(self):
        return len(self.tiles)
