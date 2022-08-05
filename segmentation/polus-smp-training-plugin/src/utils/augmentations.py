import logging
import typing

import math
import torch
import torchvision
from albumentations.core.transforms_interface import BasicTransform
from torch import Tensor

from . import helpers

__all__ = [
    'LocalNorm',
    'PoissonTransform',
]

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("augmentations")
logger.setLevel(helpers.POLUS_LOG)


class LocalNorm(object):
    def __init__(
            self,
            radius: int = 16,
            max_response: typing.Union[int, float] = 6,
    ):
        self.radius: int = radius
        self.window_size: int = 1 + 2 * radius
        self.max_response: float = float(max_response)
        self.pad = torchvision.transforms.Pad(radius + 1, padding_mode='reflect')
        # Mode can be 'test', 'train' or 'eval'.
        self.mode: str = 'eval'

    def __call__(self, x: Tensor):
        return torch.clip(
            self.__local_response(self.pad(x)),
            min=-self.max_response,
            max=self.max_response,
        )

    def __image_filter(self, image: Tensor) -> Tensor:
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

    def __local_response(self, image: Tensor):
        """ Regional normalization.
        This method normalizes each pixel using the mean and standard deviation of
        all pixels within the window_size. The window_size parameter should be
        2 * radius + 1 of the desired region of pixels to normalize by. The image should
        be padded by window_size // 2 on each side.
        """
        local_mean: Tensor = self.__image_filter(image) / (self.window_size ** 2)
        local_mean_square: Tensor = self.__image_filter(image.pow(2)) / (self.window_size ** 2)

        # Use absolute difference because sometimes error causes negative values
        local_std = torch.clip(
            (local_mean_square - local_mean.pow(2)).abs().sqrt(),
            min=1e-3,
        )

        min_i, max_i = self.radius, -self.radius - 1
        response = image[..., min_i:max_i, min_i:max_i]

        return (response - local_mean) / local_std


class GlobalNorm:
    # TODO(Najib)
    pass


class PoissonTransform(BasicTransform):
    """ Apply poisson noise to float32 images.
    """

    def __init__(self, peak: int, p: float = 0.5):
        """
        Args:
            peak: [1-10] high values introduces more noise in the image.
            p: probability of applying the transform.
        """
        if not 1 <= peak <= 10:
            message = f'\'peak\' must be in the range [1, 10]. Got {peak} instead.'
            logger.error(message)
            raise ValueError(message)

        super(PoissonTransform, self).__init__(p=p)
        self.peak: int = peak

    def apply(self, image: Tensor, **_):
        value = torch.tensor(math.exp(10 - self.peak))

        if torch.any(torch.isnan(image)):
            message = f'image had nan values.'
            logger.error(message)
            raise ValueError(message)

        if torch.any(torch.lt(image, 0)):
            message = f'image had negative values.'
            logger.error(message)
            raise ValueError(message)

        noisy_image = torch.poisson(image * value).float() / value
        return noisy_image

    def update_params(self, params, **kwargs):
        if hasattr(self, "peak"):
            params["peak"] = self.peak
        return params

    @property
    def targets(self):
        return {"image": self.apply}

    def get_params_dependent_on_targets(
            self,
            params: typing.Dict[str, typing.Any],
    ) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError

    def get_transform_init_args_names(self) -> typing.Tuple[str, ...]:
        raise NotImplementedError
