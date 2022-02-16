import abc
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import Type

import numpy
import scipy.stats
from bfio import BioReader

import utils

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('selectors')
logger.setLevel(utils.POLUS_LOG)


class Selector(abc.ABC):
    """ Base class for tile-selection methods.
    """

    __slots__ = (
        '__files',
        '__is_high_better',
        '__num_tiles_per_channel',
        '__scores',
        '__selected_tiles',
        '__image_mins',
        '__image_maxs',
    )

    def __init__(self, files: list[Path], num_tiles_per_channel: int, is_high_better: bool = True):
        """ Scores all tiles in each image and selects the best few from each
            image for training a model.
        
        Args:
            files: List of paths to images from which tiles will be selected.
            num_tiles_per_channel: How many tiles to select from each channel.
            is_high_better: Whether higher scoring tiles are better.
        """
        self.__files = files
        self.__num_tiles_per_channel = num_tiles_per_channel
        self.__is_high_better = is_high_better

        self.__image_mins = list()
        self.__image_maxs = list()
        self.__scores: list[utils.ScoresDict] = list()

        # noinspection PyTypeChecker
        self.__selected_tiles: utils.TileIndices = None

    def fit(self):
        with ProcessPoolExecutor(max_workers=utils.NUM_THREADS) as executor:
            futures: list[Future[tuple[utils.ScoresDict, int, int]]] = [
                executor.submit(self._score_tiles_thread, file_path)
                for file_path in self.__files
            ]
            for future in futures:
                score, image_min, image_max = future.result()
                self.__scores.append(score)
                self.__image_mins.append(image_min)
                self.__image_maxs.append(image_max)

        self.__selected_tiles = self._select_best_tiles()

    @property
    def selected_tiles(self) -> utils.TileIndices:
        return self.__selected_tiles

    @property
    def image_mins(self) -> list[int]:
        return self.__image_mins

    @property
    def image_maxs(self) -> list[int]:
        return self.__image_maxs

    @abc.abstractmethod
    def _score_tile(self, tile: numpy.ndarray) -> float:
        raise NotImplementedError(f'Any subclass of Criterion must implement the \'_score_tile\' method.')

    def _score_tiles_thread(self, file_path: Path) -> tuple[utils.ScoresDict, int, int]:
        """ This method runs in a single thread and scores all tiles for a
            single file.

        Args:
            file_path: Path to image for which the tiles need to be scored.

        Returns:
            A Dictionary of tile-scores. See `utils/types.py`
        """
        with BioReader(file_path, max_workers=utils.NUM_THREADS) as reader:

            scores_dict: utils.ScoresDict = dict()
            logger.info(f'Ranking tiles in {file_path.name}...')
            num_tiles = utils.count_tiles(reader)
            image_min = numpy.iinfo(reader.dtype).max
            image_max = -numpy.iinfo(reader.dtype).min

            for i, (_, _, y_min, y_max, x_min, x_max) in enumerate(utils.tile_indices(reader)):
                if i % 10 == 0:
                    logger.info(f'Ranking tiles in {file_path.name}. Progress {100 * i / num_tiles:6.2f} %')

                tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, 0, 0, 0])

                # TODO: Actually handle 3d images properly with 3d tile-chunks.
                key = (0, 1, y_min, y_max, x_min, x_max)
                if key in scores_dict:
                    scores_dict[key] = (max if self.__is_high_better else min)(scores_dict[key], self._score_tile(tile))
                else:
                    scores_dict[key] = self._score_tile(tile)

                image_min = numpy.min(tile[tile > 0], initial=image_min)
                image_max = numpy.max(tile, initial=image_max)

        return scores_dict, image_min, image_max

    def _select_best_tiles(self) -> utils.TileIndices:
        """ Sort the tiles based on their scores and select the best few from
            each channel

        Returns:
            List of indices of the best tiles. See `utils.types.py`
        """
        return list(set(
            coordinates for scores_dict in self.__scores
            for coordinates, _ in list(sorted(
                scores_dict.items(),
                key=itemgetter(1),
                reverse=self.__is_high_better,
            ))[:self.__num_tiles_per_channel]
        ))


class Entropy(Selector):
    """ Select tiles with the highest entropy.
    """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        counts, _ = numpy.histogram(tile.flat, bins=128, density=True)
        return float(scipy.stats.entropy(counts))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel)


class MeanIntensity(Selector):
    """ Select tiles with the highest mean intensity.
    """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.mean(tile))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel)


class MedianIntensity(Selector):
    """ Select tiles with the highest median intensity.
    """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.median(tile))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel)


class IntensityRange(Selector):
    """ Select tiles with the largest difference between the 90th and 10th
        percentile intensities.
    """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.percentile(tile, 90) - numpy.percentile(tile, 10))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel)


""" A Dictionary to let us use a Selector by name.
"""
SELECTORS: dict[str, Type[Selector]] = {
    'Entropy': Entropy,
    'MeanIntensity': MeanIntensity,
    'MedianIntensity': MedianIntensity,
    'IntensityRange': IntensityRange,
}
