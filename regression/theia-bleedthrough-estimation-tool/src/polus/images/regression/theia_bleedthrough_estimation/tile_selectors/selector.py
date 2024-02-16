"""Selector base class for Theia Bleedthrough Estimation plugin."""

import abc
import operator
import pathlib

import bfio
import numpy

from ..utils import constants
from ..utils import helpers

logger = helpers.make_logger(__name__)


ScoresDict = dict[tuple[int, int, int, int, int, int], float]
"""A dictionary of scores for each tile in an image.

key: (6-tuple of indices) (z_min, z_max, y_min, y_max, x_min, x_max)
value: (float) score
"""

TileIndices = list[tuple[int, int, int, int, int, int]]
"""A list of coordinates for each tile that was selected by a Selector.

Each item is a 6-tuple of indices: (z_min, z_max, y_min, y_max, x_min, x_max)
"""


class Selector(abc.ABC):
    """Base class for tile-selection methods."""

    __slots__ = (
        "__files",
        "__is_high_better",
        "__num_tiles_per_channel",
        "__scores",
        "__selected_tiles",
        "__image_mins",
        "__image_maxs",
    )

    def __init__(
        self,
        files: list[pathlib.Path],
        num_tiles_per_channel: int,
        is_high_better: bool = True,
    ) -> None:
        """Scores all tiles in images and selects the best few for training a model.

        Args:
            files: List of paths to images from which tiles will be selected.
            num_tiles_per_channel: How many tiles to select from each channel.
            is_high_better: Whether higher scoring tiles are better.
        """
        self.__files = files
        self.__num_tiles_per_channel = num_tiles_per_channel
        self.__is_high_better = is_high_better

        self.__image_mins: list[int] = []
        self.__image_maxs: list[int] = []
        self.__scores: list[ScoresDict] = []
        self.__selected_tiles: TileIndices = []

    def fit(self) -> None:
        """Scores all tiles in images and selects the best few for training a model.

        This method must be called before using the `selected_tiles` property.
        """
        # with preadator.ProcessManager(
        # ) as executor:
        #     futures: list[concurrent.futures.Future[tuple[ScoresDict, int, int]]] = [
        #         for i, file_path in enumerate(self.__files)
        #     for future in concurrent.futures.as_completed(futures):

        # for _, (score, image_min, image_max) in results:

        for file_path in self.__files:
            score, image_min, image_max, _ = self._score_tiles_thread(file_path, 0)
            self.__scores.append(score)
            self.__image_mins.append(image_min)
            self.__image_maxs.append(image_max)

        self.__selected_tiles = self._select_best_tiles()

    @property
    def selected_tiles(self) -> TileIndices:
        """Returns the indices of the selected tiles."""
        return self.__selected_tiles

    @property
    def image_mins(self) -> list[int]:
        """Returns the minimum intensity of each image."""
        return self.__image_mins

    @property
    def image_maxs(self) -> list[int]:
        """Returns the maximum intensity of each image."""
        return self.__image_maxs

    @abc.abstractmethod
    def _score_tile(self, tile: numpy.ndarray) -> float:
        pass

    def _score_tiles_thread(
        self,
        file_path: pathlib.Path,
        index: int,
    ) -> tuple[ScoresDict, int, int, int]:
        """This method runs in a single thread and scores all tiles for a single file.

        Args:
            file_path: Path to image for which the tiles need to be scored.
            index: Index of the file in the list of files. This is used to keep track
            of the thread's return order.

        Returns:
            A Dictionary of tile-scores.
        """
        with bfio.BioReader(file_path, max_workers=constants.NUM_THREADS) as reader:
            scores_dict: ScoresDict = {}
            logger.debug(f"Ranking tiles in {file_path.name}...")
            num_tiles = helpers.count_tiles_2d(reader)
            image_min = numpy.Infinity
            image_max = numpy.NINF

            for i, (_, y_min, y_max, x_min, x_max) in enumerate(
                helpers.tile_indices_2d(reader),
            ):
                if i % 10 == 0:
                    logger.debug(
                        f"Ranking tiles in {file_path.name}. "
                        f"Progress {100 * i / num_tiles:6.2f} %",
                    )

                tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, 0, 0, 0])

                # TODO: Actually handle 3d images properly with 3d tile-chunks.
                key = (0, 1, y_min, y_max, x_min, x_max)
                if key in scores_dict:
                    scores_dict[key] = (max if self.__is_high_better else min)(
                        scores_dict[key],
                        self._score_tile(tile),
                    )
                else:
                    scores_dict[key] = self._score_tile(tile)

                image_min = numpy.min(tile[tile > 0], initial=image_min)
                image_max = numpy.max(tile, initial=image_max)

        return scores_dict, image_min, image_max, index

    def _select_best_tiles(self) -> TileIndices:
        """Sort the tiles by their scores and select the best few from each channel.

        Returns:
            List of indices of the best tiles.
        """
        return list(
            {
                coordinates
                for scores_dict in self.__scores
                for coordinates, _ in sorted(
                    scores_dict.items(),
                    key=operator.itemgetter(1),
                    reverse=self.__is_high_better,
                )[: self.__num_tiles_per_channel]
            },
        )
