import abc
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Type

import filepattern
import numpy
import scipy.ndimage
from bfio import BioReader
from bfio import BioWriter
from sklearn import linear_model

import utils
from tile_selectors import Selector

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('models')
logger.setLevel(utils.POLUS_LOG)


class Model(abc.ABC):
    """ Base class for models that can be trained to estimate bleed-through.
    """

    __slots__ = (
        '__files',
        '__channel_overlap',
        '__coefficients',
        '__image_mins',
        '__image_maxs',
        '__kernel_size',
        '__num_pixels',
    )

    def __init__(self, files: list[Path], channel_overlap: int, kernel_size: int):
        """ Trains a model on the given files.

        Args:
            files: Paths to images on which the model will be trained.
            channel_overlap: The number of adjacent channels to consider for
                bleed-through estimation.
            kernel_size: The size of the convolutional kernel used to estimate
                bleed-through.
        """
        self.__files = files
        self.__channel_overlap = min(len(self.__files) - 1, max(1, channel_overlap))
        self.__kernel_size = kernel_size

        self.__num_pixels = max(
            utils.MAX_DATA_SIZE // (4 * (kernel_size ** 2) * self.__channel_overlap * 2),
            utils.MIN_DATA_SIZE,
        )

        self.__image_mins = None
        self.__image_maxs = None
        self.__coefficients = None

    def fit(self, selector: Selector):
        """ Fits the model on the given selected tiles.

        Args:
            selector: A tile-selector that has already selected the tiles to
                use.
        """
        self.__image_mins = selector.image_mins
        self.__image_maxs = selector.image_maxs
        self.__coefficients = self._fit(selector.selected_tiles)
        return

    @abc.abstractmethod
    def _init_model(self):
        """ Initialize a model.
        """
        pass

    @property
    def coefficients(self) -> numpy.ndarray:
        """ Returns the matrix of mixing coefficients from the trained model.
        """
        return self.__coefficients

    @property
    def image_mins(self) -> list[int]:
        return self.__image_mins

    @property
    def image_maxs(self) -> list[int]:
        return self.__image_maxs

    def _get_neighbors(self, source_index: int) -> list[int]:
        """ Get the neighboring channels for the given source-channel.
        """
        neighbor_indices = [source_index - i - 1 for i in range(self.__channel_overlap)]
        neighbor_indices.extend(source_index + i + 1 for i in range(self.__channel_overlap))
        neighbor_indices = list(filter(lambda i: 0 <= i < len(self.__files), neighbor_indices))
        return neighbor_indices

    def _get_kernel_indices(self, source_index: int) -> list[int]:
        kernel_size = self.__kernel_size ** 2
        return [
            i * kernel_size + j
            for i in self._get_neighbors(source_index)
            for j in range(kernel_size)
        ]

    def _select_pixels(self, tile: numpy.ndarray) -> numpy.ndarray:
        """ Returns the indices of the brightest few pixels from the given tile.
        """
        return numpy.argsort(tile)[-self.__num_pixels:]

    def _fit_thread(self, source_index: int, selected_tiles: utils.TileIndices) -> list[float]:
        """ Trains a single model on a single source-channel and
         returns the mixing coefficients with the adjacent channels.

        This function can be run inside a thread in a ProcessPoolExecutor.

        Args:
            source_index: Index of the source channel.

        Returns:
            A list of the mixing coefficient with each neighboring channel
             within self.__channel_overlap of the source channel.
        """
        with BioReader(self.__files[source_index], max_workers=utils.NUM_THREADS) as source_reader:
            neighbor_readers = [
                BioReader(self.__files[i], max_workers=utils.NUM_THREADS)
                for i in self._get_neighbors(source_index)
            ]

            mins = [self.image_mins[source_index]]
            mins.extend([self.image_mins[i] for i in self._get_neighbors(source_index)])
            maxs = [self.image_maxs[source_index]]
            maxs.extend([self.image_maxs[i] for i in self._get_neighbors(source_index)])

            logger.info(f'Fitting {self.__class__.__name__} {source_index} on {len(selected_tiles)} tiles...')
            model = self._init_model()
            for i, (_, _, y_min, y_max, x_min, x_max) in enumerate(selected_tiles):
                logger.info(
                    f'Fitting {self.__class__.__name__} {source_index}: '
                    f'Progress: {100 * i / len(selected_tiles):6.2f} %'
                )
                numpy.random.seed(i)

                images: list[numpy.ndarray] = [source_reader[y_min:y_max, x_min:x_max, 0, 0, 0]]
                images.extend((
                    reader[y_min:y_max, x_min:x_max, 0, 0, 0]
                    for reader in neighbor_readers
                ))

                pad = self.__kernel_size // 2
                source_tile = utils.normalize_tile(images[0][pad:-pad, pad:-pad], mins[0], maxs[0]).flatten()

                if source_tile.size > self.__num_pixels:
                    temp_indices = self._select_pixels(source_tile)

                    for image in images:
                        temp_indices = numpy.concatenate(arrays=(
                            temp_indices,
                            self._select_pixels(image[pad:-pad, pad:-pad].flatten())
                        ))

                    indices: numpy.ndarray = numpy.random.permutation(numpy.unique(indices))[-self.__num_pixels:]
                else:
                    indices = numpy.arange(0, source_tile.size)

                tiles: list[numpy.ndarray] = [source_tile[indices]]

                for tile, min_val, max_val in zip(images[1:], mins[1:], maxs[1:]):
                    tile = utils.normalize_tile(tile, min_val, max_val)

                    for r in range(self.__kernel_size):
                        tile_y_min, tile_y_max = r, 1 + r - self.__kernel_size + tile.shape[0]

                        for c in range(self.__kernel_size):
                            tile_x_min, tile_x_max = c, 1 + c - self.__kernel_size + tile.shape[1]

                            cropped_tile = tile[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
                            tiles.append(cropped_tile.flatten()[indices])

                tiles = numpy.asarray(tiles, dtype=numpy.float32).T

                source, neighbors = tiles[:, 0], tiles[:, 1:]
                interactions: numpy.ndarray = numpy.sqrt(numpy.expand_dims(source, axis=1) * neighbors)
                neighbors = numpy.concatenate([neighbors, interactions], axis=1)

                model.fit(neighbors, source)

            coefficients = list(map(float, model.coef_))
            del model
            [reader.close() for reader in neighbor_readers]

        return coefficients

    def _fit(self, selected_tiles: utils.TileIndices) -> numpy.ndarray:
        """ Fits the model on the images and returns a matrix of mixing
            coefficients.
        """

        with ProcessPoolExecutor(max_workers=utils.NUM_THREADS) as executor:
            coefficients_list: list[Future[list[float]]] = [
                executor.submit(self._fit_thread, source_index, selected_tiles)
                for source_index in range(len(self.__files))
            ]
            coefficients_list: list[list[float]] = [future.result() for future in coefficients_list]

        coefficients_matrix = numpy.zeros(
            shape=(len(self.__files), 2 * len(self.__files) * (self.__kernel_size ** 2)),
            dtype=numpy.float32,
        )
        for i, coefficients in enumerate(coefficients_list):
            kernel_indices = self._get_kernel_indices(i)

            interaction_offset = len(self.__files) * (self.__kernel_size ** 2)
            interaction_indices = [interaction_offset + j for j in kernel_indices]
            indices = kernel_indices + interaction_indices

            coefficients_matrix[i, indices] = coefficients

        return coefficients_matrix

    def coefficients_to_csv(self, csv_dir: Path, pattern: str, group: list[utils.FPFileDict]):
        """ Write the matrix of mixing coefficients to a csv.

        TODO: Learn how to use `filepattern` better and cut down on the input
         params. Ideally, we would only need the `destination_dir` as an input
         and the name for the csv would be generated from `self.__files`.

        Args:
            csv_dir: Directory were to write the csv.
            pattern: The pattern used for grouping images.
            group: The group of images for which the csv will be generated.
        """
        # noinspection PyTypeChecker
        name = filepattern.output_name(pattern, group, dict())

        csv_path = csv_dir.joinpath(f'{name}_coefficients.csv')
        with open(csv_path, 'w') as outfile:
            header_1 = ','.join(
                f'c{c}k{k}'
                for c in range(len(self.__files))
                for k in range(self.__kernel_size ** 2)
            )
            header_2 = ','.join(
                f'i{c}k{k}'
                for c in range(len(self.__files))
                for k in range(self.__kernel_size ** 2)
            )
            outfile.write(f'channel,{header_1},{header_2}\n')

            for channel, row in enumerate(self.coefficients):
                row = ','.join(f'{c:.6e}' for c in row)
                outfile.write(f'c{channel},{row}\n')
        return

    def coefficients_from_csv(self, csv_dir: Path, pattern: str, group: list[utils.FPFileDict], selector: Selector):
        # noinspection PyTypeChecker
        name = filepattern.output_name(pattern, group, dict())

        csv_path = csv_dir.joinpath(f'{name}_coefficients.csv')
        with open(csv_path, 'r') as infile:
            infile.readline()  # skip header

            coefficients: list[list[float]] = [
                list(map(float, line.split(',')[1:]))
                for line in infile.readlines()
            ]

        self.__coefficients = numpy.asarray(coefficients, dtype=numpy.float32)

        self.__image_mins = selector.image_mins
        self.__image_maxs = selector.image_maxs
        return

    def write_components(self, destination_dir: Path):
        """ Write out the estimated bleed-through components.

        These bleed-through components can be subtracted from the original
        images to achieve bleed-through correction.

        Args:
            destination_dir: Path to the directory where the output images will
                be written.
        """
        with ProcessPoolExecutor(max_workers=utils.NUM_THREADS) as executor:
            processes = list()
            for source_index, input_path in enumerate(self.__files):
                writer_name = utils.replace_extension(input_path.name)
                processes.append(executor.submit(
                    self._write_components_thread,
                    destination_dir,
                    writer_name,
                    source_index,
                ))

            for process in processes:
                process.result()
        return

    def _write_components_thread(
            self,
            output_dir: Path,
            image_name: str,
            source_index: int,
    ):
        """ Writes the bleed-through components for a single image.

        This function can be run in a single thread in a ProcessPoolExecutor.

        Args:
            output_dir: Path for the directory of the bleed-through components.
            image_name: name of the source image.
            source_index: index of the source channel.
        """
        neighbor_indices = self._get_neighbors(source_index)
        neighbor_mins = [self.image_mins[i] for i in neighbor_indices]
        neighbor_maxs = [self.image_maxs[i] for i in neighbor_indices]

        coefficients = self.__coefficients[source_index]

        neighbor_readers = [
            BioReader(self.__files[i], max_workers=utils.NUM_THREADS)
            for i in neighbor_indices
        ]

        with BioReader(self.__files[source_index], max_workers=utils.NUM_THREADS) as source_reader:
            metadata = source_reader.metadata
            num_tiles = utils.count_tiles_2d(source_reader)
            tile_indices = list(utils.tile_indices_2d(source_reader))

            with BioWriter(
                    output_dir.joinpath(image_name),
                    metadata=metadata,
                    max_workers=utils.NUM_THREADS,
            ) as writer:

                logger.info(f'Writing components for {image_name}...')
                for i, (z, y_min, y_max, x_min, x_max) in enumerate(tile_indices):
                    tile = numpy.squeeze(source_reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])

                    original_component = numpy.zeros_like(tile)

                    if i % 10 == 0:
                        logger.info(f'Writing {image_name}: Progress {100 * i / num_tiles:6.2f} %')

                    all_kernel_indices = numpy.asarray(self._get_kernel_indices(source_index), dtype=numpy.uint64)
                    for neighbor_index, (neighbor_reader, min_val, max_val) in enumerate(zip(
                            neighbor_readers, neighbor_mins, neighbor_maxs
                    )):
                        neighbor_tile = utils.normalize_tile(
                            tile=numpy.squeeze(neighbor_reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0]),
                            min_val=min_val,
                            max_val=max_val,
                        )

                        kernel_size = self.__kernel_size ** 2
                        kernel_indices = all_kernel_indices[
                            kernel_size * neighbor_index:
                            kernel_size * (1 + neighbor_index)
                        ]
                        kernel = coefficients[kernel_indices]
                        kernel = numpy.reshape(kernel, newshape=(self.__kernel_size, self.__kernel_size))

                        if numpy.any(kernel > 0):
                            if self.__kernel_size > 1:
                                smoothed_tile = scipy.ndimage.gaussian_filter(neighbor_tile, 2)
                                smoothed_tile = numpy.min(numpy.dstack((smoothed_tile, neighbor_tile)), axis=-1)
                            else:
                                smoothed_tile = neighbor_tile

                            # apply the coefficient
                            current_component = scipy.ndimage.correlate(smoothed_tile, kernel)

                            # Rescale, but do not add in the minimum value offset.
                            current_component *= (max_val - min_val)
                            original_component += current_component.astype(tile.dtype)

                    # Make sure bleed-through is not higher than the original signal.
                    original_component = numpy.min(numpy.dstack((tile, original_component)), axis=-1)

                    writer[y_min:y_max, x_min:x_max, z:z + 1, 0, 0] = original_component

        [reader.close() for reader in neighbor_readers]
        return


class Lasso(Model):
    """ Uses sklearn.linear_model.Lasso

    This is the model used by the paper and source code (linked below) which we
        used as the seed for this plugin.

    https://doi.org/10.1038/s41467-021-21735-x
    https://github.com/RoysamLab/whole_brain_analysis
    """

    def _init_model(self):
        return linear_model.Lasso(alpha=1e-4, copy_X=True, positive=True, warm_start=True, max_iter=10)


class ElasticNet(Model):
    """ Uses sklearn.linear_model.ElasticNet
    """

    def _init_model(self):
        return linear_model.ElasticNet(alpha=1e-4, warm_start=True)


class PoissonGLM(Model):
    """ Uses sklearn.linear_model.PoissonRegressor
    """

    def _init_model(self):
        return linear_model.PoissonRegressor(alpha=1e-4, warm_start=True)


""" A dictionary to let us use a model by name.
"""
MODELS: dict[str, Type[Model]] = {
    'Lasso': Lasso,
    'PoissonGLM': PoissonGLM,
    'ElasticNet': ElasticNet,
}
