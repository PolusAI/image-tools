import argparse
import logging
import pickle
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from pathlib import Path

import filepattern
import matplotlib.pylab
import numpy
import scipy.stats
import seaborn
from bfio import BioReader
from bfio import BioWriter
from filepattern import FilePattern

import tile_selectors
import utils

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('bench')
logger.setLevel(utils.POLUS_LOG)


def write_corrected_images(
        *,
        group: list[utils.FPFileDict],
        channel_ordering: list[int],
        components_dir: Path,
        output_dir: Path,
):
    logger.info(f'writing corrected images...')

    files = [file['file'] for file in group]
    if len(channel_ordering) == 0:
        channel_ordering = list(range(len(files)))
    files = [files[c] for c in channel_ordering]

    for image_path in files:

        component_path = components_dir.joinpath(image_path.name)
        assert component_path.exists()

        output_path = output_dir.joinpath(image_path.name)
        if output_path.exists():
            continue
        logger.info(f'writing image {image_path.name}...')

        with BioReader(image_path) as image_reader, BioReader(component_path) as component_reader:
            with BioWriter(output_path, metadata=image_reader.metadata) as writer:

                for y_min in range(0, writer.Y, utils.TILE_SIZE_2D):
                    y_max = min(writer.Y, y_min + utils.TILE_SIZE_2D)

                    for x_min in range(0, writer.X, utils.TILE_SIZE_2D):
                        x_max = min(writer.X, x_min + utils.TILE_SIZE_2D)

                        image_tile = numpy.squeeze(image_reader[y_min:y_max, x_min:x_max, 0, 0, 0])
                        component_tile = numpy.squeeze(component_reader[y_min:y_max, x_min:x_max, 0, 0, 0])

                        writer[y_min:y_max, x_min:x_max, 0, 0, 0] = image_tile - component_tile

    return


def _estimate_pearson_coefficient(
        path_1: Path,
        path_2: Path,
        selected_tiles: utils.TileIndices,
) -> float:
    total_r = 0

    with BioReader(path_1) as reader_1, BioReader(path_2) as reader_2:
        for _, _, y_min, y_max, x_min, x_max in selected_tiles:
            tile_1 = numpy.squeeze(reader_1[y_min:y_max, x_min:x_max, 0, 0, 0]).flatten()
            tile_2 = numpy.squeeze(reader_2[y_min:y_max, x_min:x_max, 0, 0, 0]).flatten()

            random_indices = numpy.random.permutation(tile_1.shape[0])[:100_000]
            tile_1 = tile_1[random_indices]
            tile_2 = tile_2[random_indices]

            total_r += scipy.stats.pearsonr(tile_1, tile_2)[0]

    return total_r / len(selected_tiles)


def _pearson_thread(
        path_1: Path,
        path_2: Path,
        selected_tiles: utils.TileIndices,
        corrected_dir: Path,
) -> tuple[float, float]:
    r_before = _estimate_pearson_coefficient(path_1, path_2, selected_tiles)

    corrected_path_1 = corrected_dir.joinpath(path_1.name)
    corrected_path_2 = corrected_dir.joinpath(path_2.name)
    r_after = _estimate_pearson_coefficient(corrected_path_1, corrected_path_2, selected_tiles)

    return r_before, r_after


def _select_tiles(
        files: list[Path],
        selector_name: str,
        tiles_path: Path,
):
    logger.info('selecting tiles...')
    selector = tile_selectors.SELECTORS[selector_name](files, num_tiles_per_channel=10)
    selector.fit()

    with open(tiles_path, 'wb') as outfile:
        pickle.dump(selector.selected_tiles, outfile)

    return


def plot_pearson_coefficients(
        *,
        group: list[utils.FPFileDict],
        pattern: str,
        channel_ordering: list[int],
        corrected_dir: Path,
        selector_name: str,
        heatmaps_dir: Path,
        replicate: int,
):
    files = [file['file'] for file in group]
    if len(channel_ordering) == 0:
        channel_ordering = list(range(len(files)))
    files = [files[c] for c in channel_ordering]

    # noinspection PyTypeChecker
    heatmaps_name: str = filepattern.output_name(pattern, group, dict()).split('.')[0]

    tiles_path = heatmaps_dir.joinpath(f'tiles_{heatmaps_name}.pickle')
    if not tiles_path.exists():
        _select_tiles(files, selector_name, tiles_path)
    with open(tiles_path, 'rb') as infile:
        selected_tiles: utils.TileIndices = pickle.load(infile)
    logger.info(f'selected {len(selected_tiles)} tiles...')

    before_path = heatmaps_dir.joinpath(f'before_{heatmaps_name}.pickle')
    after_path = heatmaps_dir.joinpath(f'after_{heatmaps_name}.pickle')
    coefficients_before = numpy.zeros(shape=(len(files), len(files)), dtype=numpy.float32)
    coefficients_after = numpy.zeros(shape=(len(files), len(files)), dtype=numpy.float32)

    if not (before_path.exists() and after_path.exists()):
        processes: dict[tuple[int, int], Future[tuple[float, float]]] = dict()
        with ProcessPoolExecutor(max_workers=utils.NUM_THREADS) as executor:
            for i, image_path_1 in enumerate(files):
                for j, image_path_2 in enumerate(files[i + 1:], start=i + 1):
                    processes[(i, j)] = executor.submit(
                        _pearson_thread,
                        image_path_1,
                        image_path_2,
                        selected_tiles,
                        corrected_dir,
                    )

            logger.info(f'calculating pearson coefficients...')
            process_values = list(processes.values())
            done, not_done = wait(process_values, 0)
            while len(not_done) > 0:
                logger.info(f'Percent complete: {100 * len(done) / len(process_values):6.3f}%')
                for process in done:
                    process.result()
                done, not_done = wait(process_values, 15)
            logger.info(f'Percent complete: {100 * len(done) / len(process_values):6.3f}%')

        for (i, j), future in processes.items():
            before, after = future.result()
            coefficients_before[i, j] = coefficients_before[j, i] = before
            coefficients_after[i, j] = coefficients_before[j, i] = after

        with open(before_path, 'wb') as outfile:
            pickle.dump(coefficients_before, outfile)
        with open(after_path, 'wb') as outfile:
            pickle.dump(coefficients_after, outfile)

    else:
        with open(before_path, 'rb') as infile:
            coefficients_before = pickle.load(infile)
        with open(after_path, 'rb') as infile:
            coefficients_after = pickle.load(infile)

    draw_heatmap(coefficients_before, coefficients_after, heatmaps_dir, heatmaps_name, replicate)

    return


def draw_heatmap(
        before: numpy.ndarray,
        after: numpy.ndarray,
        heatmaps_dir: Path,
        heatmaps_name: str,
        replicate: int,
):
    heat = numpy.zeros_like(before, dtype=numpy.float32)
    heat[numpy.tril_indices_from(heat)] = after[numpy.tril_indices_from(after)]
    heat[numpy.triu_indices_from(heat)] = before[numpy.triu_indices_from(before)]
    heat = heat.T

    mask = numpy.zeros_like(heat, dtype=bool)
    mask[numpy.diag_indices_from(mask)] = True

    with seaborn.axes_style('white'):
        _ = seaborn.heatmap(
            heat, mask=mask, vmin=-1, vmax=1, cmap='vlag', linewidth=0.5,
            annot=True, fmt='.3f', annot_kws={'size': 8},
        )
        matplotlib.pylab.title(f'Replicate {replicate}')

        plot_path = heatmaps_dir.joinpath(f'{heatmaps_name}.png')
        matplotlib.pylab.savefig(str(plot_path), dpi=300)

    matplotlib.pylab.close('all')

    return


def bench(
        input_dir: Path,
        output_dir: Path,
        file_pattern: str,
        replicate: int,
):
    components_dir = output_dir.joinpath('components')
    components_dir.mkdir(exist_ok=True)

    coefficients_dir = output_dir.joinpath('csvs')
    coefficients_dir.mkdir(exist_ok=True)

    corrected_dir = output_dir.joinpath('images')
    corrected_dir.mkdir(exist_ok=True)

    heatmaps_dir = output_dir.joinpath('plots')
    heatmaps_dir.mkdir(exist_ok=True)

    channel_ordering = [1, 0, 3, 2, 4, 5, 7, 6, 8, 9]
    fp = FilePattern(input_dir, file_pattern)
    for group in fp(['c']):
        write_corrected_images(
            group=group,
            channel_ordering=channel_ordering,
            components_dir=components_dir,
            output_dir=corrected_dir,
        )
        plot_pearson_coefficients(
            group=group,
            pattern=file_pattern,
            channel_ordering=channel_ordering,
            corrected_dir=corrected_dir,
            selector_name='HighMeanIntensity',
            heatmaps_dir=heatmaps_dir,
            replicate=replicate,
        )
    return


if __name__ == '__main__':
    logger.info("Parsing arguments...")
    _parser = argparse.ArgumentParser(
        prog='bench',
        description='Recreate benchmarks from paper.',
    )

    """ Define the arguments """
    _parser.add_argument(
        '--input-dir', dest='input_dir', type=str, required=True,
        help='Path to input images.',
    )

    _parser.add_argument(
        '--output-dir', dest='output_dir', type=str, required=True,
        help='Input file name pattern.',
    )

    _args = _parser.parse_args()
    _error_messages = list()

    _input_dir = Path(_args.input_dir).resolve()
    assert _input_dir.exists() and _input_dir.is_dir()

    _output_dir = Path(_args.output_dir).resolve()
    assert _output_dir.exists() and _output_dir.is_dir()

    # TODO: Figure out filepatterns for raw data from Maric repo instead of polus-data repo.
    #  Alternatively, make the ratbrain data in polus-data into a public dataset.
    _patterns = [
        'r001_c{ccc}_z000.ome.tif',
        'r002_c{ccc}_z000.ome.tif',
        'r003_c{ccc}_z000.ome.tif',
        'r004_c{ccc}_z000.ome.tif',
        'r005_c{ccc}_z000.ome.tif',
    ]

    for _i, _pattern in enumerate(_patterns, start=1):
        bench(_input_dir, _output_dir, _pattern, _i)
