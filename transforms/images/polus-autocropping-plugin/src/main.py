import argparse
import logging
from pathlib import Path

import filepattern

from autocrop import crop_image_group
from utils import constants

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("main")
logger.setLevel(constants.POLUS_LOG)

if __name__ == "__main__":
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='main',
        description='Autocropping 2d and 3d images by estimating entropy of rows, columns, and z-slices.',
    )

    # Input arguments
    parser.add_argument('--inputDir', dest='inputDir', type=str, required=True,
                        help='Input image collection to be processed by this plugin.')

    parser.add_argument('--filePattern', dest='filePattern', type=str, required=True,
                        help='File pattern to use for grouping images.')

    parser.add_argument('--groupBy', dest='groupBy', type=str, required=True,
                        help='Variables to use for grouping images. Each group is cropped to the same bounding-box.')

    parser.add_argument('--cropX', dest='cropX', type=str, required=False, default='true',
                        help='Whether to crop along the x-axis.')

    parser.add_argument('--cropY', dest='cropY', type=str, required=False, default='true',
                        help='Whether to crop along the y-axis.')

    parser.add_argument('--cropZ', dest='cropZ', type=str, required=False, default='true',
                        help='Whether to crop along the z-axis.')

    parser.add_argument('--smoothing', dest='smoothing', type=str, required=False, default='true',
                        help='Whether to use gaussian smoothing on images to add more tolerance to noise.')

    # Output arguments
    parser.add_argument('--outputDir', dest='outputDir', type=str, required=True,
                        help='Output collection.')

    # Parse the arguments
    args = parser.parse_args()
    error_messages = list()

    input_dir = Path(args.inputDir).resolve()
    if input_dir.joinpath('images').is_dir():
        # switch to images folder if present
        input_dir = input_dir.joinpath('images')
    if not input_dir.exists():
        error_messages.append(f'inputDir {input_dir} does not exist.')

    pattern = args.filePattern

    group_by = args.groupBy
    if len(set(group_by) - set(filepattern.VARIABLES)) > 0:
        error_messages.append(
            f'groupBy variables must be from among {list(filepattern.VARIABLES)}. '
            f'Got {group_by} instead...'
        )

    crop_x = args.cropX
    if crop_x in {'true', 'false'}:
        crop_x = (crop_x == 'true')
    else:
        error_messages.append('cropX must be either \'true\' or \'false\'')

    crop_y = args.cropY
    if crop_y in {'true', 'false'}:
        crop_y = (crop_y == 'true')
    else:
        error_messages.append('cropY must be either \'true\' or \'false\'')

    crop_z = args.cropZ
    if crop_z in {'true', 'false'}:
        crop_z = (crop_z == 'true')
    else:
        error_messages.append('cropZ must be either \'true\' or \'false\'')

    smoothing = args.smoothing
    if smoothing in {'true', 'false'}:
        smoothing = smoothing == 'true'
    else:
        error_messages.append('smoothing must be either \'true\' or \'false\'')

    output_dir = Path(args.outputDir).resolve()
    if not output_dir.exists():
        error_messages.append(f'outputDir {output_dir} does not exist.')

    if len(error_messages) > 0:
        error_messages.append('See the README for more details on what these parameters should be.')
        message = f'Oh no! Something went wrong:\n' + '\n'.join(error_messages)
        logger.error(message)
        raise ValueError(message)
    else:
        logger.info(f'inputDir = {input_dir}')
        logger.info(f'filePattern = {pattern}')
        logger.info(f'groupBy = {group_by}')
        logger.info(f'cropX = {crop_x}')
        logger.info(f'cropY = {crop_y}')
        logger.info(f'cropZ = {crop_z}')
        logger.info(f'smoothing = {smoothing}')
        logger.info(f'outputDir = {output_dir}')

        fp = filepattern.FilePattern(input_dir, pattern)
        groups = list(fp(group_by=group_by))
        for i, group in enumerate(groups):
            if len(group) == 0:
                continue
            file_paths = [files['file'] for files in group]
            logger.info(f'Working on group {i + 1}/{len(groups)} containing {len(file_paths)} images...')
            crop_image_group(
                file_paths=file_paths,
                crop_axes=(crop_x, crop_y, crop_z),
                smoothing=smoothing,
                output_dir=output_dir,
            )
