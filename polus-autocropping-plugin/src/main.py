import argparse
import logging
from pathlib import Path
from typing import Set

import filepattern

from autocrop import crop_image_group

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main(
        input_dir: Path,
        pattern: str,
        group_by: str,
        axes: Set[str],
        extension: str,
        smoothing: bool,
        output_dir: Path,
):
    """ Main execution function

    Args:
        input_dir: path to directory containing the input images.
        pattern: FilePattern of input images.
        group_by: The variables by which to group files.
        axes: Cropping rows, cols or both?
        extension: The extension to use when writing the resulting images.
                    Must be either '.ome.tif' or '.ome.zarr'.
        smoothing: Whether to use gaussian smoothing.
        output_dir: path to directory where to store the output images.
    """
    fp = filepattern.FilePattern(input_dir, pattern)
    groups = list(fp(group_by=list(group_by)))
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        file_paths = [files['file'] for files in group]
        logger.info(f'Working on group {i + 1}/{len(groups)} containing {len(file_paths)} images...')
        crop_image_group(file_paths, axes, extension, smoothing, output_dir)
    return


if __name__ == "__main__":
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='main',
        description='Autocropping Plugin.',
    )
    
    # Input arguments
    parser.add_argument(
        '--inputDir',
        dest='inputDir',
        type=str,
        help='Input image collection to be processed by this plugin.',
        required=True,
    )

    parser.add_argument(
        '--filePattern',
        dest='filePattern',
        type=str,
        help='File pattern to use for grouping images.',
        required=True,
    )

    parser.add_argument(
        '--groupBy',
        dest='groupBy',
        type=str,
        help='Which file-pattern variables to use for grouping images.',
        required=True,
    )

    parser.add_argument(
        '--axes',
        dest='axes',
        type=str,
        help='Whether to crop rows, columns or both.',
        required=False,
        default='both',
    )

    parser.add_argument(
        '--smoothing',
        dest='smoothing',
        type=str,
        help='Whether to use gaussian smoothing on images to add more tolerance to noise.',
        required=False,
        default='true',
    )

    # Output arguments
    parser.add_argument(
        '--outputDir',
        dest='outputDir',
        type=str,
        help='Output collection.',
        required=True,
    )
    
    # Parse the arguments
    args = parser.parse_args()

    _input_dir = Path(args.inputDir).resolve()
    if Path.is_dir(Path(args.inputDir).joinpath('images')):
        # switch to images folder if present
        _input_dir = Path(args.inputDir).joinpath('images').resolve()
    logger.info(f'inputDir = {_input_dir}')

    _pattern = args.filePattern
    logger.info(f'filePattern = {_pattern}')

    _group_by = args.groupBy
    if len(set(_group_by) - set(filepattern.VARIABLES)) > 0:
        _message = f'groupBy variables must be from among {list(filepattern.VARIABLES)}. Got {_group_by} instead...'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'groupBy = {_group_by}')

    _axes = args.axes
    if _axes in {'rows', 'cols'}:
        _axes = {_axes}
    elif _axes == 'both':
        _axes = {'rows', 'cols'}
    else:
        _message = f'axes must be one of \'rows\', \'cols\' or \'both\'.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'axes = {_axes}')

    # TODO: Verify that I even need to check file-name extensions
    # TODO: Convert this to an environment variable that can be set in a docker
    #  container in WIPP.
    _extension = '.ome.tif'
    if _extension not in {'.ome.tif', '.ome.zarr'}:
        _message = 'extension must be either \'.ome.tif\' or \'.ome.zarr\''
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'extension = {_extension}')

    _smoothing = args.smoothing
    if _smoothing in {'true', 'false'}:
        _smoothing = _smoothing == 'true'
    else:
        _message = 'smoothing must be either \'true\' or \'false\''
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'smoothing = {_smoothing}')

    _output_dir = Path(args.outputDir).resolve()
    logger.info(f'outputDir = {_output_dir}')

    main(
        input_dir=_input_dir,
        pattern=_pattern,
        group_by=_group_by,
        axes=_axes,
        extension=_extension,
        smoothing=_smoothing,
        output_dir=_output_dir,
    )
