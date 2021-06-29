import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path

from bfio.bfio import BioReader
from bfio.bfio import BioWriter

from rolling_ball import rolling_ball

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main(
        input_dir: Path,
        ball_radius: int,
        light_background: bool,
        output_dir: Path,
) -> None:
    """ Main execution function

    Args:
        input_dir: path to directory containing the input images.
        ball_radius: radius of ball to use for the rolling-ball algorithm.
        light_background: whether the image has a light or dark background.
        output_dir: path to directory where to store the output images.
    """

    for in_path in input_dir.iterdir():
        in_path = Path(in_path)
        out_path = Path(output_dir).joinpath(in_path.name)

        # Load the input image
        with BioReader(in_path) as reader:
            logger.info(f'Working on {in_path.name} with shape {reader.shape}')

            # Initialize the output image
            with BioWriter(out_path, metadata=reader.metadata, max_workers=cpu_count()) as writer:
                rolling_ball(
                    reader=reader,
                    writer=writer,
                    ball_radius=ball_radius,
                    light_background=light_background,
                )
    return


if __name__ == "__main__":
    """ Argument parsing """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='A WIPP plugin to perform background subtraction using the rolling-ball algorithm.')
    
    # Input arguments
    parser.add_argument(
        '--inputDir',
        dest='input_dir',
        type=str,
        help='Input image collection to be processed by this plugin.',
        required=True,
    )
    parser.add_argument(
        '--ballRadius',
        dest='ball_radius',
        type=str,
        default='25',
        help='Radius of the ball used to perform background subtraction.',
        required=False,
    )
    parser.add_argument(
        '--lightBackground',
        dest='light_background',
        type=str,
        default='false',
        help='Whether the image has a light or dark background.',
        required=False,
    )
    # Output arguments
    parser.add_argument(
        '--outputDir',
        dest='output_dir',
        type=str,
        help='Output collection',
        required=True,
    )
    
    # Parse the arguments
    args = parser.parse_args()

    _input_dir = Path(args.input_dir).resolve()
    if _input_dir.joinpath('images').is_dir():
        # switch to images folder if present
        _input_dir = _input_dir.joinpath('images').resolve()
    logger.info(f'inputDir = {_input_dir}')

    _ball_radius = int(args.ball_radius)
    logger.info(f'ballRadius = {_ball_radius}')

    _light_background = args.light_background
    if _light_background in {'true', 'false'}:
        _light_background = (_light_background == 'true')
    else:
        raise ValueError(f'lightBackground must be either \'true\' or \'false\'')
    logger.info(f'lightBackground = {_light_background}')

    _output_dir = args.output_dir
    logger.info(f'outputDir = {_output_dir}')

    main(
        input_dir=_input_dir,
        ball_radius=_ball_radius,
        light_background=_light_background,
        output_dir=_output_dir,
    )
