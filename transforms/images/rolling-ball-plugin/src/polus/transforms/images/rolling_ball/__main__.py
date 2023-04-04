import typer
import logging
from multiprocessing import cpu_count
from pathlib import Path
from os import environ

from bfio.bfio import BioReader
from bfio.bfio import BioWriter

from src.polus.transforms.images.rolling_ball.rolling_ball import rolling_ball

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)

def _main(
        input_dir: Path,
        ball_radius: int,
        light_background: bool,
        output_dir: Path,
) -> None:
    """ Main execution function.

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

def main(
        input_dir: Path = typer.Option(..., "--inputDir", "-i", help="Input image collection to be processed by this plugin."),
        ball_radius: int = typer.Option(25, "--ballRadius", "-r", help="Radius of the ball used to perform background subtraction."),
        light_background: bool = typer.Option(False, "--lightBackground", "-l", help="Whether the image has a light or dark background."),
        output_dir: Path = typer.Option(..., "--outputDir", "-o", help="Output collection.")
    ):
    """A WIPP plugin to perform background subtraction using the rolling-ball algorithm."""

    logger.info("Parsing arguments...")

    if input_dir.joinpath('images').is_dir():
        # switch to images folder if present
        input_dir = input_dir.joinpath('images').resolve()
    logger.info(f'inputDir = {input_dir}')

    logger.info(f'ballRadius = {ball_radius}')

    logger.info(f'lightBackground = {light_background}')

    logger.info(f'outputDir = {output_dir}')

    _main(
        input_dir,
        ball_radius,
        light_background,
        output_dir
    )

typer.run(main)
