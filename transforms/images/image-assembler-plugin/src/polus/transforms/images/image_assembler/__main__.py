from polus.transforms.images.image_assembler.main import main as m

# Base packages
import logging
import typer
from os import environ
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)
logging.getLogger("bfio").setLevel(POLUS_LOG)

def main(imgPath: Path = typer.Option(..., "--imgPath", "-i", help="Absolute path to the input image collection directory to be processed by this plugin."),
         stitchPath : Path = typer.Option(..., "--stitchPath", "-s", help="Absolute path to a stitching vector directory."),
         outDir : Path = typer.Option(..., "--outDir", "-o",  help="Absolute path to the output collection directory."),
         timesliceNaming : bool = typer.Option(False, "--timesliceNaming", "-t", help="Use timeslice number as image name.")
         ):
    """Assemble images from a single stitching vector."""

    # if the input image collection contains a images subdirectory, we use that
    if imgPath.joinpath('images').is_dir():
        imgPath = imgPath.joinpath('images')

    logger.info('imgPath: {}'.format(imgPath))
    logger.info('stitchPath: {}'.format(stitchPath))
    logger.info('outDir: {}'.format(outDir))
    logger.info('timesliceNaming: {}'.format(timesliceNaming))


    m(imgPath,
        stitchPath,
        outDir,
        timesliceNaming)

typer.run(main)
