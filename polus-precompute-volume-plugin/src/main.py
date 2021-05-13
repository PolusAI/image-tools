import logging, argparse, time, multiprocessing, traceback

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat

import utils
import os
from pathlib import Path

import filepattern
from filepattern import FilePattern as fp

import numpy as np

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(input_dir: str,
         output_dir: str,
         imagetype: str,
         filepattern: str,
         mesh: bool):
    
    # Get list of images that we are going to through
    # Get list of output paths for every image
    logger.info("\n Getting the images...")
    fp_images = fp(Path(input_dir),filepattern)
    input_images = []
    output_images = []
    for i in fp_images():
        image = i[0]['file']
        if os.path.exists(image):
            input_images.append(image)
            output_images.append(os.path.join(output_dir, os.path.basename(image)))
    assert len(input_images) == len(output_images)

    # Build one pyramid for each image in the input directory
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(utils.build_pyramid, input_images, output_images, repeat(imagetype), repeat(mesh))


if __name__ == "__main__":

    # Setup the Argument parsing
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Generate a precomputed slice for Polus Volume Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='The type of image, image or segmentation', required=True)
    parser.add_argument('--filePattern', dest='file_pattern', type=str,
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--mesh', dest='mesh', type=bool,
                        default=False, help='True or False for creating meshes', required=False)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    imagetype = args.image_type
    filepattern = args.file_pattern
    mesh = args.mesh

    if imagetype != 'segmentation' and mesh == True:
        logger.warning("Can only generate meshes if imageType is segmentation")

    logger.info('Input Directory = {}'.format(input_dir))
    logger.info('Output Directory = {}'.format(output_dir))
    logger.info('Image Type = {}'.format(imagetype))
    logger.info('Image Pattern = {}'.format(filepattern))
    logger.info('Mesh = {}'.format(mesh))

    if filepattern == None:
        filepattern = ".*"

    main(input_dir,
         output_dir,
         imagetype,
         filepattern,
         mesh)
