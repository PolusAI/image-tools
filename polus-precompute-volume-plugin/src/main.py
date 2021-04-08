import logging, argparse, time, multiprocessing, traceback

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

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
         imagepattern: str,
         mesh: bool):
    
    # Get list of images that we are going to through
    logger.info("\n Getting the images...")
    fp_images = fp(Path(input_dir),imagepattern)
    images = [os.path.basename(i[0]['file']) for i in fp_images() 
             if os.path.exists(os.path.join(input_dir, os.path.basename(i[0]['file'])))]
    images.sort()
    num_images = len(images)


    # Build one pyramid for each image in the input directory
    # Each stack is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    for image in images:
        p = Process(target=utils.build_pyramid, args=(os.path.join(input_dir, image), os.path.join(output_dir, image), imagetype, mesh))
        p.start()
        p.join()

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
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--mesh', dest='mesh', type=bool,
                        default=False, help='True or False for creating meshes', required=False)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    imagetype = args.image_type
    imagepattern = args.image_pattern
    mesh = args.mesh

    if imagetype != 'segmentation' and mesh == True:
        logger.warning("Can only generate meshes if imageType is segmentation")

    logger.info('Input Directory = {}'.format(input_dir))
    logger.info('Output Directory = {}'.format(output_dir))
    logger.info('Image Type = {}'.format(imagetype))
    logger.info('Image Pattern = {}'.format(imagepattern))
    logger.info('Mesh = {}'.format(mesh))

    if imagepattern == None:
        imagepattern = ".*"

    main(input_dir,
         output_dir,
         imagetype,
         imagepattern,
         mesh)
