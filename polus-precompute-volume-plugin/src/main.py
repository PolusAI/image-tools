import logging, argparse, time, multiprocessing, traceback

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat

import utils
import os

from filepattern import FilePattern as fp

import numpy as np

<<<<<<< HEAD
# Import environment variables, if POLUS_LOG empty then automatically sets to INFO
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))

=======
>>>>>>> 40c934ac1c15f51043855a01b4bd5dbb7eb6eeb0
# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
<<<<<<< HEAD
logger.setLevel(POLUS_LOG)
=======
logger.setLevel(logging.INFO)
>>>>>>> 40c934ac1c15f51043855a01b4bd5dbb7eb6eeb0

def main(input_dir: str,
         output_dir: str,
         imagetype: str,
         filepattern: str,
         mesh: bool):
    
    # Get list of images that we are going to through
    # Get list of output paths for every image
    logger.info("\n Getting the {}s...".format(imagetype))
    fp_images = fp(input_dir,filepattern)
<<<<<<< HEAD
    
    input_images = [str(f[0]['file']) for f in fp_images]
    output_images = [os.path.join(output_dir, os.path.basename(f)) for f in input_images]
    num_images = len(input_images)

    for image in range(num_images):
        utils.build_pyramid(input_image=input_images[image],
                            output_image=output_images[image],
                            imagetype = imagetype,
                            mesh = mesh)
=======
    input_images = [str(f[0]['file']) for f in fp_images]
    output_images = [os.path.join(output_dir, os.path.basename(f)) for f in input_images]
    
    # Build one pyramid for each image in the input directory
        # Max of 2 workers since building individual pyrmaids allocates
        # more CPUS as well. 
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(utils.build_pyramid, input_images, output_images, repeat(imagetype), repeat(mesh))

>>>>>>> 40c934ac1c15f51043855a01b4bd5dbb7eb6eeb0

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

    # If plugin generates an image or metadata subdirectories, then it 
     # reroute the input_dir to the images directory
    if os.path.exists(os.path.join(input_dir, "images")):
        input_dir = os.path.join(input_dir, "images")

    # There are only two types of inputs
    assert imagetype == "segmentation" or imagetype == "image"

    if imagetype != 'segmentation' and mesh == True:
        logger.warning("Can only generate meshes if imageType is segmentation")

    logger.info('Input Directory = {}'.format(input_dir))
    logger.info('Output Directory = {}'.format(output_dir))
    logger.info('Image Type = {}'.format(imagetype))
    logger.info('Image Pattern = {}'.format(filepattern))
    logger.info('Mesh = {}'.format(mesh))

    if filepattern == None:
        filepattern = ".*"

    main(input_dir=input_dir,
         output_dir=output_dir,
         imagetype=imagetype,
         filepattern=filepattern,
         mesh=mesh)
