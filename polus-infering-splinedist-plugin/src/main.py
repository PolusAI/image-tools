import logging, argparse
import os 

import utils

from tensorflow.python.client import device_lib

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(image_dir : str,
         base_dir : str,
         imagepattern : str,
         gpu : bool,
         output_directory : str):

    utils.predict_nn(image_dir=image_dir,
                    base_dir=base_dir,
                    output_directory=output_directory,
                    gpu=gpu,
                    imagepattern=imagepattern)

if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDir', dest='input_images_dir', type=str,
                        help='Path to folder with intesity based images for training', required=True)
    parser.add_argument('--inpBaseDir', dest='input_base_dir', type=str,
                        help='Path to folder with weights for neural network', required=True)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory where plots are saved', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)

    # Parse the arguments
    args = parser.parse_args()
    image_dir = args.input_images_dir
    base_dir = args.input_base_dir
    output_directory = args.output_directory
    imagepattern = args.image_pattern
    
    gpu = False
    local_device_protos = device_lib.list_local_devices()
    gpulist = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpulist) > 0:
        gpu = True
    
    logger.info("Input Directory containing Images: {}".format(image_dir))
    logger.info("Input Directory containing Parameters: {}".format(base_dir))
    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("Use GPU: {}".format(gpu))
    logger.info("Output Directory to save Results: {}".format(output_directory))

    main(image_dir = image_dir,
         base_dir = base_dir,
         imagepattern = imagepattern,
         gpu = gpu,
         output_directory = output_directory)


