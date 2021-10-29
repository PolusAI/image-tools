import logging, argparse
import os 

import utils
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tensorflow.python.client import device_lib

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(image_dir_train : str,
         label_dir_train : str,
         image_dir_valid : str,
         label_dir_valid : str,
         output_directory : str,
         gpu : bool,
         M : int,
         epochs : int,
         imagepattern: str):

    utils.train_nn(image_dir_train  = image_dir_train,
                   label_dir_train  = label_dir_train,
                   image_dir_valid  = image_dir_valid,
                   label_dir_valid  = label_dir_valid,
                   output_directory = output_directory,
                   gpu              = gpulist,
                   M                = M,
                   epochs           = epochs,
                   imagepattern     = imagepattern)

if __name__ == "__main__":

    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    # Key for NPZ file OR Pathway to Directories
    parser.add_argument('--inpImageTrain', dest='input_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=False)
    parser.add_argument('--inpLabelTrain', dest='input_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=False)
    parser.add_argument('--inpImageValid', dest='input_images_valid', type=str,
                        help='Path to folder with intesity based images for validation', required=False)
    parser.add_argument('--inpLabelValid', dest='input_labels_valid', type=str,
                        help='Path to folder with labelled segments, ground truth for validation', required=False)
    
    parser.add_argument('--controlPoints', dest='controlPoints', type=int,
                        help='Define the number of control points', required=False)
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='The number of epochs to run', required=False)

    # Parameters not directly related to Building a Model
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)

    # Parse the arguments
    args = parser.parse_args()

    image_train = args.input_images_train
    label_train = args.input_labels_train
    image_valid = args.input_images_valid
    label_valid = args.input_labels_valid

    M = args.controlPoints
    epochs = args.epochs
    
    # Parameters not directly related to Model Building
    output_directory = args.output_directory
    imagepattern = args.image_pattern

    # If there is a GPU to use, then use it
    gpu = False
    local_device_protos = device_lib.list_local_devices()
    gpulist = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpulist) > 0:
        gpu = True

    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("Output Directory: {}".format(output_directory))

    logger.info("Input Training Directory for Intensity Based Images: {}".format(image_train))
    logger.info("Input Training Directory for Labelled Images: {}".format(label_train))
    logger.info("Input Validation Directory for Intensity Based Images: {}".format(image_valid))
    logger.info("Input Validation Directory for Labelled Images: {}".format(label_valid))
        
    logger.info("Number of Control Points {}".format(M))
    logger.info("Number of Epochs {}".format(epochs))

    logger.info("Is there a GPU? {}".format(gpu))

    if imagepattern == None:
        imagepattern = '.*'

    main(image_dir_train  = image_train,
         label_dir_train  = label_train,
         image_dir_valid   = image_valid,
         label_dir_valid   = label_valid,
         output_directory = output_directory,
         gpu              = gpu,
         M                = M,
         epochs           = epochs,
         imagepattern     = imagepattern)


