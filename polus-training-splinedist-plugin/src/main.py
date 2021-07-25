import logging, argparse
import os 

import utils
from tensorflow.python.client import device_lib

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(image_dir_train : str,
         label_dir_train : str,
         split_percentile : int,
         output_directory : str,
         gpu : bool,
         M : int,
         epochs : int,
         imagepattern: str):

    utils.train_nn(image_dir_input=image_dir_train,
                    label_dir_input=label_dir_train,
                    split_percentile=split_percentile,
                    output_directory=output_directory,
                    gpu=gpu,
                    imagepattern=imagepattern,
                    M=M,
                    epochs=epochs)


if __name__ == "__main__":

    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDirTrain', dest='input_directory_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=False)
    parser.add_argument('--inpLabelDirTrain', dest='input_directory_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=False)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=False)
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
    
    # for training neural network
    image_dir_train = args.input_directory_images_train
    label_dir_train = args.input_directory_labels_train
    split_percentile = args.split_percentile
    M = args.controlPoints
    epochs = args.epochs
    
    # Parameters not directly related to Model Building
    output_directory = args.output_directory
    imagepattern = args.image_pattern

    # exclusively define:
    # split_percentile OR (image_dir_test and label_dir_test)
    if split_percentile == None:
        assert image_dir_train != None
        assert label_dir_train != None
    
    if split_percentile != None:
        assert image_dir_train == None
        assert label_dir_train == None

    # If there is a GPU to use, then use it
    gpu = False
    local_device_protos = device_lib.list_local_devices()
    gpulist = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpulist) > 0:
        gpu = True

    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Action: {} a neural network".format(action))

    if split_percentile == None:
        logger.info("Input Training Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Training Directory for Labelled Images: {}".format(label_dir_train))
        
    else:
        logger.info("Input Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Directory for Labelled Images: {}".format(label_dir_train))
        logger.info("Splitting Input Directory into {}:{} Ratio for Training:Testing".format(split_percentile, 100-split_percentile))
    
    logger.info("Number of Control Points {}".format(M))
    logger.info("Number of Epochs {}".format(epochs))

    logger.info("Is there a GPU? {}".format(gpu))

    if imagepattern == None:
        imagepattern = '.*'

    main(image_dir_train=image_dir_train,
         label_dir_train=label_dir_train,
         split_percentile=split_percentile,
         output_directory=output_directory,
         gpu=gpu,
         M=M,
         epochs=epochs,
         imagepattern=imagepattern)


