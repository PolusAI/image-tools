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
         image_dir_test : str,
         label_dir_test : str,
         trained_imageDir : str,
         trained_labelDir : str,
         modeldir : str,
         split_percentile : int,
         action : str,
         output_directory : str,
         gpu : bool,
         M : int,
         epochs : int,
         imagepattern: str):


    if action == 'training':
        utils.train_nn(image_dir_input=image_dir_train,
                       label_dir_input=label_dir_train,
                       image_dir_test=image_dir_test,
                       label_dir_test=label_dir_test,
                       split_percentile=split_percentile,
                       output_directory=output_directory,
                       gpu=gpu,
                       imagepattern=imagepattern,
                       M=M,
                       epochs=epochs)

    elif action == 'testing':
        utils.test_nn(image_dir_test=trained_imageDir,
                      label_dir_test=trained_labelDir,
                      model_basedir=modeldir,
                      output_directory=output_directory,
                      gpu=gpu,
                      imagepattern=imagepattern)

    else:
        raise ValueError("Action Variable is Incorrect")


if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--action', dest='action', type=str,
                        help='Either loading, creating, or continuing to train a neural network', required=True)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)

    # all the unique parameters if action is train
    parser.add_argument('--inpImageDirTrain', dest='input_directory_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=False)
    parser.add_argument('--inpLabelDirTrain', dest='input_directory_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=False)
    parser.add_argument('--inpImageDirTest', dest='input_directory_images_test', type=str,
                        help='Path to folder with intesity based images for testing', required=False)
    parser.add_argument('--inpLabelDirTest', dest='input_directory_labels_test', type=str,
                        help='Path to folder with labelled segments, ground truth for testing', required=False)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=False)
    parser.add_argument('--controlPoints', dest='controlPoints', type=int,
                        help='Define the number of control points', required=False)
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='The number of epochs to run', required=False)

    # all the unique parameters if action is test
    parser.add_argument('--TrainedImageDir', dest='trained_ImageDir', type=str,
                        help='Path to folder with intesity based images for testing the trained neural network', required=False)
    parser.add_argument('--TrainedLabelDir', dest='trained_LabelDir', type=str,
                        help='Path to folder with labelled segments, ground truth for testing the trained neural network', required=False)
    parser.add_argument('--modelDir', dest='model_directory', type=str,
                        help='Path to saved model', required=False)


    # Parse the arguments
    args = parser.parse_args()
    
    #required by both
    action = args.action
    output_directory = args.output_directory
    imagepattern = args.image_pattern

    # for training neural network
    image_dir_train = args.input_directory_images_train
    label_dir_train = args.input_directory_labels_train
    image_dir_test = args.input_directory_images_test
    label_dir_test = args.input_directory_labels_test
    split_percentile = args.split_percentile
    M = args.controlPoints
    epochs = args.epochs

    # exclusively define:
    # split_percentile OR (image_dir_test and label_dir_test)
    if split_percentile == None:
        assert image_dir_test != None
        assert label_dir_test != None
    
    if split_percentile != None:
        assert image_dir_test == None
        assert label_dir_test == None

    # for testing trained neural network
    trained_imageDir = args.trained_ImageDir
    trained_labelDir = args.trained_LabelDir
    modeldir = args.model_directory
    
    gpu = False
    local_device_protos = device_lib.list_local_devices()
    gpulist = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpulist) > 0:
        gpu = True

    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Action: {} a neural network".format(action))

    if action == 'training':
        if split_percentile == None:
            logger.info("Input Training Directory for Intensity Based Images: {}".format(image_dir_train))
            logger.info("Input Training Directory for Labelled Images: {}".format(label_dir_train))
            logger.info("Input Testing Directory for Intensity Based Images: {}".format(image_dir_test))
            logger.info("Input Testing Directory for Labelled Images: {}".format(label_dir_test))
            
        else:
            logger.info("Input Directory for Intensity Based Images: {}".format(image_dir_train))
            logger.info("Input Directory for Labelled Images: {}".format(label_dir_train))
            logger.info("Splitting Input Directory into {}:{} Ratio".format(split_percentile, 100-split_percentile))
        
        logger.info("Number of Control Points {}".format(M))
        logger.info("Number of Epochs {}".format(epochs))
    else:
        logger.info("Intensity Based Image Directory for Testing Neural Network: {}".format(trained_imageDir))
        logger.info("Labelled Data Directory for Testing Neural Network: {}".format(trained_labelDir))
        logger.info("Directory containing the saved weights: {}".format(modeldir))
    logger.info("Is there a GPU? {}".format(gpu))


    if imagepattern == None:
        imagepattern = '.*'

    main(image_dir_train=image_dir_train,
         label_dir_train=label_dir_train,
         image_dir_test=image_dir_test,
         label_dir_test=label_dir_test,
         trained_imageDir = trained_imageDir,
         trained_labelDir = trained_labelDir,
         modeldir = modeldir,
         split_percentile=split_percentile,
         action=action,
         output_directory=output_directory,
         gpu=gpu,
         M=M,
         epochs=epochs,
         imagepattern=imagepattern)


