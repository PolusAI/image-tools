import logging, argparse
import os 

import utils

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(image_dir_train : str,
         label_dir_train : str,
         image_dir_test : str,
         label_dir_test : str,
         split_percentile : int,
         action : str,
         output_directory : str,
         gpu : bool,
         M : int,
         epochs : int,
         imagepattern: str):

    if action == 'train':
        utils.train_nn(image_dir_train,
                        label_dir_train,
                        image_dir_test,
                        label_dir_test,
                        split_percentile,
                        output_directory,
                        gpu,
                        imagepattern,
                        M,
                        epochs)

    elif action == 'test':
        utils.test_nn(image_dir_test,
                      label_dir_test,
                      output_directory,
                      gpu,
                      imagepattern)
    
    elif action == 'predict':
        utils.predict_nn(image_dir_test,
                        output_directory,
                        gpu,
                        imagepattern)

    else:
        raise ValueError("Action Variable is Incorrect")


if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDirTrain', dest='input_directory_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=False)
    parser.add_argument('--inpLabelDirTrain', dest='input_directory_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=False)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=False)
    parser.add_argument('--inpImageDirTest', dest='input_directory_images_test', type=str,
                        help='Path to folder with intesity based images for testing', required=False)
    parser.add_argument('--inpLabelDirTest', dest='input_directory_labels_test', type=str,
                        help='Path to folder with labelled segments, ground truth for testing', required=False)
    parser.add_argument('--gpuAvailability', dest='GPU', type=bool,
                        help='Is there a GPU to use?', required=False, default=False)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)
    parser.add_argument('--action', dest='action', type=str,
                        help='Either loading, creating, or continuing to train a neural network', required=True)
    parser.add_argument('--controlPoints', dest='controlPoints', type=int,
                        help='Define the number of control points', required=False)
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='The number of epochs to run', required=False)

    # Parse the arguments
    args = parser.parse_args()
    image_dir_train = args.input_directory_images_train
    label_dir_train = args.input_directory_labels_train
    image_dir_test = args.input_directory_images_test
    label_dir_test = args.input_directory_labels_test
    split_percentile = args.split_percentile
    gpu = args.GPU
    M = args.controlPoints
    epochs = args.epochs
    output_directory = args.output_directory
    imagepattern = args.image_pattern
    action = args.action
    
    if split_percentile == None:
        logger.info("Input Training Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Training Directory for Labelled Images: {}".format(label_dir_train))
        logger.info("Input Testing Directory for Intensity Based Images: {}".format(image_dir_test))
        logger.info("Input Testing Directory for Labelled Images: {}".format(label_dir_test))
        
    else:
        logger.info("Input Directory for Intensity Based Images: {}".format(image_dir_train))
        logger.info("Input Directory for Labelled Images: {}".format(label_dir_train))
        logger.info("Splitting Input Directory into {}:{} Ratio".format(split_percentile, 100-split_percentile))
    
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("Number of Control Points {}".format(M))
    logger.info("Number of Epochs {}".format(epochs))
    logger.info("GPU: {}".format(gpu))
    logger.info("Action: {} a neural network".format(action))

    main(image_dir_train=image_dir_train,
         label_dir_train=label_dir_train,
         image_dir_test=image_dir_test,
         label_dir_test=label_dir_test,
         split_percentile=split_percentile,
         action=action,
         output_directory=output_directory,
         gpu=gpu,
         M=M,
         epochs=epochs,
         imagepattern=imagepattern)


