import logging, argparse
import os 

import train

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
         imagepattern: str):

    if action == 'train':
        train.train_nn(image_dir_train,
                        label_dir_train,
                        image_dir_test,
                        label_dir_test,
                        action,
                        output_directory,
                        gpu,
                        imagepattern)


if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDirTrain', dest='input_directory_images_train', type=str,
                        help='Path to folder with intesity based images for training', required=True)
    parser.add_argument('--inpLabelDirTrain', dest='input_directory_labels_train', type=str,
                        help='Path to folder with labelled segments, ground truth for training', required=True)
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

    # Parse the arguments
    args = parser.parse_args()
    image_dir_train = args.input_directory_images_train
    label_dir_train = args.input_directory_labels_train
    image_dir_test = args.input_directory_images_test
    label_dir_test = args.input_directory_labels_test
    split_percentile = args.split_percentile
    gpu = args.GPU
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
    logger.info("GPU: {}".format(gpu))
    logger.info("{} a neural network".format(action))

    main(image_dir_train,
         label_dir_train,
         image_dir_test,
         label_dir_test,
         action,
         output_directory,
         gpu,
         imagepattern)


