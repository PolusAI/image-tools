import logging, argparse
import os 

import numpy as np
import collections

import bfio
from bfio import BioReader

from csbdeep.utils import normalize
from splinedist import fill_label_holes



# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main(image_dir,
         label_dir,
         output_dir,
         imagepattern):
    
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))

    num_images = len(images)
    num_labels = len(labels)

    assert num_images > 1, "Not Enough Training Data"
    assert num_images == num_labels, "The number of images does not match the number of labels"
    
    logger.info("\n Spliting Data for Training and Testing  ...")
    rng = np.random.RandomState(42)
    index = rng.permutation(num_images)
    n_val = np.ceil(0.15 * num_images).astype('int')
    ind_train, ind_val = index[:-n_val], index[-n_val:]
    X_val, Y_val = [images[i] for i in ind_val]  , [labels[i] for i in ind_val] # splitting data into train and testing
    X_trn, Y_trn = [images[i] for i in ind_train], [labels[i] for i in ind_train] 
    num_trained = len(ind_train)
    num_tested = len(ind_val)

    logger.info("{}/{} for training".format(num_trained, num_images))
    logger.info("{}/{} for testing".format(num_tested, num_images))

    assert collections.Counter(X_val) == collections.Counter(Y_val), "Image Test Data does not match Label Test Data for neural network"
    assert collections.Counter(X_trn) == collections.Counter(Y_trn), "Image Train Data does not match Label Train Data for neural network"

    array_images_trained = np.zeros((512, 512, num_trained), dtype = 'uint8')
    array_labels_trained = np.zeros((512, 512, num_trained), dtype = 'uint8')

    for im in range(num_trained):

        image = os.path.join(image_dir, X_trn[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_trained[:,:,im] = normalize(im_array)

    for lab in range(num_trained):
        label = os.path.join(label_dir, Y_trn[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_trained[:,:,lab] = fill_label_holes(lab_array)


if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpDir_images', dest='input_directory_images', type=str,
                        help='Path to folder with intesity based images', required=True)
    parser.add_argument('--inpDir_labels', dest='input_directory_labels', type=str,
                        help='Path to folder with labelled segments, ground truth', required=True)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)

    # Parse the arguments
    args = parser.parse_args()
    image_dir = args.input_directory_images
    label_dir = args.input_directory_labels
    output_directory = args.output_directory
    imagepattern = args.image_pattern
    
    logger.info("Input Directory for Intensity Based Images: {}".format(image_dir))
    logger.info("Input Directory for Labelled Images: {}".format(label_dir))
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Image Pattern: {}".format(imagepattern))
    
    main(image_dir,
         label_dir,
         output_directory,
         imagepattern)