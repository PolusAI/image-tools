import logging, argparse
import os 

import numpy as np
import collections

import bfio
from bfio import BioReader

from csbdeep.utils import normalize

from splinedist import fill_label_holes
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max

import cv2

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def random_fliprot(img, mask): 
    img = np.array(img)
    mask = np.array(mask)
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax) # reverses the order of elements
            mask = np.flip(mask, axis=ax) # reverses the order of elements
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def train(image_dir,
         label_dir,
         output_dir,
         split_percentile,
         gpu,
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
    n_val = np.ceil((split_percentile/100) * num_images).astype('int')
    ind_train, ind_val = index[:-n_val], index[-n_val:]
    X_val, Y_val = [images[i] for i in ind_val]  , [labels[i] for i in ind_val] # splitting data into train and testing
    X_trn, Y_trn = [images[i] for i in ind_train], [labels[i] for i in ind_train] 
    num_trained = len(ind_train)
    num_tested = len(ind_val)

    logger.info("{}/{} ({}%) for training".format(num_trained, num_images, 100-split_percentile))
    logger.info("{}/{} ({}%) for testing".format(num_tested, num_images, split_percentile))

    assert collections.Counter(X_val) == collections.Counter(Y_val), "Image Test Data does not match Label Test Data for neural network"
    assert collections.Counter(X_trn) == collections.Counter(Y_trn), "Image Train Data does not match Label Train Data for neural network"

    array_images_trained = []
    array_labels_trained = []

    array_images_tested = []
    array_labels_tested = []

    axis_norm = (0,1)
    n_channel = None

    for im in range(num_trained):
        image = os.path.join(image_dir, X_trn[im])
        br_image = BioReader(image, max_workers=1)
        if im == 0:
            n_channel = br_image.shape[2]
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    for im in range(num_tested):
        image = os.path.join(image_dir, X_val[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    contoursize_max = 0
    logger.info("\n Getting Max Contoursize  ...")

    for lab in range(num_trained):
        label = os.path.join(label_dir, Y_trn[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_trained.append(fill_label_holes(lab_array))

        obj_list = np.unique(lab_array)
        obj_list = obj_list[1:]

        for j in range(len(obj_list)):
            mask_temp = lab_array.copy()     
            mask_temp[mask_temp != obj_list[j]] = 0
            mask_temp[mask_temp > 0] = 1

            mask_temp = mask_temp.astype(np.uint8)    
            contours,_ = cv2.findContours(mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = [cv2.contourArea(cnt) for cnt in contours]    
            max_ind = np.argmax(areas)
            contour = np.squeeze(contours[max_ind])
            contour = np.reshape(contour,(-1,2))
            contour = np.append(contour,contour[0].reshape((-1,2)),axis=0)
            contoursize_max = max(int(contour.shape[0]), contoursize_max)
    
    for lab in range(num_tested):
        label = os.path.join(label_dir, Y_val[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))


    logger.info("Max Contoursize: {}".format(contoursize_max))

    M = 8 # control points
    n_params = 2 * M

    grid = (2,2)
    
    conf = Config2D (
    n_params        = n_params,
    grid            = grid,
    n_channel_in    = n_channel,
    contoursize_max = contoursize_max,
    )
    conf.use_gpu = gpu
    
    logger.info("\n Generating phi and grids ... ")
    phi_generator(M, conf.contoursize_max, '.')
    grid_generator(M, conf.train_patch_size, conf.grid, '.')

    model = SplineDist2D(conf, name='models', basedir=output_dir)
    model.train(array_images_trained,array_labels_trained, validation_data=(array_images_tested, array_labels_tested), augmenter=augmenter, epochs = 300)

    logger.info("\n Done Training.")

if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDir', dest='input_directory_images', type=str,
                        help='Path to folder with intesity based images', required=True)
    parser.add_argument('--inpLabelDir', dest='input_directory_labels', type=str,
                        help='Path to folder with labelled segments, ground truth', required=True)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=True)
    parser.add_argument('--gpuAvailability', dest='GPU', type=bool,
                        help='Is there a GPU to use?', required=False, default=False)
    parser.add_argument('--outDir', dest='output_directory', type=str,
                        help='Path to output directory containing the neural network weights', required=True)
    parser.add_argument('--imagePattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input_images and input_labels', required=False)

    # Parse the arguments
    args = parser.parse_args()
    image_dir = args.input_directory_images
    label_dir = args.input_directory_labels
    split_percentile = args.split_percentile
    gpu = args.GPU
    output_directory = args.output_directory
    imagepattern = args.image_pattern
    
    logger.info("Input Directory for Intensity Based Images: {}".format(image_dir))
    logger.info("Input Directory for Labelled Images: {}".format(label_dir))
    logger.info("Output Directory: {}".format(output_directory))
    logger.info("Image Pattern: {}".format(imagepattern))
    logger.info("GPU: {}".format(gpu))
    
    train(image_dir,
         label_dir,
         output_directory,
         split_percentile,
         gpu,
         imagepattern)