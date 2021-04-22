import logging
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
from splinedist import random_label_cmap

import keras.backend as K
import tensorflow as tf
from tensorflow import keras

import sklearn.metrics
from sklearn.metrics import jaccard_score

import cv2

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
lbl_cmap = random_label_cmap()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)


def get_jaccard_index(prediction : np.ndarray,
                      ground_truth : np.ndarray):
    """ This function gets the jaccard index between the 
    predicted image and its ground truth.

    Args:
        prediction - the predicted output from trained neural network
        ground_truth - ground truth given by inputs
    Returns:
        jaccard - The jaccard index between the two inputs
                  https://en.wikipedia.org/wiki/Jaccard_index
    Raises:
        None
    """
    imageshape = prediction.shape
    prediction = prediction.ravel()
    ground_truth = ground_truth.ravel()
    prediction[prediction > 0] = 1.0
    ground_truth[ground_truth > 0] = 1.0

    jaccard = jaccard_score(prediction, ground_truth)

    return jaccard

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

def augmenter(x : np.ndarray, 
              y : np.ndarray):
    """Augmentation of a single input/label image pair.
        Taken from SplineDist's training notebook.

    Args:
        x - is an input image
        y - is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y


def create_test_plots(array_images : list, 
                 array_labels : list, 
                 input_len : int, 
                 output_dir : list, 
                 model):
    
    """ This function generates subplots of the 
    original image, the ground truth and prediction with 
    the jaccard index specified at the bottom of the image.

    Args:
        array_images - a list of numpy arrays of the intensity based images
        array_labels - a list of numpy arrays of the images's corresponding 
                       ground truth
        input_len - the number of images in array_images and array_label
        output_dir - the location where the jpeg files are saved
        model - the neural network used to make the prediction
    Returns:
        None, saves images in output directory
    """
    jaccard_indexes = []
    
    for i in range(input_len):
        fig, (a_image,a_groundtruth,a_prediction) = plt.subplots(1, 3, 
                                                            figsize=(12,5), 
                                                            gridspec_kw=dict(width_ratios=(1,1,1)))

        image = array_images[i]
        ground_truth = array_labels[i]
        prediction, details = model.predict_instances(image)
        
        plt_image = a_image.imshow(image)
        a_image.set_title("Image")

        plt_groundtruth = a_groundtruth.imshow(ground_truth)
        a_groundtruth.set_title("Ground Truth")

        plt_prediction = a_prediction.imshow(prediction)
        a_prediction.set_title("Prediction")

        jaccard = get_jaccard_index(prediction, ground_truth)
        jaccard_indexes.append(jaccard)
        plot_file = "{}.jpg".format(i)
        fig.text(0.50, 0.02, 'Jaccard Index = {}'.format(jaccard), 
            horizontalalignment='center', wrap=True)
        plt.savefig(os.path.join(output_dir, plot_file))
        plt.clf()
        plt.cla()
        plt.close(fig)
 
        logger.info("{} has a jaccard index of {}".format(plot_file, jaccard))

    average_jaccard = sum(jaccard_indexes)/input_len
    logger.info("Average Jaccard Index for Testing Data: {}".format(average_jaccard))

def train_nn(image_dir_input : str,
             label_dir_input : str,
             image_dir_test : str,
             label_dir_test : str,
             split_percentile : int,
             output_directory : str,
             gpu : bool,
             imagepattern : str,
             M : int,
             epochs : int,
             learning_rate : float):

    """ This function either trains or continues to train a neural network 
    for SplineDist.
    Either testing directories are specified or the the input directory gets 
    split into training and testing data. 

    Args:
        image_dir_input: location for Intensity Based Images
        label_dir_input: location for Ground Truths of the images
        image_dir_test: Specifies the location for Intensity Based Images for testing
        label_dir_test: Specifies the location for Ground truth images for testing
        split_percentile: Specifies what percentages of the input should be allocated for tested
        output_directory: Specifies the location for the output generated
        gpu: Specifies whether or not to use a GPU
        imagepattern: The imagepattern of files to iterate through within a directory
        M: Specifies the number of control points
        epochs : Specifies the number of epochs to be run

    Returns:
        None, a trained neural network whose performance is calculated by the jaccard index
    
    Raises:
        AssertionError: If there is less than one training image
        AssertionError: If the number of images to do not match the number of ground truths 
                        available.
    """

    assert isinstance(M, int), "Neeed to specify the number of control points"
    assert isinstance(epochs, int), "Need to specify the number of epochs to run"

    # get the inputs
    input_images = sorted(os.listdir(image_dir_input))
    input_labels = sorted(os.listdir(label_dir_input))
    num_inputs = len(input_images)
    
    logger.info("\n Getting Data for Training and Testing  ...")
    if split_percentile == None:
        # if images are already allocated for testing then use those
        logger.info("Getting From Testing Directories")
        X_trn = input_images
        Y_trn = input_labels
        X_val = sorted(os.listdir(image_dir_test))
        Y_val = sorted(os.listdir(label_dir_test))
    else:
        logger.info("Splitting Input Directories")
        # Used when no directory has been specified for testing
        # Splits the input directories into testing and training
        rng = np.random.RandomState(42)
        index = rng.permutation(num_inputs)
        n_val = np.ceil((split_percentile/100) * num_inputs).astype('int')
        ind_train, ind_val = index[:-n_val], index[-n_val:]
        X_val, Y_val = [input_images[i] for i in ind_val]  , [input_labels[i] for i in ind_val] # splitting data into train and testing
        X_trn, Y_trn = [input_images[i] for i in ind_train], [input_labels[i] for i in ind_train] 
        image_dir_test = image_dir_input
        label_dir_test = label_dir_input
    
    # Lengths
    num_images_trained = len(X_trn)
    num_labels_trained = len(Y_trn)
    num_images_tested = len(X_val)
    num_labels_tested = len(Y_val)

    # Renamed inputs
    del input_images
    del input_labels
    del num_inputs

    # assertions
    assert num_images_trained > 1, "Not Enough Training Data"
    assert num_images_trained == num_labels_trained, "The number of images does not match the number of ground truths for training"
    assert num_images_tested == num_images_tested, "The number of images does not match the number of ground for testing"

    # Make sure every image has a corresponding ground truth thats used for training and testing
    assert collections.Counter(X_val) == collections.Counter(Y_val), "Image Test Data does not match Label Test Data for neural network"
    assert collections.Counter(X_trn) == collections.Counter(Y_trn), "Image Train Data does not match Label Train Data for neural network"

    # Get logs for end user
    totalimages = num_images_trained+num_images_tested
    logger.info("{}/{} inputs used for training".format(num_images_trained, totalimages))
    logger.info("{}/{} inputs used for testing".format(num_images_trained, totalimages))

    # Need a list of numpy arrays to feed to SplineDist
    array_images_trained = []
    array_labels_trained = []
    array_images_tested = []
    array_labels_tested = []

    # Neural network parameters
    axis_norm = (0,1)
    n_channel = 1 # this is based on the input data

    # Read the input images used for testing
    for im in range(num_images_tested):
        image = os.path.join(image_dir_test, X_val[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    # Read the input labels used for testing 
    for lab in range(num_labels_tested):
        label = os.path.join(label_dir_test, Y_val[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))
    
    # Read the input images used for training
    for im in range(num_images_trained):
        image = os.path.join(image_dir_input, X_trn[im])
        br_image = BioReader(image, max_workers=1)
        if im == 0:
            n_channel = br_image.shape[2]
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))
        

    model_dir_name = 'models'
    model_dir_path = os.path.join(output_directory, model_dir_name)
    if os.path.exists(os.path.join(output_directory, model_dir_name)):
        # if model exists, then we need to continue training on it
        model = SplineDist2D(None, name=model_dir_name, basedir=output_directory)
        logger.info("\n Done Loading Model ...")
        model.optimize_thresholds
        logger.info("Optimized thresholds")

        logger.info("\n Getting extra files ...")
        if not os.path.exists("./phi_{}.npy".format(M)):
            contoursize_max = model.config.contoursize_max
            logger.info("Contoursize Max for phi_{}.npy: {}".format(M, contoursize_max))
            phi_generator(M, contoursize_max, '.')
        logger.info("Generated phi")
        if not os.path.exists("./grid_{}.npy".format(M)):
            training_patch_size = model.config.train_patch_size
            logger.info("Training Patch Size {} for grid_{}.npy: {}".format(M, training_patch_size))
            grid_generator(M, training_patch_size, conf.grid, '.')
        logger.info("Generated grid")

        for lab in range(num_labels_trained):
            label = os.path.join(label_dir_input, Y_trn[lab])
            br_label = BioReader(label, max_workers=1)
            lab_array = br_label[:,:,0:1,0:1,0:1]
            lab_array = lab_array.reshape(br_label.shape[:2])
            array_labels_trained.append(fill_label_holes(lab_array))

    else:
        # otherwise we need to build a new model and get the appropriate
            # parameters for it
        contoursize_max = 0
        logger.info("\n Getting Max Contoursize  ...")

        for lab in range(num_labels_trained):
            label = os.path.join(label_dir_input, Y_trn[lab])
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

        logger.info("Max Contoursize: {}".format(contoursize_max))

        n_params = M*2
        grid = (2,2)
        conf = Config2D (
        n_params            = n_params,
        grid                = grid,
        n_channel_in        = n_channel,
        contoursize_max     = contoursize_max,
        train_learning_rate = learning_rate,
        train_epochs        = epochs,
        use_gpu             = gpu
        )
        

        logger.info("\n Generating phi and grids ... ")
        if not os.path.exists("./phi_{}.npy".format(M)):
            phi_generator(M, conf.contoursize_max, '.')
        logger.info("Generated phi")
        if not os.path.exists("./grid_{}.npy".format(M)):
            grid_generator(M, conf.train_patch_size, conf.grid, '.')
        logger.info("Generated grid")
 
        model = SplineDist2D(conf, name=model_dir_name, basedir=output_directory)

    # After creating or loading model, train it
    # model.config.train_tensorboard = False
    # model.config.use_gpu = gpu
    logger.info("\n Parameters in Config File ...")
    config_dict = model.config.__dict__
    for ky,val in config_dict.items():
        logger.info("{}: {}".format(ky, val))

    model.train(array_images_trained,array_labels_trained, 
                validation_data=(array_images_tested, array_labels_tested), 
                augmenter=augmenter, epochs=epochs)
    logger.info("\n Done Training Model ...")


def test_nn(image_dir_test : str,
            label_dir_test : str,
            output_directory : str,
            gpu : bool,
            imagepattern : str):

    model_dir_name = 'models'
    model_dir_path = os.path.join(output_directory, model_dir_name)
    assert os.path.exists(model_dir_path), \
        "{} does not exist".format(model_dir_path)

    model = SplineDist2D(None, name=model_dir_name, basedir=output_directory)
    logger.info("\n Done Loading Model ...")

    weights_best = os.path.join(model_dir_path, "weights_best.h5")
    model.keras_model.load_weights(weights_best)
    logger.info("\n Done Loading Best Weights ...")

    logger.info("\n Parameters in Config File ...")
    config_dict = model.config.__dict__
    for ky,val in config_dict.items():
        logger.info("{}: {}".format(ky, val))

    X_val = sorted(os.listdir(image_dir_test))
    Y_val = sorted(os.listdir(label_dir_test))
    num_images = len(X_val)
    num_labels = len(Y_val)

    assert num_images > 0, "Input Directory is empty"
    assert num_images == num_labels, "The number of images do not match the number of ground truths"

    array_images_tested = []
    array_labels_tested = []

    # Neural network parameters
    axis_norm = (0,1)
    n_channel = 1 # this is based on the input data

    # Read the input images used for testing
    for im in range(num_images):
        image = os.path.join(image_dir_test, X_val[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    # Read the input labels used for testing 
    for lab in range(num_labels):
        label = os.path.join(label_dir_test, Y_val[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))

    create_test_plots(array_images_tested, array_labels_tested, num_images, output_directory, model)

def predict_nn(image_dir_test : str,
               output_directory : str,
               gpu : bool,
               imagepattern : str):
    return None