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
from splinedist import random_label_cmap

import cv2

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
lbl_cmap = random_label_cmap()
# matplotlib inline
# config InlineBackend.figure_format = 'retina'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def get_jaccard_index(prediction, ground_truth):
    imageshape = prediction.shape
    prediction[prediction > 0] = 1
    ground_truth[ground_truth > 0] = 1

    totalsum = np.sum(prediction == ground_truth)
    jaccard = totalsum/(imageshape[0]*imageshape[1])

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

def predict(image_dir,
            output_dir,
            gpu,
            imagepattern):

    images = sorted(os.listdir(image_dir))
    num_images = len(images)

    assert num_images > 0, "Directory is Empty"

    axis_norm = (0,1)

    model_dir = 'models'
    if os.path.exists(os.path.join(output_dir, model_dir)):
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        
        logger.info("\n Done Loading Model ...")
    else:
        raise ValueError("No Neural Network Found")


    array_image_predicted = []
    for im in range(num_images):
        image = os.path.join(image_dir, images[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(im_array.shape[:2])
        norm_image = normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm)
        prediction, details = model.predict_instances(norm_image)

        fig, (a_image,a_prediction) = plt.subplots(1, 2, 
                                                    figsize=(12,5), 
                                                    gridspec_kw=dict(width_ratios=(1,1)))
        plt_image = a_image.imshow(im_array)
        a_image.set_title("Image")

        plt_prediction = a_prediction.imshow(prediction)
        a_prediction.set_title("Prediction")

        plot_file = "{}_{}.jpg".format(images[im], im)
        plt.savefig(plot_file)

        logger.info("Save Figure {}".format(plot_file))


    

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

    for im in range(num_tested):
        image = os.path.join(image_dir, X_val[im])
        br_image = BioReader(image, max_workers=1)
        im_array = br_image[:,:,0:1,0:1,0:1]
        im_array = im_array.reshape(br_image.shape[:2])
        array_images_tested.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))

    for lab in range(num_tested):
        label = os.path.join(label_dir, Y_val[lab])
        br_label = BioReader(label, max_workers=1)
        lab_array = br_label[:,:,0:1,0:1,0:1]
        lab_array = lab_array.reshape(br_label.shape[:2])
        array_labels_tested.append(fill_label_holes(lab_array))

    model_dir = 'models'
    if os.path.exists(os.path.join(output_dir, model_dir)):
        model = SplineDist2D(None, name=model_dir, basedir=output_dir)
        
        logger.info("\n Done Loading Model ...")
    
    else:
        for im in range(num_trained):
            image = os.path.join(image_dir, X_trn[im])
            br_image = BioReader(image, max_workers=1)
            if im == 0:
                n_channel = br_image.shape[2]
            im_array = br_image[:,:,0:1,0:1,0:1]
            im_array = im_array.reshape(br_image.shape[:2])
            array_images_trained.append(normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm))


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

        model = SplineDist2D(conf, name=model_dir, basedir=output_dir)
        model.train(array_images_trained,array_labels_trained, validation_data=(array_images_tested, array_labels_tested), augmenter=augmenter, epochs = 300)

        logger.info("\n Done Training Model ...")

    logger.info("\n Getting {} Jaccard Indexes ...".format(num_tested))

    for i in range(num_tested):
        image = array_images_tested[i]
        ground_truth = array_labels_tested[i]
        prediction, details = model.predict_instances(ground_truth)

        fig, (a_image,a_groundtruth,a_prediction) = plt.subplots(1, 3, 
                                                                 figsize=(12,5), 
                                                                 gridspec_kw=dict(width_ratios=(1,1,1)))
        plt_image = a_image.imshow(image)
        a_image.set_title("Image")

        plt_groundtruth = a_groundtruth.imshow(ground_truth)
        a_groundtruth.set_title("Ground Truth")

        plt_prediction = a_prediction.imshow(prediction)
        a_prediction.set_title("Prediction")

        jaccard = get_jaccard_index(prediction, ground_truth)

        plot_file = "{}.jpg".format(i)
        fig.text(0.50, 0.02, 'Jaccard Index = {}'.format(jaccard), 
            horizontalalignment='center', wrap=True)
        plt.savefig(os.path.join(output_dir, plot_file))

        logger.info("{} has a jaccard index of {}".format(plot_file, jaccard))




if __name__ == "__main__":
    
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Training SplineDist')

    parser.add_argument('--inpImageDir', dest='input_directory_images', type=str,
                        help='Path to folder with intesity based images', required=True)
    parser.add_argument('--inpLabelDir', dest='input_directory_labels', type=str,
                        help='Path to folder with labelled segments, ground truth', required=False)
    parser.add_argument('--splitPercentile', dest='split_percentile', type=int,
                        help='Percentage of data that is allocated for testing', required=False)
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
    
    if split_percentile != None:
        train(image_dir,
            label_dir,
            output_directory,
            split_percentile,
            gpu,
            imagepattern)

    else:
        predict(image_dir,
                output_directory,
                gpu,
                imagepattern)