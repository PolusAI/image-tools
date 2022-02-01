import os
import errno
import numpy as np

import deepcell
from deepcell.utils.data_utils import reshape_matrix
from deepcell.model_zoo.panopticnet import PanopticNet

from deepcell.utils.train_utils import get_callbacks
from deepcell.utils.train_utils import count_gpus

from tifffile import TiffFile, imread, imwrite
from bfio import BioReader, BioWriter, LOG4J, JARS
from pathlib import Path
import filepattern

from timeit import default_timer
from deepcell.applications import Mesmer, NuclearSegmentation, CytoplasmSegmentation

import tensorflow as tf

from deepcell_toolbox.deep_watershed import deep_watershed
import logging
import math
import cv2


logger = logging.getLogger("segmenting")
logger.setLevel(logging.INFO)

tile_overlap = 64
tile_size = 2048

def padding(image, shape_1, shape_2, second, size):
    
    ''' The unet expects the height and width of the image to be 256 x 256
        This function adds the required reflective padding to make the image 
        dimensions a multiple of 256 x 256. This will enable us to extract tiles
        of size 256 x 256 which can be processed by the network'''
        
    row,col=image.shape
    
    # Determine the desired height and width after padding the input image
    if second:
       m,n =math.ceil(row/size),math.ceil(col/size)
    else:
       m,n =math.ceil(shape_1/size),math.ceil(shape_2/size)

    required_rows=m*size
    required_cols=n*size
    if required_rows != required_cols:
       required_rows = max(required_rows ,  required_cols)
       required_cols = required_rows
     
    # Check whether the image dimensions are even or odd. If the image dimesions
    # are even, then the same amount of padding can be applied to the (top,bottom)
    # or (left,right)  of the image.  
    
    if row%2==0:   
        
        # no. of rows to be added to the top and bottom of the image        
        top = int((required_rows-row)/2) 
        bottom = top
    else:          
        top = int((required_rows-row)/2) 
        bottom = top+1 
          
    if col%2==0:  
        
        # no. of columns to be added to left and right of the image
        left = int((required_cols-col)/2) 
        right = left
    else: 
        left = int((required_cols-col)/2) 
        right = left+1
        
    pad_dimensions=(top,bottom,left,right)
    
    final_image=np.zeros((required_rows,required_cols))
    
    # Add relective Padding    
    final_image=cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)  
        
    # return padded image and pad dimensions
    return final_image,pad_dimensions


def get_data(rootdir, filePattern1, filePattern2, size, model):
    data = []
    inpDir = Path(rootdir)
    fp = filepattern.FilePattern(inpDir,filePattern1)
    for fP in fp():
        for PATH in fP:
           with BioReader(PATH.get("file")) as br:
              shape_1 = 0; shape_2 = 0
              for z in range(br.Z):
                  for y in range(0,br.Y,tile_size):
                      for x in range(0,br.X,tile_size):
                          x_min = max(0, x - tile_overlap)
                          x_max = min(br.X, x + tile_size + tile_overlap)
                          y_min = max(0, y - tile_overlap)
                          y_max = min(br.Y, y + tile_size + tile_overlap)
                          tile = np.squeeze(br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])
                          if tile.shape[0] < shape_1 or tile.shape[1] < shape_2:
                                shape_1 = max(tile.shape[0], shape_1)
                                shape_2 = max(tile.shape[1], shape_2)
                                second = False
                          else:
                                second = True
                                shape_1, shape_2 = tile.shape[0], tile.shape[1]
                          padded_img,pad_dimensions=padding(tile, shape_1, shape_2, second, size)

                          if model == "mesmerNuclear":
                             if filePattern2 is not None:
                                 string = PATH.get("file").name
                                 if "*" in filePattern1:
                                    filePattern1 = filePattern1.split("*")[1]
                                 name = string.replace(filePattern1, filePattern2)
                                 with BioReader(Path(str(rootdir)+"/"+name)) as br_whole:
                                     tile_whole = np.squeeze(br_whole[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])
                                     padded_img_cyto,pad_dimensions_cyto=padding(tile_whole, shape_1, shape_2, second, size)
                                     image = np.stack((padded_img, padded_img_cyto), axis=-1)
                             else:
                                 im1 = np.zeros((padded_img.shape[0], padded_img.shape[1]))
                                 image = np.stack((padded_img, im1), axis=-1)
                          elif model == "mesmerWholeCell":
                             string = PATH.get("file").name
                             if "*" in filePattern1:
                                filePattern1 = filePattern1.split("*")[1]
                             name = string.replace(filePattern1, filePattern2)
                             with BioReader(Path(str(rootdir)+"/"+name)) as br_whole:
                                 tile_whole = np.squeeze(br_whole[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])
                                 padded_img_nuclear,pad_dimensions_nuclear=padding(tile_whole, shape_1, shape_2, second, size)
                                 image = np.stack((padded_img_nuclear, padded_img), axis=-1)
                          else:
                             image = np.expand_dims(padded_img, axis=-1)
                          data.append(image)
    return data

def save_data(rootdir, y_pred, size, filePattern, out_path, model):
    inpDir = Path(rootdir)
    out_path = Path(out_path) 
    ind = 0
    fp = filepattern.FilePattern(inpDir,filePattern)
    for fP in fp():
        for PATH in fP:
            with BioReader(PATH.get("file")) as br:
                shape_1 = 0; shape_2 = 0
                with BioWriter(out_path.joinpath(PATH.get("file").name),metadata = br.metadata) as bw:
                    logger.info('Saving image {}'.format(PATH.get("file")))
                    bw.dtype = np.uint16
                    for z in range(br.Z):
                        for y in range(0,br.Y,tile_size):
                            for x in range(0,br.X, tile_size):
                                x_min = max(0, x - tile_overlap)
                                x_max = min(br.X, x + tile_size + tile_overlap)
                                y_min = max(0, y - tile_overlap)
                                y_max = min(br.Y, y + tile_size + tile_overlap)

                                tile = np.squeeze(br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])
                                if tile.shape[0] < shape_1 or tile.shape[1] < shape_2:
                                   shape_1 = max(tile.shape[0], shape_1)
                                   shape_2 = max(tile.shape[1], shape_2)
                                   second = False
                                else:
                                   second = True
                                   shape_1, shape_2 = tile.shape[0], tile.shape[1]

                                padded_img,pad_dimensions=padding(tile, shape_1, shape_2, second, size)

                                out_img=np.zeros((padded_img.shape[0],padded_img.shape[1]))

                                if model == "BYOM":
                                    for i in range(int(padded_img.shape[0]/size)):
                                        for j in range(int(padded_img.shape[1]/size)):
                                            new_img = np.squeeze(y_pred[ind])
                                            out_img[i*size:(i+1)*size,j*size:(j+1)*size]=new_img
                                            ind+=1
                                else:
                                    out_img = np.squeeze(y_pred[ind])
                                    ind+=1

                                top_pad,bottom_pad,left_pad,right_pad=pad_dimensions
                                output = out_img[top_pad:out_img.shape[0]-bottom_pad,left_pad:out_img.shape[1]-right_pad]
                                output = output.astype(np.uint16)

                                x_overlap, x_min, x_max = x - x_min, x, min(br.X, x + tile_size)
                                y_overlap, y_min, y_max = y - y_min, y, min(br.Y, y + tile_size)

                                final = output[y_overlap:y_max - y_min + y_overlap, x_overlap:x_max - x_min+ x_overlap]
                                output_image_5channel=np.zeros((final.shape[0], final.shape[1],1,1,1),dtype=np.uint16)

                                output_image_5channel[:,:,0,0,0]=final

                                bw[y_min:y_max, x_min:x_max,0:1,0,0] = output_image_5channel


def predict_(xtest_path, ytest_path, size, model_path, filePattern1, filePattern2, model, out_path):

    print("entered predict")
    size = int(size)
    model_path = Path(model_path)
    rootdir = Path(xtest_path)
    x_test = get_data(rootdir, filePattern1, filePattern2, size, model)
    X_test = np.asarray(x_test)

    rootdir = Path(ytest_path)
    y_test = get_data(rootdir, filePattern1, filePattern2, size, model)
    y_test = np.asarray(y_test)

    X_test, y_test = reshape_matrix(X_test, y_test, reshape_size=size)

    classes = {
        'inner_distance': 1,  # inner distance
        'outer_distance': 1,  # outer distance
    }

    prediction_model = PanopticNet(
        backbone='resnet50',
        input_shape=X_test.shape[1:],
        norm_method='std',
        num_semantic_heads=2,
        num_semantic_classes=classes,
        location=True,  # should always be true
        include_top=True)

    model_name = 'watershed_centroid_nuclear_general_std.h5'
    model_path = model_path.joinpath(model_name)
    prediction_model.load_weights(model_path, by_name=True)

    start = default_timer()
    outputs = prediction_model.predict(X_test)
    watershed_time = default_timer() - start

    logger.info('Watershed segmentation of shape {} in {} seconds.'.format(outputs[0].shape, watershed_time))

    y_pred = []

    masks = deep_watershed(
       outputs,
       min_distance=10,
       detection_threshold=0.1,
       distance_threshold=0.01,
       exclude_border=False,
       small_objects_threshold=0)

    for i in range(masks.shape[0]):
       y_pred.append(masks[i,...])

    save_data(Path(xtest_path), y_pred, size, filePattern1, out_path, model)
    logger.info("Segmentation complete.")


def run(xtest_path, ytest_path, size, model_path, filePattern1, filePattern2, model, out_path):
#    print(model_path)

    inpDir = Path(xtest_path)
    out_path = Path(out_path)
    size = int(size)
    data = []
    if model in ["mesmerNuclear", "nuclear", "cytoplasm","mesmerWholeCell"]:
       x_test = get_data(inpDir, filePattern1, filePattern2, size, model)
       X_test = np.asarray(x_test)
       if model == "mesmerNuclear":
          MODEL_DIR = os.path.expanduser(os.path.join('~', '.keras', 'models'))
          modelPath = os.path.join(MODEL_DIR, "MultiplexSegmentation")
          modelM = tf.keras.models.load_model(modelPath)
          app = Mesmer(model=modelM)
          output = app.predict(X_test, compartment="nuclear")
       elif model == "mesmerWholeCell":
          MODEL_DIR = os.path.expanduser(os.path.join('~', '.keras', 'models'))
          modelPath = os.path.join(MODEL_DIR, "MultiplexSegmentation")
          modelM = tf.keras.models.load_model(modelPath)
          app = Mesmer(model=modelM)
          output = app.predict(X_test, compartment="whole-cell")
       elif model == "nuclear":
          app = NuclearSegmentation()
          output = app.predict(X_test)
       elif model == "cytoplasm":
          app = CytoplasmSegmentation() 
          output = app.predict(X_test)

       save_data(inpDir, output, size, filePattern1, out_path, model)
       logger.info("Segmentation complete.")

    elif model == "BYOM":
       predict_(xtest_path, ytest_path, size, model_path, filePattern1, filePattern2, model, out_path)
