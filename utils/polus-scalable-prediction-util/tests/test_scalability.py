
# %%
import os
import shutil
import tempfile

import itertools
from itertools import repeat

from concurrent import futures
from multiprocessing import cpu_count
from multiprocessing import Queue

import bfio
from bfio import BioReader, BioWriter

import numpy as np

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from csbdeep.utils import normalize
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

# import torch
# torch.multiprocessing.set_start_method('spawn',force=True)
# import cellpose
# import cellpose.models as cellpose_models

import scalability as scal_utils

# %% 
def prediction_cellpose(intensity_img, model):

    # x=0
    # y=0
    # z=0
    # TILE_OVERLAP = 64
    # TILE_SIZE = 1024
    # x_min = max([0, x - TILE_OVERLAP])
    # x_max = min([br.X, x + TILE_SIZE + TILE_OVERLAP])
    # y_min = max([0, y - TILE_OVERLAP])
    # y_max = min([br.Y, y + TILE_SIZE + TILE_OVERLAP])




    x_min = 0
    x_max = 1048
    y_min = 0
    y_max = 1048


    intensity_img_shape = intensity_img.shape
    intensity_img = np.reshape(intensity_img, (intensity_img_shape[0], intensity_img_shape[1], 1))
    # diameter = model.eval(intensity_img, channels=[0,0])
    # print("MODEL DIAMETER MEAN: ", model.diam_mean)

    # rescale = model.diam_mean / np.asarray(diameter)

    intensity_img = cellpose.transforms.convert_image(intensity_img,[0,0],False,True,False)

    model.batch_size = 8
    # output is only one variable
    tiled_prediction = model._run_cp(intensity_img[np.newaxis,...], rescale=30, resample=True, compute_masks=False)
    # print(prob.shape)
    # prob = prob[...,np.newaxis,np.newaxis,np.newaxis]
    # dP = dP[...,np.newaxis,np.newaxis]


    # prob_array = prob[y_overlap:y_max - y_min + y_overlap,
    #                     x_overlap:x_max - x_min + x_overlap,
    #                     ...].transpose(4,3,2,0,1)
    # dp_array = dP[:,
    #             y_overlap:y_max - y_min + y_overlap,
    #             x_overlap:x_max - x_min + x_overlap,
    #             ...].transpose(4,0,3,1,2)

    # tiled_pred = np.stack((prob_array, dp_array), axis=1)
    return tiled_prediction

def prediction_splinedist(intensity_img, model, pmin_val, pmax_val):

    input_intensity_shape = intensity_img.shape

    tiled_prediction = normalize(intensity_img, pmin=pmin_val, pmax=pmax_val, axis=(0,1),dtype=int)
    tiled_prediction, _ = model.predict_instances(tiled_prediction)
    tiled_prediction = np.reshape(tiled_prediction, (input_intensity_shape[0], 
                                                     input_intensity_shape[1],
                                                     1,
                                                     1,
                                                     1))
    tiled_prediction = tiled_prediction.astype(np.float64)

    return tiled_prediction

def main():

    model_path = '/home/ec2-user/workdir/splinedist_scalability/models'
    image_path = '/home/ec2-user/workdir/data/scalability/r01c01f_001-121_p_01-60_-ch1sk1fk1fl1.ome.tif'


    window_size = (1048, 1048, 1, 1, 1)
    step_size = (1024, 1024, 1, 1, 1)
    tile_len = 1024

    # Not using GPU helps save memory
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Get image location parameters
    image_directory = os.path.dirname(image_path)
    base_image = os.path.basename(image_path)
    # output image is a zarr image
    zarr_image = os.path.splitext(base_image)[0] + ".zarr"

    with bfio.BioReader(image_path, max_workers=2) as br_image:

        # Get the image_shape of the image that we are predicting from
        image_shape = br_image.shape

        amount_to_pad = lambda x : int(min(abs(x - np.floor(x/1024)*1024), abs(x - np.ceil(x/1024)*1024))) 
        biowriter_padding = [amount_to_pad(shape) if shape != 1 else 0 for shape in image_shape ]
        
        with bfio.BioWriter(zarr_image,
                        Y = image_shape[0] + biowriter_padding[0],
                        X = image_shape[1] + biowriter_padding[1],
                        Z = image_shape[2] + biowriter_padding[2],
                        C = image_shape[3] + biowriter_padding[3],
                        T = image_shape[4] + biowriter_padding[4],
                        dtype=np.float64) as bw_pred:
            
            pmin = 1
            pmax = 99.8
            splinedist_model = SplineDist2D(None, name='/home/ec2-user/workdir/splinedist_scalability/models')
            splinedist_prediction_lambda = lambda input_intensity_image: prediction_splinedist(intensity_img=input_intensity_image, model=splinedist_model, pmin_val=pmin, pmax_val=pmax)

            scal_utils.scalable_prediction(bioreader_obj=br_image,
                                            biowriter_obj=bw_pred,
                                            biowriter_obj_location = zarr_image,
                                            overlap_size =(24,24,0,0,0),
                                            prediction_fxn=splinedist_prediction_lambda)
    
            # USE_GPU = torch.cuda.is_available()
            # devs = []
            # for dev in range(torch.cuda.device_count()):
            #     replicates = min([int(torch.cuda.get_device_properties(0).total_memory/(3.6*10**9)),2])
            #     print("Minimum number of Replicates: ", replicates)
            #     for _ in range(replicates):
            #         print(dev)
            #         devs.append(torch.device(f"cuda:{dev}"))
            
            # DEV = Queue(len(devs))
            # for d in devs:
            #     DEV.put(d)

            # print("T/F GPU: ", USE_GPU)
            # d = DEV.get()
            # cuda_device = torch.device("cuda:0")
            # cellpose_model = cellpose_models.CellposeModel(model_type='ctyo',
            #                                       gpu=USE_GPU,
            #                                       device=torch.device("cuda:0"))
            
            # cellpose_prediction_lambda = lambda input_intensity_image: prediction_cellpose(intensity_img=input_intensity_image,model = cellpose_model)
            # scal_utils.scalable_prediction(bioreader_obj=br_image,
            #                                biowriter_obj=bw_pred,
            #                                biowriter_obj_location=zarr_image,
            #                                overlap_size=(24,24,0,0,0),
            #                                prediction_fxn=cellpose_prediction_lambda)

    import matplotlib.pyplot as plt
    with bfio.BioReader(zarr_image) as br_check:
        br_check = br_check[:]
        plt.imshow(br_check)
        plt.savefig("br_check_cellpose.png")

main()


