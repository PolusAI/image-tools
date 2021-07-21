
# %%
import os, sys
import shutil
import unittest

import argparse

sys.path.append("../scalable_prediction")
import scalability as scal_utils

import tempfile

import bfio
from bfio import BioReader, BioWriter

import numpy as np

from csbdeep.utils import normalize
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

from sklearn.metrics import jaccard_score
from sklearn.metrics import fbeta_score



# %% 
class TestEncodingDecoding(unittest.TestCase):

    # parser = argparse.ArgumentParser(prog='main', \
    #     description='Testing Tiled Predictions')

    # parser.add_argument('--image', dest='image_path', type=str,
    #                     help='Path to Image Tested', required=True)
    # parser.add_argument('--model', dest='model_path', type=str,
    #                     help='Path to Model Directory', required=True)
    # parser.add_argument('--phi', dest='phi_file', type=str,
    #                     help='Path to phi file for SplineDist', required=True)
    # parser.add_argument('--grid', dest='grid_file', type=str,
    #                     help='Path to grid file for SplineDist', required=True)

    # args = parser.parse_args()
    # image_path = args.image_path
    # model_path = args.model_path
    # phi_file   = args.phi_file
    # grid_file  = args.grid_file

    
    def test_splinedist(self):

        model_path = '/home/ec2-user/workdir/splinedist_scalability/models'
        image_path = '/home/ec2-user/workdir/data/scalability/cropped_r01c01f_001-121_p_01-60_-ch1sk1fk1fl1.ome.tif'
        phi_file   = '/home/ec2-user/workdir/splinedist_scalability/phi_6.npy'
        grid_file  = '/home/ec2-user/workdir/splinedist_scalability/grid_6.npy'

        def get_scores(y_true,
                       y_pred,
                       average):

            j_score = jaccard_score(y_true=y_true, 
                                    y_pred=y_pred,
                                    average=average)
            f1_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average,
                                   beta=1)
            f2_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average,
                                   beta=2)
            f3_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average,
                                   beta=3)

            return (j_score, f1_score, f2_score, f3_score)

            
        def prediction_splinedist(intensity_img, 
                                  model, 
                                  pmin_val, 
                                  pmax_val):

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

        # Get image location parameters
        image_directory = os.path.dirname(image_path)
        base_image = os.path.basename(image_path)
        # output image is a zarr image
        tiled_zarr_image_name = "tiled_" + os.path.splitext(base_image)[0] + ".zarr"
        whole_zarr_image_name = "whole_" + os.path.splitext(base_image)[0] + ".zarr"

        with tempfile.TemporaryDirectory() as temp_dir:

            with bfio.BioReader(image_path, max_workers=2) as br_image:

                # Get the image_shape of the image that we are predicting from
                image_shape = br_image.shape

                amount_to_pad = lambda x : int(min(abs(x - np.floor(x/1024)*1024), abs(x - np.ceil(x/1024)*1024))) 
                biowriter_padding = [amount_to_pad(shape) if shape != 1 else 0 for shape in image_shape ]
                pmin = 1
                pmax = 99.8
                splinedist_model = SplineDist2D(None, name='/home/ec2-user/workdir/splinedist_scalability/models')
                
                

                tiled_output_zarr = os.path.join(temp_dir, tiled_zarr_image_name)
                with bfio.BioWriter(tiled_output_zarr,
                                Y = image_shape[0] + biowriter_padding[0],
                                X = image_shape[1] + biowriter_padding[1],
                                Z = image_shape[2] + biowriter_padding[2],
                                C = image_shape[3] + biowriter_padding[3],
                                T = image_shape[4] + biowriter_padding[4],
                                dtype=np.float64) as tiled_bw_pred:

                    
                    
                    splinedist_prediction_lambda = lambda input_intensity_image: \
                                prediction_splinedist(intensity_img=input_intensity_image, 
                                                      model=splinedist_model, 
                                                      pmin_val=pmin, 
                                                      pmax_val=pmax)

                    scal_utils.scalable_prediction(bioreader_obj=br_image,
                                                    biowriter_obj=tiled_bw_pred,
                                                    biowriter_obj_location = tiled_output_zarr,
                                                    overlap_size =(24,24,0,0,0),
                                                    prediction_fxn=splinedist_prediction_lambda)


                whole_output_zarr = os.path.join(temp_dir, whole_zarr_image_name)
                with bfio.BioWriter(whole_output_zarr,
                                Y = image_shape[0] + biowriter_padding[0],
                                X = image_shape[1] + biowriter_padding[1],
                                Z = image_shape[2] + biowriter_padding[2],
                                C = image_shape[3] + biowriter_padding[3],
                                T = image_shape[4] + biowriter_padding[4],
                                dtype=np.float64) as whole_bw_pred:
                    whole_bw_pred[:] = prediction_splinedist(intensity_img=br_image[:], 
                                                             model=splinedist_model, 
                                                             pmin_val=pmin, 
                                                             pmax_val=pmax)

            with bfio.BioReader(tiled_output_zarr, max_workers=2) as br_tiled:
                with bfio.BioReader(whole_output_zarr, max_workers=2) as br_whole:
                    
                    tiled_image = br_tiled[:]
                    whole_image = br_whole[:]

                    step = 256
                    for y1 in range(0, image_shape[0], step):
                        for x1 in range(0, image_shape[1], step):

                            y2 = y1 + step
                            x2 = x1 + step
                            print("Y: ({}, {}), X: ({}, {})".format(y1,y2,x1,x2))

                            chunk_tiled_image = tiled_image[y1:y2, x1:x2]
                            chunk_whole_image = whole_image[y1:y2, x1:x2]

                            unravel_chunk_tiled = chunk_tiled_image.ravel()
                            unravel_chunk_whole = chunk_whole_image.ravel()

                            unique_chunked_tiled = np.unique(chunk_tiled_image)
                            unique_chunked_whole = np.unique(chunk_whole_image) 


                            if len(unique_chunked_tiled) == len(unique_chunked_whole):

                                overlap = np.array(list(zip(unravel_chunk_tiled,unravel_chunk_whole)), 
                                                dtype=('i4,i4')).reshape(chunk_whole_image.shape)
                                unique_overlap = np.unique(overlap)
                                unique_overlap = [list(uni) for uni in unique_overlap if 0 not in uni]

                                unique_frequency = {}
                                for uni in unique_overlap:
                                    if uni[0] in unique_frequency.keys():
                                        unique_frequency[uni[0]] += 1
                                    else:
                                        unique_frequency[uni[0]] = 1
                                
                                for freq in unique_frequency.keys():
                                    if unique_frequency[freq] == 1:
                                        for uni in unique_overlap:
                                            if uni[0] == freq:
                                                looking_for = uni[1]
                                                looking_for_list = [uni for uni in unique_overlap if (uni[1] == looking_for) and (uni[0] != freq)]
                                                for look in looking_for_list:
                                                    unique_overlap.remove(look)

                                new_chunk_tiled_image = np.zeros(chunk_tiled_image.shape)
                                for uni in unique_overlap:
                                    new_chunk_tiled_image[chunk_tiled_image==uni[1]] = uni[0]

                                chunk_tiled_image = new_chunk_tiled_image
                                del new_chunk_tiled_image
                                unravel_chunk_tiled = chunk_tiled_image.ravel()

                                scores = get_scores(y_true = unravel_chunk_whole,
                                                    y_pred = unravel_chunk_tiled,
                                                    average = 'macro')

                                print(scores)

                            else:

                                binary_chunk_whole = chunk_whole_image.copy()
                                binary_chunk_whole[binary_chunk_whole > 0] = 1
                                binary_chunk_tiled = chunk_tiled_image.copy()
                                binary_chunk_tiled[binary_chunk_tiled > 0] = 1

                                unravel_binary_chunk_whole = binary_chunk_whole.ravel()
                                unravel_binary_chunk_tiled = binary_chunk_tiled.ravel()

                                scores = get_scores(y_true = unravel_chunk_whole,
                                                    y_pred = unravel_chunk_tiled,
                                                    average = 'binary')

                                print(scores)
                            
                            print(" ")

# %%
if __name__ == '__main__':
    unittest.main()

