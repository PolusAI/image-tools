import os, sys
import unittest
import importlib

# initializing parent directory
polus_scalable = os.path.abspath(__file__ + 2 * '/..')
sys.path.append(polus_scalable)
from scalable_prediction.scalability import \
    scalable_prediction

# libraries related to reading and writing from input and outputs, respectively
import tempfile
import bfio
from bfio import BioReader, BioWriter

# common python libraries for array manipulation
import numpy as np

# libraries relating to splinedist
from csbdeep.utils import normalize
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D


# libraries for calculating metrics
from sklearn.metrics import fbeta_score, jaccard_score




class TestEncodingDecoding(unittest.TestCase):

    def test_splinedist(self):

        # need to define inputs
        model_path = ""
        image_path = ""

        def get_scores(y_true : np.ndarray,
                       y_pred : np.ndarray,
                       average_val : str):
            """ This function is used to return metrics that help evaluate 
            the tiled prediction algorithm's accuracy.  
            Args:
                y_true : the whole prediction numpy array unraveled
                y_pred : the tiled prediction numpy array unraveled
                average_val : Either "macro" or "binary".  
                            Macro if the number of unique values is 
                                equal in y_true and y_pred
                            Binary if the number of unique values is 
                                not equal in y_true and y_pred
            Returns:
                scores : a list of the four metrics evaulated in order
                         Jaccardian Index, F1_Score, F2_Score, F3_Score
            """
            # https://en.wikipedia.org/wiki/Jaccard_index
            j_score = jaccard_score(y_true=y_true, 
                                    y_pred=y_pred,
                                    average=average_val)

            # https://en.wikipedia.org/wiki/F-score
            f1_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average_val,
                                   beta=1)
            f2_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average_val,
                                   beta=2)
            f3_score = fbeta_score(y_true=y_true, 
                                   y_pred=y_pred,
                                   average=average_val,
                                   beta=3)
            scores = [j_score, f1_score, f2_score, f3_score]
            return scores

        def prediction_splinedist(intensity_img : np.ndarray, 
                                  model : SplineDist2D, 
                                  pmin_val : float, 
                                  pmax_val : float):
            """ This function is used as an input for the scalabile_prediction algorithm.
                This function generates a mask for intensity_img using SplineDist. 
                Args:
                    intensity_img : the intensity-based input image
                    model : the SplineDist model that runs the prediction on the input
                    pmin_val : the smallest global value in input intensity image
                    pmax_val : the largest global value in input intensity image
            """
            # Get shape of input
            input_intensity_shape = intensity_img.shape

            # Normalize the input
            tiled_prediction = normalize(intensity_img, pmin=pmin_val, pmax=pmax_val, axis=(0,1),dtype=int)
            # Prediction on normalized image
            tiled_prediction, _ = model.predict_instances(tiled_prediction)
            # Reshape to be compatible with bfio objects
            tiled_prediction = np.reshape(tiled_prediction, (input_intensity_shape[0], 
                                                             input_intensity_shape[1],
                                                             1,
                                                             1,
                                                             1))
            # convert to np.float64
            tiled_prediction = tiled_prediction.astype(np.float64)

            return tiled_prediction

        # Get image location parameters
        image_directory = os.path.dirname(image_path)
        base_image = os.path.basename(image_path)
        # the two output images are zarr images
        tiled_zarr_image_name = "tiled_" + os.path.splitext(base_image)[0] + ".zarr"
        whole_zarr_image_name = "whole_" + os.path.splitext(base_image)[0] + ".zarr"

        # Generate a temporary directory to store the outputs
        with tempfile.TemporaryDirectory() as temp_dir:

            # Only one input image to be evaluated
            with bfio.BioReader(image_path, max_workers=2) as br_image:

                # Get the image_shape of the image that we are predicting from
                image_shape = br_image.shape

                # Need to make sure that the shape of the image is compatible with bfio
                amount_to_pad = lambda x : int(min(abs(x - np.floor(x/1024)*1024), 
                                        abs(x - np.ceil(x/1024)*1024))) 
                biowriter_padding = [amount_to_pad(shape) if shape != 1 else 0 for shape in image_shape ]

                # parameters for prediction_function
                pmin = 1
                pmax = 99.8
                splinedist_model = SplineDist2D(None, name=model_path)
                M = (splinedist_model.config.n_params//2)

                # When running splinedist, need to make sure that
                # the phi and grid files exist in your current directory 
                # Otherwise call the test_scalability from that directory
                assert os.path.exists(f"./phi_{M}.npy")
                assert os.path.exists(f"./grid_{M}.npy")

                # Name of the output for tiled predictions
                tiled_output_zarr = os.path.join(temp_dir, tiled_zarr_image_name)
                with bfio.BioWriter(tiled_output_zarr,
                                Y = image_shape[0] + biowriter_padding[0],
                                X = image_shape[1] + biowriter_padding[1],
                                Z = image_shape[2] + biowriter_padding[2],
                                C = image_shape[3] + biowriter_padding[3],
                                T = image_shape[4] + biowriter_padding[4],
                                dtype=np.float64) as tiled_bw_pred:

                    # define lambda function for scalable prediciton, because there
                    # are parameters that are consistent with every prediction.  Only 
                    # one input changes
                    splinedist_prediction_lambda = lambda input_intensity_image: \
                                prediction_splinedist(intensity_img=input_intensity_image, 
                                                      model=splinedist_model, 
                                                      pmin_val=pmin, 
                                                      pmax_val=pmax)

                    # Run the prediction on tiles.
                    scalable_prediction(bioreader_obj=br_image,
                                        biowriter_obj=tiled_bw_pred,
                                        biowriter_obj_location = tiled_output_zarr,
                                        overlap_size =(24,24,0,0,0),
                                        prediction_fxn=splinedist_prediction_lambda)

                # Name of the output when it is predicted all at once
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

            # Need to compare the two outputs.  "whole" vs "tiled"
            step = 256 # will compare tiles sizes of step
            num_chunks_y = image_shape[0]//step # num of chunks in y direction
            num_chunks_x = image_shape[1]//step # num of chunks in x direction
            # comparing by evaulating certain scores.  Look at get_scores function
            scores = np.zeros(shape=(int(num_chunks_y*num_chunks_x), 4)) 
            with bfio.BioReader(tiled_output_zarr, max_workers=2) as br_tiled:
                with bfio.BioReader(whole_output_zarr, max_workers=2) as br_whole:

                    # need to iterate through the two outputs in chunks
                    for y1 in range(0, image_shape[0], step):
                        for x1 in range(0, image_shape[1], step):

                            # chunks range from (y1-y2), (x1-x2)
                            y2 = y1 + step
                            x2 = x1 + step

                            # fragment position of chunks relative to the entire outputs
                            ypos, xpos = y1//step, x1//step
                            counter = int((num_chunks_y*(y1//step))+(x1//step))

                            # the chunks themselves
                            chunk_tiled_image = br_tiled[y1:y2, x1:x2]
                            chunk_whole_image = br_whole[y1:y2, x1:x2]

                            # need to unravel the chunk for get_scores fxn
                            unravel_chunk_tiled = chunk_tiled_image.ravel()
                            unravel_chunk_whole = chunk_whole_image.ravel()

                            # getting unique values to compare whether or not 
                                # the same number of labels are generated between the two
                            unique_chunked_tiled = np.unique(chunk_tiled_image)
                            unique_chunked_whole = np.unique(chunk_whole_image) 

                            # if the number of labels are equal in the two chunks
                            if len(unique_chunked_tiled) == len(unique_chunked_whole):

                                # then we try to reorder the segments to have the same label values 
                                    # between the two chunks by getting which two segments overlap when 
                                    # stacking the two chunks via depth
                                overlap = np.array(list(zip(unravel_chunk_tiled,unravel_chunk_whole)), 
                                                dtype=('i4,i4')).reshape(chunk_whole_image.shape)
                                unique_overlap = np.unique(overlap)
                                unique_overlap = [list(uni) for uni in unique_overlap if 0 not in uni]

                                # if it overlaps with more than one segment, then we assign the two segments
                                    # to the same label that have the most overlap
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

                                # the tiled predicted chunk gets updaeted to match 
                                    # the wholely predicted chunk
                                new_chunk_tiled_image = np.zeros(chunk_tiled_image.shape)
                                for uni in unique_overlap:
                                    new_chunk_tiled_image[chunk_tiled_image==uni[1]] = uni[0]

                                chunk_tiled_image = new_chunk_tiled_image
                                del new_chunk_tiled_image
                                unravel_chunk_tiled = chunk_tiled_image.ravel()

                                # get scores as to how similar the two chunks are
                                score = np.array(get_scores(y_true = unravel_chunk_whole,
                                                            y_pred = unravel_chunk_tiled,
                                                            average_val = 'macro'))

                                assert (score >= .70).all()
                                scores[counter, :] = score

                            else:
                                
                                # if th number of labels do not match then we convert 
                                    # the chunks to binary
                                binary_chunk_whole = chunk_whole_image.copy()
                                binary_chunk_whole[binary_chunk_whole > 0] = 1
                                binary_chunk_tiled = chunk_tiled_image.copy()
                                binary_chunk_tiled[binary_chunk_tiled > 0] = 1

                                # unravel the binaries so that it can be used as inputs for 
                                    # the get_scores function
                                unravel_binary_chunk_whole = binary_chunk_whole.ravel()
                                unravel_binary_chunk_tiled = binary_chunk_tiled.ravel()

                                score = np.array(get_scores(y_true = unravel_binary_chunk_whole,
                                                            y_pred = unravel_binary_chunk_tiled,
                                                            average_val = 'binary'))

                                assert (score >= .70).all()
                                scores[counter, :] = score

            # make sure that the average scores among all tiles are at least greater than
                # 85%, otherwise there may have been an error to look into.
            avg = np.average(scores, axis = 1)
            assert avg[0] >= .85, "Jaccardian Index average to less than 85% \
                                    with a score of {}".format(avg[0])
            assert avg[1] >= .85, "F1 Scores average to less than 85%, \
                                    with a score of {}".format(avg[1])
            assert avg[2] >= .85, "F2 Scores average to less than 85% \
                                    with a score of {}".format(avg[2])
            assert avg[3] >= .85, "F3 Scores average to less than 85% \
                                    with a score of {}".format(avg[3])

if __name__ == '__main__':
    unittest.main()

