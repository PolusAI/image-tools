# Tile Prediction Utility

Function to use a trained model to create labelled data in tiles rather than all at once to save on memory usage.

# Tiled Prediction Function

The function containing the algorithm that allows users to predict in tiles is written in `predict_tiles.py`.  
This function is able to reconcile labelled data that occurs at the edges of the tiles.   
More information on the function is written in the python file. 

# Tiled Prediction Inputs 

| Name                    | Description                                                                     | I/O    | Type                  |
|-------------------------|---------------------------------------------------------------------------------|--------|-----------------------|
| `bioreader_obj`         | Input Bfio Object to Reader from                                                | Input  | bfio.bfio.BioReader   |  
| `biowriter_obj`         | Input Bfio Object to Write from                                                 | Input  | bfio.bfio.BioWriter   |
| `overlap_size`          | The amount of overlap between every tile                                        | Input  | tuple                 |
| `prediction_fxn`        | Function that generates Labelled Data for the tiles                             | Input  | Callable              |
| `tile_size`             | The size of the tile - usually (1024, 1024, 1, 1, 1) to be compatible with Bfio | Input  | Tuple                 |
| `biowriter_obj_location`| Location for where the output bfio object should be saved                       | Output | String                |


## Examples

### Making predictions on an image with SplineDist 
https://github.com/uhlmanngroup/splinedist

```python
import bfio 
from bfio import BioReader, BioWriter 

import numpy

from csbdeep.utils import normalize
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D

from tiledpredictions.predict_tiles import predict_in_tiles


# need to define inputs
model_path = ""
image_path = ""



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



def main():

        # Get image location parameters
        image_directory = os.path.dirname(image_path)
        base_image = os.path.basename(image_path)
        # the two output images are zarr images
        tiled_zarr_image_name = "tiled_" + os.path.splitext(base_image)[0] + ".zarr"
        whole_zarr_image_name = "whole_" + os.path.splitext(base_image)[0] + ".zarr"

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
                predict_in_tiles(bioreader_obj=br_image,
                                 biowriter_obj=tiled_bw_pred,
                                 biowriter_obj_location = tiled_output_zarr,
                                 overlap_size =(24,24,0,0,0),
                                 prediction_fxn=splinedist_prediction_lambda)

main()

```
