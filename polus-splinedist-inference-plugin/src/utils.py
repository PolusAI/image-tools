import sys, os 

import numpy as np

import bfio
from bfio import BioReader, BioWriter, LOG4J, JARS


import itertools
import concurrent

from csbdeep.utils import normalize
from csbdeep.utils import normalize_mi_ma
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

import filepattern
from filepattern import FilePattern as fp

import tempfile

# Import environment variables, if POLUS_LOG empty then automatically sets to INFO
import logging

POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("infer")
logger.setLevel(POLUS_LOG)

class BfioReaderLessThan5D():
    def __init__(self, 
                 br: bfio.bfio.BioReader, 
                 n_dims: int) -> None:
        """This function initializes the wrapper for bfio.BioReader 
        Objects, so it can read only the YX axes
        Args:
            br: the input bfio reader object
            n_dims: the number dimensions we want to output
        Returns:

        """
        self.br     = br
        self.ndim = n_dims
        self.shape  = br.shape[:self.ndim]
    
    def __getitem__(self, 
                    keys : tuple) -> np.ndarray:
        """Returns reshaped numpy array that has n_dims
            dimensions"""
        return_img = self.br[keys]
        return_img_shape = return_img.shape[:self.ndim]
        return np.reshape(return_img, return_img_shape)
    
class BfioWriterWrapper():
    def __init__(self, 
                 bw: bfio.bfio.BioWriter, 
                 br: bfio.bfio.BioReader, 
                 shape: tuple) -> None:
        """ This class is wrapper for bfio.BioWriter objects
        This class allows any slices to be written to the Biowriter
        by opening up the necessary tiles. 
        Args:
            bw: input BioWriter object that gets written to
            br: input BioReader object that reads the biowriter (bw)
            shape: the shape of the biowriter and bioreader
        """
        self.br = br
        self.bw = bw
        
        assert self.bw._file_path == self.br._file_path, \
            f"File Paths for BioWriter ({self.bw._file_path}) and " + \
            f"BioReader ({self.br._file_path}) do not match"
        
        self.shape = shape
        self.TILE_SIZE = bw._TILE_SIZE
        self.ndim = len(shape)

    def __getitem__(self, keys):
        """ Reads image from the biowriter """
        return self.br[keys]

    def __setitem__(self, keys, newvalue):
        """ The bounding box of the pixels being written to 
            biowriter """
        min_bounds = []
        max_bounds = []
        for key in keys:
            min_bounds.append(key.start)
            max_bounds.append(key.stop)

        self.write_block([min_bounds, max_bounds], newvalue)

    def get_nodes(self, min_tile, max_tile):
        """ This function returns a list of tiles write_block 
            iterates through so that it can properly write to
            the biowriter 
        Args:
            min_tile : the min nodes of the bounding box for biowriter
            max_tile : the max nodes of the bounding box for biowriter
        Returns:
            combinations: a list of nodes that the class iterates through 
        """
        nodes = [list(range(min_tile[n], max_tile[n]+1)) \
                    for n in range(self.ndim)]
        combinations = itertools.product(*nodes)
        
        return combinations

    def write_block(self, 
                    coords: tuple, 
                    pixels: np.ndarray) -> None:
        """This function writes to the biowriter by normalizing the tiles to nodes.
        It iterates through the necessary nodes (usually sizes of (tile_size, tile_size)
        and saves the new pixels one tile at a time.
        Args:
            coords: the bounding box of where the pixels go in the biowriter
            pixels: the pixels that get saved to the biowriter
        """

        min_bound, max_bound = np.array(coords)

        assert pixels.ndim == len(min_bound), \
            f"Number of Dimensions in Min Bounds ({min_bound}) does NOT match " + \
                f"Number of Dimensions in Pixels ({pixels.ndim})" 
        assert pixels.ndim == len(max_bound), \
            f"Number of Dimensions in Min Bounds ({max_bound}) does NOT match " + \
                f"Number of Dimensions in Pixels ({pixels.ndim})" 
        assert (max_bound - min_bound == pixels.shape).all()

        # the tile index of bounding box
        min_tile = min_bound // self.TILE_SIZE
        max_tile = max_bound // self.TILE_SIZE

        # nodes to iterate through
        nodes_to_iterate = self.get_nodes(min_tile, max_tile)
        for node_to_iterate in nodes_to_iterate:
            node_to_iterate = np.array(node_to_iterate)

            # indices for the tiles - typically multiples of TILE_SIZE
            tile_origin = node_to_iterate*self.TILE_SIZE
            tile_bounds = np.minimum((node_to_iterate+1)*self.TILE_SIZE, self.shape)

            # indices of what values we care about in the pixels that are getting saved in biowriter
            min_img     = np.maximum(tile_origin - min_bound, np.zeros(self.ndim))
            max_img     = np.minimum(tile_bounds - min_bound, pixels.shape[:self.ndim])

            # indices of the new tile that get updated by pixels
                # Tiles need to be written to the biowriter. 
                # These indices are offfset by the min_img, max_img 
            min_local = (min_bound - tile_origin) + min_img
            max_local = (min_bound - tile_origin) + max_img

            slices_tiles = tuple(slice(int(tile_origin[n]),   int(tile_bounds[n])) \
                for n in range(self.ndim))
            slices_local = tuple(slice(int(min_local[n]), int(max_local[n])) \
                for n in range(self.ndim))
            slices_img   = tuple(slice(int(min_img[n]), int(max_img[n])) \
                for n in range(self.ndim))
            logger.debug(f"Slices Tiles: {slices_tiles}")
            logger.debug(f"Slices Local: {slices_local}")
            logger.debug(f"Slices Image: {slices_img}")

            # if the biowriter exists
            if self.br._file_path:
                tile = self.br[slices_tiles]
                tile[slices_local] = pixels[slices_img]
                self.bw[slices_tiles] = tile


def get_ranges(image_shape : tuple,
               window_size : tuple,
               step_size   : tuple):

    """ This function generates a list of yxczt dimensions to iterate 
    through so that bfio objects can be read in tiles. 
    Args:
        image_shape : shape of the whole bfio object
        tile_len : the size of tiles
    Returns:
        yxzct_ranges : a zipped list of dimensions to iterate through. 
                       [((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2)), ... ,
                        ((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2))]
    """

    ndims = len(image_shape)
    nodes = [[slice(int(i), int(image_shape[n])) if (i+window_size[n] > image_shape[n]) \
                    else slice(int(i), int(i+window_size[n])) \
                    for i in range(0, image_shape[n], step_size[n])] \
                        for n in range(ndims)]
    # https://docs.python.org/3/library/itertools.html#itertools.product
    yxzct_ranges = itertools.product(*nodes)
    return yxzct_ranges

def get_image_minmax(br_image   : bfio.bfio.BioReader, 
                     image_size : np.ndarray,
                     tile_size  : int):

    """ This function is used when the image is analyzed in tiles, 
        because of how large it is.  For splinedist, it is necessary 
        to the global min and max value from the entire image.  However,
        the entire the image may not be able to load into memory, so we 
        iterate through it in tiles and constantly update the values
    Args:
        br_image: input bfio object 
        image_size: size of the bfio object
        tile_size: size of the tiles to read bfio object in
    Returns:
        image_min_val: the smallest value in the bfio object
        image_max_val: the largest value in the bfio object
    """

    # (lambda) function to get the min and max values of the tiles
    get_min_max = lambda image: [np.min(image), np.max(image)]
    
    # size of tiles to iterate through
    tile_size_dims = [tile_size] * len(image_size)

    # appending all the min and max values of every tile into a list
    with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
        futures = executor.map(get_min_max, (br_image[yxzct_slice] \
            for yxzct_slice in get_ranges(image_shape = image_size,
                                          window_size = tile_size_dims,
                                          step_size   = tile_size_dims)))
    
    # going through all the min and max values of the tiles and 
    # getting the global min and max value.
    min_max_values  = np.array(list(futures))
    image_min_value = np.min(min_max_values[:, 0])
    image_max_value = np.max(min_max_values[:, 1])

    return image_min_value, image_max_value

def prediction_splinedist(normalized_img : bfio.bfio.BioReader, 
                          model          : SplineDist2D, 
                          bw             : bfio.bfio.BioWriter,
                          bw_location    : str,
                          min_val        = None,
                          max_val        = None,):
    """ This function is used as an input for the scalabile_prediction algorithm.
        This function generates a mask for intensity_img using SplineDist. 
        Args:
            intensity_img : the intensity-based input image
            model : the SplineDist model that runs the prediction on the input
            min_val : the smallest global value in input intensity image
            max_val : the largest global value  in input intensity image
    """
    # Get shape of input
    input_img_shape = normalized_img.shape
    
    bw[:] = np.zeros(bw.shape)
    with BioReader(bw_location, backend='zarr') as read_bw:
        # if large image, then run scalable predictions
        input_img = BfioReaderLessThan5D(normalized_img, 2) # need a wrapper to input 2D bfio reader object
        if max(input_img_shape) > bw._TILE_SIZE:
            # need a wrapper for bfio writer object
            t_wrapper = BfioWriterWrapper(bw=bw, br=read_bw, shape=bw.shape[:2])
            # min_overlap is the minimum amount of overlap between the blocks.
            # For example (one dimension): 
                # Size: 1080, min_overlap = 56, block_size = 1024
                # slices can vary from 0:1024, 56:1080) to (0:1024, 968:1080) 
            min_overlap = np.mod(t_wrapper.shape, bw._TILE_SIZE)
            min_overlap[min_overlap == 0] = 32
            block_size = min_overlap + 1024
            block_size = np.min((block_size,t_wrapper.shape-min_overlap), axis=0)
            pred, _ = model.predict_instances_big(img = input_img, axes='YX', \
                block_size=block_size, min_overlap = min_overlap, labels_out = t_wrapper, context=(0,0)) 
        # otherwise save output directly to output
        else:
            bw[:], _ = model.predict_instances(img = input_img) #splinedist calls input_img.ndim, 
                                                    # BfioReaderLessThan5D object has that attribute


def predict_nn(image_dir : str,
                base_dir : str,
                output_directory : str,
                gpu : bool,
                imagepattern : str,):
    """ The function uses the model saved in base_dir to predict the 
    segmentations of the images in the image_dir.

    Args:
    image_dir: Directory with all the images 
    base_dir: Directory of the model's weights
    output_directory: Directory where the outputs gets saved
    gpu: Specifies whether or not there is a GPU to use
    imagepattern: Pattern of the files the user wants to use

    Returns:
    None, an output_directory filled with jpeg images of segmentations
        and the performance of the neural network on every image

    Raises:
    Assertion Error: 
        If the base_dir does not exist
        If the phi_file does not exist
        If the grid_file does not exist
    """

    assert os.path.exists(base_dir), \
        "{} does not exist".format(base_dir)

    # Change working dir to model base directory
        # Model directory will most likely contain 
        # phi and grid numpy files, and those files
        # must be in the current working directory
    os.chdir(base_dir)

    # grab the images that match imagepattern
    fp_images = fp(image_dir,imagepattern)
    images = []
    for files in fp_images():
        image = files[0]['file']
        if os.path.exists(image):
            images.append(image)
    num_images = len(images)

    # Load the Model
    model_dir_name = '.'
    model = SplineDist2D(None, name=model_dir_name, basedir=base_dir)
    logger.info("\n Done Loading Model ...")

    logger.info("\n Parameters in Config File ...")
    config_dict = model.config.__dict__
    for ky,val in config_dict.items():
        logger.info("{}: {}".format(ky, val))
    
    logger.info("\n Looking for Extra Files ...")
    controlPoints = int(config_dict['n_params']/2)
    logger.info("Number of Control Points: {}".format(controlPoints))

    # make sure phi and grid exist in current directory, otherwise create.
    logger.info("\n Getting extra files ...")

    conf = model.config
    logger.info("\n Parameters in Config File ...")
    for ky,val in model.config.__dict__.items():
        logger.info("{}: {}".format(ky, val))
        
    M = int(conf.n_params/2)
    if not os.path.exists("./phi_{}.npy".format(M)):
        contoursize_max = conf.contoursize_max
        logger.info("Contoursize Max for phi_{}.npy: {}".format(M, contoursize_max))
        phi_generator(M, contoursize_max, '.')
        logger.info("Generated phi")
    if not os.path.exists("./grid_{}.npy".format(M)):
        training_patch_size = conf.train_patch_size
        logger.info("Training Patch Size for grid_{}.npy: {}".format(training_patch_size, M))
        grid_generator(M, training_patch_size, conf.grid, '.')
        logger.info("Generated grid")

    # sanity check
    assert os.path.exists(f"./phi_{M}.npy")
    assert os.path.exists(f"./grid_{M}.npy")
    
    axis_norm = (0,1)

    # img_counter is for the logger
    img_counter = 1
    for im in images:
        
        with tempfile.TemporaryDirectory() as temp_zarr_directory:

            # initialize file names
            base_image = os.path.basename(im) # base image
            # need a normalized image.  The whole normalized image gets passed into splinedist.  
                # Do not have control of what tiles get passed into splinedist
            normalized_image = os.path.join(temp_zarr_directory, 'normalized_' + os.path.splitext(base_image)[0] + ".zarr")
            # end product: output zarr file 
            output_zarr_path = os.path.join(output_directory, os.path.splitext(base_image)[0] + ".zarr")

            tile_size = 1024
            with BioReader(im) as br_image:
                
                # creates a normalized image, need the min and max of input image
                br_image_shape = br_image.shape
                global_min, global_max = get_image_minmax(br_image   = br_image, 
                                                          image_size = br_image.shape,
                                                          tile_size  = tile_size)
                pmin_val = np.percentile([[global_min, global_max]], 1,    axis=(0,1), keepdims=True)
                pmax_val = np.percentile([[global_min, global_max]], 99.8, axis=(0,1), keepdims=True)

                # saves the normalized in a temporary file
                with BioWriter(file_path=normalized_image,
                                      Y = br_image.Y,
                                      X = br_image.X,
                                      Z = br_image.Z,
                                      C = br_image.C,
                                      T = br_image.T, 
                                  dtype = np.float64, backend='zarr') as br_norm:
                    
                    for yxzct_slice in get_ranges(image_shape = br_image_shape,
                                                  window_size = (tile_size,tile_size,1,1,1),
                                                  step_size   = (tile_size,tile_size,1,1,1)):
                        
                        tiled_intensity_image = br_image[yxzct_slice]
                        normalized_tile       = normalize_mi_ma(tiled_intensity_image, mi=pmin_val, ma=pmax_val)
                        br_norm[yxzct_slice]  = normalized_tile
            
                br_norm.close()
            br_image.close()

            # use the normalized images to make the predictions
            model.config.axes = "YXC"
            with BioReader(normalized_image, backend='zarr', max_workers=1) as br_normed:
                with BioWriter(file_path = output_zarr_path, \
                               metadata  = br_normed.metadata, \
                                   dtype = np.int32, backend='zarr') as bw:
        
                    prediction_splinedist(normalized_img = br_normed,
                                          model          = model,
                                          bw             = bw,
                                          bw_location    = output_zarr_path)
                bw.close()
            br_normed.close()
                
            logger.info("{}/{} -- Created Output Prediction for {}. ".format(img_counter, num_images, base_image))
            logger.debug(f"Output Location: {output_zarr_path}")
            img_counter += 1
