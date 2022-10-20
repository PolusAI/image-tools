
import argparse, logging
from concurrent.futures import ThreadPoolExecutor

import filepattern
from filepattern import FilePattern as fp

from typing import Any

import numpy as np
import cv2

import os
import utils

POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

def main(inp_dir: str,
         out_dir: str,
         int_kernel: int,
         threshold_area_rm_large: int,
         threshold_area_rm_small: int,
         iterations_dilation: int,
         iterations_erosion: int,
         operations: str,
         file_pattern: str,
         override_instances: bool,
         structuring_shape: Any):

    try:

        # Getting the relevant input images based on filepattern
        input_images = [str(f[0]['file']) 
                        for f in fp(inp_dir,file_pattern) 
                            if os.path.exists(str(f[0]['file']))]
        num_inputs = len(input_images)
        logger.info(f"Number of Inputs = {num_inputs}")

        # A dictionary specifying the function that will be run based on user input. 
        dispatch = {
            'inversion': utils.invert_binary,
            'opening': utils.open_binary,
            'closing': utils.close_binary,
            'morphological_gradient': utils.morphgradient_binary,
            'dilation': utils.dilate_binary,
            'erosion': utils.erode_binary,
            'fill_holes': utils.fill_holes_binary,
            'skeleton': utils.skeleton_binary,
            'top_hat': utils.tophat_binary,
            'black_hat': utils.blackhat_binary,
            'filter_area_remove_large_objects': utils.areafiltering_remove_larger_objects_binary,
            'filter_area_remove_small_objects': utils.areafiltering_remove_smaller_objects_binary
        }

        # Additional arguments for each function
        dict_n_args = {
            'inversion': None,
            'opening': None,
            'closing': None,
            'morphological_gradient': None,
            'dilation': iterations_dilation,
            'erosion': iterations_erosion,
            'fill_holes': None,
            'skeleton': None,
            'top_hat': None,
            'black_hat': None,
            'filter_area_remove_large_objects' : threshold_area_rm_large,
            'filter_area_remove_small_objects' : threshold_area_rm_small
        }

        # Need extra padding when doing operations so it does not skew results
            # Initialize variables based on operation
        
        if (threshold_area_rm_large == None) and (threshold_area_rm_small == None):
            assert int_kernel != None, "Need to Specify a Kernel Size"
            logger.info(f"Integer Kernel Size: {int_kernel}")
            extra_padding = int_kernel
            kernel = cv2.getStructuringElement(structuring_shape,(int_kernel,int_kernel))
        else:
            extra_padding = 512
            kernel = None
        function = dispatch[operations]
        extra_arguments = dict_n_args[operations]

        logger.info("\n Initializing key word arguments ...")

        kwargs = {
            "function"        : function,
            "extra_arguments" : extra_arguments,
            "extra_padding"   : extra_padding,
            "kernel"          : kernel,
            "override"        : override_instances,
            "operation"       : operations
        }

        # Loop through files in inpDir image collection and process
        # with ThreadPoolExecutor(max_workers = os.cpu_count()-1) as executor:
        lambda_utilsBinaryOperation = lambda input_path: \
                                             utils.binary_operation(input_path      = input_path, 
                                                                    output_path     = os.path.join(out_dir, os.path.basename(input_path)),
                                                                    **kwargs)

        counter = 1
        with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
            for input_imagepath in input_images: # iterating through so user can keep track of progress (instead of executor map)
                output_imagepath = executor.submit(lambda_utilsBinaryOperation, input_imagepath)
                logger.info(f"Saving Output ({counter}/{num_inputs}) at {output_imagepath.result()}")
                counter += 1

    except Exception as e:
        raise ValueError(f"Something went wrong: {e}")


if __name__=="__main__":

    # Setup the argument parsing
    logger.info("\n Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a WIPP plugin.')

    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--operation', dest='operations', type=str,
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--structuringShape', dest='structuringShape', type=str,
                        help='Shape of the structuring element can either be Elliptical, Rectangular, or Cross', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='File Pattern for images in Input Directory', default=".*", required=False)
    parser.add_argument('--overrideInstances', dest='overrideInstances', type=str,
                        help='Boolean for whether instances are allowed to be overlapped or not')

    # Extra arguments based on operation
    parser.add_argument('--kernelSize', dest='kernelSize', type=int,# not used for the area filtering
                        help='Kernel size that should be used for all operations')
    parser.add_argument('--thresholdAreaRemoveLarge', dest='thresholdAreaRemoveLarge', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--thresholdAreaRemoveSmall', dest='thresholdAreaRemoveSmall', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--iterationsDilation', dest='iterationsDilation', type=int,
                        help='Number of Iterations to apply operation', required=False)
    parser.add_argument('--iterationsErosion', dest='iterationsErosion', type=int,
                        help='Number of Iterations to apply operation', required=False)

    # Input arguments
    args = parser.parse_args()
    inp_dir: str = args.inpDir
    logger.info(f"inpDir = {inp_dir}")
    out_dir: str = args.outDir
    logger.info(f"outDir = {out_dir}")
    operations: str = args.operations
    logger.info(f"operation = {operations}")
    file_pattern: str = args.filePattern
    logger.info(f"filePattern = {file_pattern}")
    override_instances: bool = args.overrideInstances
    logger.info(f"overrideInstances = {override_instances}")


    if args.structuringShape == 'Elliptical':
        structuring_shape = cv2.MORPH_ELLIPSE # datatype is INT, but throws TypeError if specified
    elif args.structuringShape == 'Rectangular':
        structuring_shape = cv2.MORPH_RECT
    elif args.structuringShape == 'Cross':
        structuring_shape = cv2.MORPH_CROSS
    else:
        raise ValueError("Structuring Shape is not correct")
    logger.info(f"Structuring Shape = {args.structuringShape}")

    int_kernel: int = args.kernelSize
    threshold_area_rm_large: int = args.thresholdAreaRemoveLarge
    threshold_area_rm_small: int = args.thresholdAreaRemoveSmall
    iterations_dilation: int = args.iterationsDilation
    iterations_erosion: int  = args.iterationsErosion

    if 'filter_area_remove_large_objects' in operations:
        if threshold_area_rm_large == None:
            raise ValueError('Need to specify the maximum area of the segments to keep')

    if 'filter_area_remove_small_objects' in operations:
        if threshold_area_rm_small == None:
            raise ValueError('Need to specify the minimum area of the segments to keep')

    if 'dilation' in operations:
        if iterations_dilation == None:
            raise ValueError("Need to specify the number of iterations to apply the operation")

    if 'erosion' in operations:
        if iterations_erosion == None:
            raise ValueError("Need to specify the number of iterations to apply the operation")

    main(inp_dir    = inp_dir,
         out_dir    = out_dir,
         int_kernel = int_kernel,
         threshold_area_rm_large = threshold_area_rm_large,
         threshold_area_rm_small = threshold_area_rm_small,
         iterations_dilation = iterations_dilation,
         iterations_erosion  = iterations_erosion,
         operations = operations,
         structuring_shape = structuring_shape,
         file_pattern = file_pattern,
         override_instances = override_instances)

