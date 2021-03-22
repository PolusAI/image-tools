
import argparse, logging, traceback
from multiprocessing import Process

import numpy as np
import cv2

from pathlib import Path
import os
import utils

Tile_Size = 256

def main():

    try:
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

        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir()]
        logger.info("Files in input directory: {}".format(inpDir_files))

        # Need extra padding when doing operations so it does not skew results
            # Initialize variables based on operation
        if (threshold_area_rm_large != None) or (threshold_area_rm_small != None):
            extra_padding = int(Tile_Size//2)
            kernel = None
        else:
            extra_padding = intkernel
            kernel = cv2.getStructuringElement(structshape,(intkernel,intkernel))
        function = dispatch[operations]
        extra_arguments = dict_n_args[operations]


        # Loop through files in inpDir image collection and process
        for image in inpDir_files:

            p = Process(target=utils.binary_operation, args=(os.path.join(inpDir, image), os.path.join(outDir, image), 
                function, extra_arguments, extra_padding, kernel, Tile_Size))
            p.start()
            p.join()


    except:
        traceback.print_exc()

    # Always close the JavaBridge
    finally:
        logger.info('Closing the javabridge...')


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a WIPP plugin.')

    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--Operation', dest='operations', type=str,
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--structuringshape', dest='struct_shape', type=str,
                        help='Shape of the structuring element can either be Elliptical, Rectangular, or Cross', required=True)

    # Extra arguments based on operation
    parser.add_argument('--kernelsize', dest='all_kernel', type=int, # not used for the area filtering
                        help='Kernel size that should be used for all operations', required=False)
    parser.add_argument('--ThresholdAreaRemoveLarge', dest='threshold_area_rm_large', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--ThresholdAreaRemoveSmall', dest='threshold_area_rm_small', type=int,
                        help='Area threshold of objects in image', required=False)
    parser.add_argument('--IterationsDilation', dest='num_iterations_dilation', type=int,
                        help='Number of Iterations to apply operation', required=False)
    parser.add_argument('--IterationsErosion', dest='num_iterations_erosion', type=int,
                        help='Number of Iterations to apply operation', required=False)
    # Input arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    operations = args.operations
    logger.info('Operation = {}'.format(operations))

    # structshape = cv2.MORPH_ELLIPSE
    if args.struct_shape == 'Elliptical':
        structshape = cv2.MORPH_ELLIPSE
    elif args.struct_shape == 'Rectangular':
        structshape = cv2.MORPH_RECT
    elif args.struct_shape == 'Cross':
        structshape = cv2.MORPH_CROSS
    else:
        raise ValueError("Structuring Shape is not correct")
    logger.info('Structuring Shape = {}'.format(args.struct_shape))

    intkernel = args.all_kernel
    threshold_area_rm_large = args.threshold_area_rm_large
    threshold_area_rm_small = args.threshold_area_rm_small
    iterations_dilation = args.num_iterations_dilation
    iterations_erosion = args.num_iterations_erosion

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

    main()

