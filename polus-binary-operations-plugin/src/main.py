from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile
import cv2

def invert_binary(image):
    invertedimg = np.zeros(image.shape,dtype=datatype)
    invertedimg = 255 - image
    # logger.info(image[:], invertedimg[:])
    # ones = (image == 255)
    # zeros = (image == 0)
    # invertedimg[ones] = 0
    # invertedimg[zeros] = 255
    return invertedimg

def dilate_binary(image,kernel,n):
    dilatedimg = image
    image[image == 255] = 1
    dilatedimg = cv2.dilate(image, kernel, iterations=n) 
    dilatedimg[dilatedimg == 1] = 255
    return dilatedimg

def erode_binary(image,kernel,n):
    erodedimg = image
    image[image == 255] = 1
    erodedimg = cv2.erode(image, kernel, iterations=n) 
    erodedimg[erodedimg == 1] = 255
    return erodedimg

def open_binary(image,kernel):
    openimg = image
    image[image == 255] = 1
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    openimg[openimg == 1] = 255
    return openimg

def close_binary(image,kernel):
    closeimg = image
    image[image == 255] = 1
    closeimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closeimg[closeimg == 1] = 255
    return closeimg

def morphgradient_binary(image,kernel):
    mg = image
    image[image == 255] = 1
    mg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    mg[mg == 1] = 255
    return mg

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a WIPP plugin.')
    # parser.add_argument('--filePattern', dest='filePattern', type=str,
    #                     help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--Operations', dest='operations', type=str, nargs='+',
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--dilateby', dest='dilate_by', type=int,
                        help='How much do you want to dilate by?', required=False)
    parser.add_argument('--erodeby', dest='erode_by', type=int,
                        help='How much do you want to erode by?', required=False)

    # Parse the arguments
    args = parser.parse_args()
    # filePattern = args.filePattern
    # logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    operations = args.operations
    logger.info('Operations = {}'.format(operations))
    dilateby = args.dilate_by
    erodeby = args.erode_by
    if 'dilation' in operations and dilateby == None:
        raise ValueError("Need to specify --dilateby integer value")
    else:
        logger.info('Dilating by {}'.format(dilateby))

    if 'erosion' in operations and erodeby == None:
        raise ValueError("Need to specify --erodeby integer value")
    else:
        logger.info('Eroding by {}'.format(dilateby))
    
    
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir()]
    logger.info("Files in input directory: {}".format(inpDir_files))
    
    # Loop through files in inpDir image collection and process
    for f in inpDir_files:
        # Load an image
        br = BioReader(str(Path(inpDir).joinpath(f)))
        # image = np.squeeze(br.read_image())
        image = br.read_image()
        # initialize the output
        datatype = br._pix['type']
        zero_image = np.zeros(image.shape,dtype=datatype)
        kernel = np.ones((3,3), np.uint16) 

        i = 0
        for item in operations:
            if i == 0:
                if item == 'invertion':
                    out_image = invert_binary(image)
                    newfile = Path(outDir).joinpath('inverted_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                if item == 'opening':
                    out_image = open_binary(image.squeeze(),kernel)
                    newfile = Path(outDir).joinpath('open_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                if item == 'closing':
                    out_image = close_binary(image.squeeze(),kernel)
                    newfile = Path(outDir).joinpath('close_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                if item == "morphological_gradient":
                    out_image = morphgradient_binary(image.squeeze(),kernel)
                    newfile = Path(outDir).joinpath('morphgradient_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                if item == 'dilation':
                    out_image = dilate_binary(image.squeeze(), kernel,dilateby)
                    newfile = Path(outDir).joinpath('dilated_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                if item == 'erosion':
                    out_image = erode_binary(image.squeeze(), kernel,dilateby)
                    newfile = Path(outDir).joinpath('eroded_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
            else:
                if item == 'invert':
                    out_image = invert_binary(out_image)
                    outDir.imwrite(out_image[:])
                if item == 'dilate':
                    out_image = dilate_binary(out_image, kernel,dilateby)
                    newfile = Path(outDir).joinpath('dilated_'+f)
                    logger.info(newfile)
                    tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                # if item == 'dilate':
                #     out_image = dilation(image)
            i = i + 1
        

        # Write the output
        # bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
        # bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z,1,1)))
    
    
    # Close the javabridge
    logger.info('Closing the javabridge...')
    jutil.kill_vm()
    