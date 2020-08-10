from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
import tifffile
import cv2
import ast
from scipy import ndimage
import matplotlib.pyplot as plt

def invert_binary(image):
    invertedimg = np.zeros(image.shape,dtype=datatype)
    invertedimg = 255 - image
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

def skeleton_binary(image,kernel):
    image[image == 255] = 1
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape,np.uint8)

    while (not done):
        erode = cv2.erode(image,kernel)
        temp = cv2.dilate(erode,kernel)
        temp = cv2.subtract(image,temp)
        skel = cv2.bitwise_or(skel,temp)
        image = erode.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros==size:
            done = True

    skel = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    skel[skel == 1] = 255
    return skel

def holefilling_binary(image, output):
    image[image == 255] = 1
    sqimage = image.squeeze()

    plt.figure()
    plt.imshow(sqimage)
    plt.show()
    
    out_image = np.zeros(image.shape).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    hf = cv2.morphologyEx(sqimage,cv2.MORPH_OPEN,kernel)
    hf[hf == 1] = 255
    logger.info("in function")
    logger.info("output shape {}".format(out_image.shape))
    logger.info("hf shape {}".format(hf.shape))
    out_image[:,:,0,0,0] = np.array(hf)
    # logger.info(hf.shape)
    return out_image

def hitormiss_binary(image, HOMarray):
    image[image == 255] = 1
    hm = ndimage.binary_hit_or_miss(image, structure1=HOMarray)
    return hm

def tophat_binary(image,kernel):
    image[image == 255] = 1
    tophat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    tophat[tophat == 1] = 255
    return tophat

def blackhat_binary(image,kernel):
    image[image == 255] = 1
    blackhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT, kernel)
    blackhat[blackhat == 1] = 255
    return blackhat

def areafiltering_binary(image):
    image[image == 255] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    hf = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    hf[hf == 1] = 255
    return hf

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
    parser.add_argument('--Operations', dest='operations', type=str,
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--dilateby', dest='dilate_by', type=int,
                        help='How much do you want to dilate by?', required=False)
    parser.add_argument('--erodeby', dest='erode_by', type=int,
                        help='How much do you want to erode by?', required=False)
    parser.add_argument('--HOMarray', dest='hit_or_miss', type=str,
                        help='Whats the array that you are trying to find', required=False)

    # Parse the arguments
    args = parser.parse_args()
    # filePattern = args.filePattern
    # logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    operations = args.operations
    operations = operations.split(',')
    logger.info('Operations = {}'.format(operations))
    dilateby = args.dilate_by
    erodeby = args.erode_by
    hitormiss = args.hit_or_miss
    if hitormiss != None:
        hitormiss = np.array(ast.literal_eval(hitormiss))
    hitormiss = np.array(hitormiss)

    if 'dilation' in operations and dilateby == None:
        raise ValueError("Need to specify --dilateby integer value")
    else:
        logger.info('Dilating by {}'.format(dilateby))

    if 'erosion' in operations and erodeby == None:
        raise ValueError("Need to specify --erodeby integer value")
    else:
        logger.info('Eroding by {}'.format(erodeby))

    if 'hit_or_miss' in operations and hitormiss.any() == None:
        raise ValueError("Need to specify --HOMarray integer value")
    else:
        logger.info('Array that we are locating: {}'.format(hitormiss))
    
    
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir()]
    logger.info("Files in input directory: {}".format(inpDir_files))

      
    # Loop through files in inpDir image collection and process
    imagenum = 0
    try:
        for f in inpDir_files:
            # Load an image
            br = BioReader(str(Path(inpDir).joinpath(f)))
            # image = np.squeeze(br.read_image())
            image = br.read_image()
            datatype = br._pix['type']
            out_image = np.zeros(image.shape).astype('uint8')
            logger.info(out_image.shape)
            kernel = np.ones((3,3), np.uint8) 
            # newdir = Path(outDir).joinpath(f)
            # newdir.mkdir()
            out_image = None
            newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op0')
            tifffile.imwrite(str(newfile), image, photometric='minisblack')
            logger.info(newfile)

    
            i = 0
            for item in operations:
                if i == 0:
                    if item == 'invertion':
                        out_image = invert_binary(image)
                        # newfile = Path(outDir).joinpath('inverted_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'opening':
                        out_image = open_binary(image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('open_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'closing':
                        out_image = close_binary(image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('close_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == "morphological_gradient":
                        out_image = morphgradient_binary(image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('morphgradient_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'dilation':
                        out_image = dilate_binary(image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('dilated_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'erosion':
                        out_image = erode_binary(image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('eroded_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'skeleton':
                        out_image = skeleton_binary(image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('skeleton_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'fill_holes':
                        out_image = holefilling_binary(image, out_image)
                        # newfile = Path(outDir).joinpath('holefilled_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='rgb')
                        try:
                            hi = BioReader(str(Path(newfile)))
                            newim = hi.read_image()
                        except Exception as e:
                            logger.info(e)
                    elif item =='hit_or_miss':
                        out_image = hitormiss_binary(image.squeeze(),hitormiss)
                        # newfile = Path(outDir).joinpath('hitormiss_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'filter_area':
                        out_image = areafiltering_binary(image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('areafiltered_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'top_hat':
                        out_image = tophat_binary(image.squeeze(), kernel)
                        # newfile = Path(outDir).joinpath('tophat_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'black_hat':
                        out_image = blackhat_binary(image.squeeze(), kernel)
                        # newfile = Path(outDir).joinpath('blackhat_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    else:
                        raise ValueError("Operation value is incorrect")
                else:
                    if item == 'invertion':
                        out_image = invert_binary(out_image)
                        # newfile = Path(outDir).joinpath('inverted_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'opening':
                        out_image = open_binary(out_image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('open_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'closing':
                        out_image = close_binary(out_image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('close_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == "morphological_gradient":
                        out_image = morphgradient_binary(out_image.squeeze(),kernel)
                        # newfile = Path(outDir).joinpath('morphgradient_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'dilation':
                        out_image = dilate_binary(out_image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('dilated_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'erosion':
                        out_image = erode_binary(out_image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('eroded_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'skeleton':
                        out_image = skeleton_binary(out_image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('skeleton_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'fill_holes':
                        out_image = holefilling_binary(out_image.squeeze())
                        # newfile = Path(outDir).joinpath('holefilled_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item =='hit_or_miss':
                        out_image = hitormiss_binary(out_image.squeeze(),hitormiss)
                        # newfile = Path(outDir).joinpath('hitormiss_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'filter_area':
                        out_image = areafiltering_binary(out_image.squeeze(), kernel,dilateby)
                        # newfile = Path(outDir).joinpath('areafiltered_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'top_hat':
                        out_image = tophat_binary(out_image.squeeze(), kernel)
                        # newfile = Path(outDir).joinpath('tophat_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    elif item == 'black_hat':
                        out_image = blackhat_binary(out_image.squeeze(), kernel)
                        # newfile = Path(outDir).joinpath('blackhat_'+f)
                        newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1))
                        logger.info(newfile)
                        tifffile.imwrite(str(newfile), out_image, photometric='minisblack')
                    else:
                        raise ValueError("Operation value is incorrect")
                i = i + 1
            imagenum = imagenum + 1
        logger.info('Closing the javabridge...')
        jutil.kill_vm()

        # Write the output
        # bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
        # bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z,1,1)))
    
    except Exception as e:
        logger.info(e)
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
        

# %%


# %%
