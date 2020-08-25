from bfio import BioReader, BioWriter,JARS
import javabridge as jutil
import bioformats
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
import tifffile
import cv2
import ast
from scipy import ndimage
from multiprocessing import cpu_count

Tile_Size = 1024

def invert_binary(image, kernel=None, intk=None, n=None):
    invertedimg = np.zeros(image.shape,dtype=datatype)
    invertedimg = 255 - image
    return invertedimg

def dilate_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    dilatedimg = cv2.dilate(image, kernel, iterations=n)
    dilatedimg[dilatedimg == 1] = 255
    return dilatedimg

def erode_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    erodedimg = cv2.erode(image, kernel, iterations=n)
    erodedimg[erodedimg == 1] = 255
    return erodedimg

def open_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg

def close_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    closeimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closeimg[closeimg == 1] = 255
    return closeimg

def morphgradient_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    mg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return mg

def skeleton_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    done = False
    size = np.size(image)
    skel = np.zeros(image.shape,datatype)

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

def holefilling_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(intk,intk))
    hf = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    hf[hf == 1] = 255
    return hf

def hitormiss_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    hm = ndimage.binary_hit_or_miss(image, structure1=n)
    return hm

def tophat_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    tophat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    tophat[tophat == 1] = 255
    return tophat

def blackhat_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    blackhat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT, kernel)
    blackhat[blackhat == 1] = 255
    return blackhat

def areafiltering_binary(image,kernel=None, intk=None, n=None):
    image[image == 255] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(intk,intk))
    af = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
    af[af == 1] = 255
    return af

def imagetiling(br, xval, yval):
    image = br.read_image(X=xval, Y=yval)
    return image

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
    parser.add_argument('--Operation', dest='operations', type=str,
                        help='The types of operations done on image in order', required=True)
    parser.add_argument('--kernelsize', dest='all_kernel', type=int,
                        help='Kernel size that should be used for all operations', required=True)
    parser.add_argument('--tilesize', dest='tile_size', type=int,
                        help='Tile size for Image Tiling, otherwise default is 1024x1024', required=False)

    parser.add_argument('--dilateby', dest='dilate_by', type=int,
                        help='How much do you want to dilate by?', required=False)
    parser.add_argument('--erodeby', dest='erode_by', type=int,
                        help='How much do you want to erode by?', required=False)
    parser.add_argument('--HOMarray', dest='hit_or_miss', type=str,
                        help='Whats the array that you are trying to find', required=False)
    
    args = parser.parse_args()
    # filePattern = args.filePattern
    # logger.info('filePattern = {}'.format(filePattern))

    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    operations = args.operations
    logger.info('Operations = {}'.format(operations))
    if args.tile_size != None:
        Tile_Size = args.tile_size
    logger.info('Tile Size = {}'.format(Tile_Size))


    dilateby = args.dilate_by
    erodeby = args.erode_by
    hitormiss = args.hit_or_miss
    intkernel = args.all_kernel

    # openkernel = args.open_kernel
    # closekernel = args.close_kernel
    # morphkernel = args.morph_kernel
    # dilatekernel = args.dilate_kernel
    # erodekernel = args.erode_kernel
    # skeletonkernel = args.skeleton_kernel
    # areafilterkernel = args.areafilter_kernel
    # tophatkernel = args.tophat_kernel
    # blackhatkernel = args.blackhat_kernel

    if hitormiss != None:
        hitormiss = np.array(ast.literal_eval(hitormiss))
    hitormiss = np.array(hitormiss)

    if 'dilation' in operations:
        if dilateby == None:
            raise ValueError("Need to specify --dilateby integer value")
        else:
            logger.info('Dilating by {}'.format(dilateby))

    if 'erosion' in operations and erodeby == None:
        if erodeby == None:
            raise ValueError("Need to specify --erodeby integer value")
        else:
            logger.info('Eroding by {}'.format(erodeby))

    if 'hit_or_miss' in operations:
        if hitormiss.any() == None:
            raise ValueError("Need to specify --HOMarray integer value")
        else:
            logger.info('Array that we are locating: {}'.format(hitormiss))

    # A dictionary specifying the function that will be run based on user input. 
    dispatch = {
        'invertion': invert_binary,
        'opening': open_binary,
        'closing': close_binary,
        'morphological_gradient': morphgradient_binary,
        'dilation': dilate_binary,
        'erosion': erode_binary,
        'skeleton': skeleton_binary,
        'fill_holes': holefilling_binary,
        'hit_or_miss': hitormiss_binary,
        'filter_area': areafiltering_binary,
        'top_hat': tophat_binary,
        'black_hat': blackhat_binary
    }

    # Additional arguments for each function
    dict_n_args = {
        'dilation': dilateby,
        'erosion': erodeby,
        'hit_or_miss': hitormiss,
        'invertion': None,
        'opening': None,
        'closing': None,
        'morphological_gradient': None,
        'skeleton': None,
        'fill_holes': None,
        'filter_area': None,
        'top_hat': None,
        'black_hat': None
    }
    

    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir()]
    logger.info("Files in input directory: {}".format(inpDir_files))

      
    # Loop through files in inpDir image collection and process
    try:
        imagenum = 0
        for f in inpDir_files:

            # Load an image
            image = Path(inpDir).joinpath(f)

            # Read the image
            br = BioReader(str(image.absolute()))
            # br = BioReader(image.absolute(),max_workers=max([cpu_count()-1,2]))

            # Get the dimensions of the Image
            br_y, br_x, br_z = br.num_y(), br.num_x(), br.num_z()
            datatype = br.pixel_type()

            # Initialize Kernel
            kernel = np.ones((intkernel,intkernel), datatype)

            # Initialize Output
            newfile = Path(outDir).joinpath(f)
            bw = BioWriter(file_path=str(newfile), metadata=br.read_metadata())

            # Initialize the Python Generators to go through each "tile" of the image
            tsize = Tile_Size + (2*intkernel)
            readerator = br.iterate(tile_stride=[Tile_Size, Tile_Size],tile_size=[tsize, tsize], batch_size=1)
            writerator = bw.writerate(tile_size=[Tile_Size, Tile_Size], tile_stride=[Tile_Size, Tile_Size], batch_size=1)
            next(writerator)

            for images,indices in readerator:
                # Extra tiles do not need to be calculated.
                    # Indices should range from -intkernel < index value < Image_Dimension + intkernel
                if indices[0][0][0] == br_x - intkernel:
                    continue
                if indices[1][0][0] == br_y - intkernel:
                    continue

                logger.info(indices)

                # Images are (1, Tile_Size, Tile_Size, 1)
                # Need to convert to (Tile_Size, Tile_Size) to be able to do operation
                images = np.squeeze(images)

                # Initialize which function we are dispatching
                function = dispatch[operations]
                if callable(function):
                    trans_image = function(images, kernel=kernel, intk=intkernel, n=dict_n_args[operations])

                # The image needs to be converted back to (1, Tile_Size_Tile_Size, 1) to write it
                reshape_img = np.reshape(trans_image[intkernel:-intkernel,intkernel:-intkernel], (1, Tile_Size, Tile_Size, 1))
                
                # Send it to the Writerator
                writerator.send(reshape_img)

            # Close the image
            bw.close_image()

            """Use this part to help check if the Tiling was done correctly"""
            # imagecheck = np.squeeze(br.read_image())
            # imagecheck = function(imagecheck, kernel=kernel, intk=intkernel, n=dict_n_args[operations])
            # new = Path(outDir).joinpath('checkimage.ome.tif')
            # bwcheck = BioWriter(str(new), metadata=br.read_metadata())
            # bwcheck.write_image(np.reshape(imagecheck, (br_y, br_x, br_z, 1, 1)))
            # bwcheck.close_image()

    # Always close the JavaBridge
    finally:
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
           