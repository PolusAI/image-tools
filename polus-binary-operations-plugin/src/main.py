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


def invert_binary(image, kernel=None, intk=None, n=None):
    invertedimg = np.zeros(image.shape,dtype=datatype)
    invertedimg = 255 - image
    return invertedimg

def dilate_binary(image,kernel=None, intk=None, n=None):
    dilatedimg = image
    image = image.astype('uint16')
    image[image == 255] = 1
    dilatedimg = cv2.dilate(image, kernel, iterations=n)
    dilatedimg[dilatedimg == 1] = 255
    dilatedimage = dilatedimg.astype('uint8')
    return dilatedimg

def erode_binary(image,kernel=None, intk=None, n=None):
    # erodedimg = img
    image[image == 255] = 1
    erodedimg = cv2.erode(image, kernel, iterations=n)
    erodedimg[erodedimg == 1] = 255
    return erodedimg

def open_binary(mage,kernel=None, intk=None, n=None):
    openimg = image
    image[image == 255] = 1
    openimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return openimg

def close_binary(image,kernel=None, intk=None, n=None):
    closeimg = image
    image[image == 255] = 1
    closeimg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closeimg[closeimg == 1] = 255
    return closeimg

def morphgradient_binary(image,kernel=None, intk=None, n=None):
    mg = image
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
    # sqimage = image.squeeze()
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
    parser.add_argument('--kernelsize', dest='all_kernel', type=int,
                        help='Kernel size that should be used for all operations', required=True)
    parser.add_argument('--openkernelsize', dest='open_kernel', type=int,
                        help='Specify kernel size for opening if different than variable, kernelsize', required=False)
    parser.add_argument('--closekernelsize', dest='close_kernel', type=int,
                        help='Specify kernel size for closing if different than variable, kernelsize', required=False)
    parser.add_argument('--morphkernelsize', dest='morph_kernel', type=int,
                        help='Specify kernel size for morphological gradient if different than variable, kernelsize', required=False)
    parser.add_argument('--dilatekernelsize', dest='dilate_kernel', type=int,
                        help='Specify kernel size for dilation if different than variable, kernelsize', required=False)
    parser.add_argument('--erodekernelsize', dest='erode_kernel', type=int,
                        help='Specify kernel size for erosion if different than variable, kernelsize', required=False)
    parser.add_argument('--skeletonkernelsize', dest='skeleton_kernel', type=int,
                        help='Specify kernel size for skeletonization if different than variable, kernelsize', required=False)
    parser.add_argument('--areafilteringkernelsize', dest='areafilter_kernel', type=int,
                        help='Specify kernel size for area filtering if different than variable, kernelsize', required=False)
    parser.add_argument('--tophatkernelsize', dest='tophat_kernel', type=int,
                        help='Specify kernel size for tophat if different than variable, kernelsize', required=False)
    parser.add_argument('--blackhatkernelsize', dest='blackhat_kernel', type=int,
                        help='Specify kernel size for blackhat if different than variable, kernelsize', required=False)
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
    intkernel = args.all_kernel

    openkernel = args.open_kernel
    closekernel = args.close_kernel
    morphkernel = args.morph_kernel
    dilatekernel = args.dilate_kernel
    erodekernel = args.erode_kernel
    skeletonkernel = args.skeleton_kernel
    areafilterkernel = args.areafilter_kernel
    tophatkernel = args.tophat_kernel
    blackhatkernel = args.blackhat_kernel


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
    imagenum = 0
    try:
        for f in inpDir_files:
            # Load an image
            br = BioReader(str(Path(inpDir).joinpath(f)))
            br_y, br_x, br_z = br.num_y(), br.num_x(), br.num_z()
            # image = np.squeeze(br.read_image())
            global image
            image = br.read_image()
            image = image.astype('uint8')
            

            datatype = image.dtype
            logger.info("Datatype: {}".format(datatype))
            
            logger.info("Shape of Image: {}".format(image.shape))
            kernel = np.ones((intkernel,intkernel), datatype)

        
            newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op1.ome.tiff')
            bw = BioWriter(str(newfile), metadata=br.read_metadata())
            bw = bw.write_image(image)
            logger.info("orginal: {}".format(newfile.name))

            image = image.squeeze()
            global out_image
            out_image = np.zeros(image.shape, datatype)

            # brcheck = BioReader(newfile)
            # brcheck = brcheck.read_image()

            i = 1
            for op in operations:
                function = dispatch[op]
                if callable(function):
                    # logger.info(np.unique(image))
                    image = function(image, kernel=kernel, intk=intkernel, n=dict_n_args[op])
                    newfile = Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff')
                    logger.info("{}: {}".format(op, newfile.name))
                    # logger.info(np.unique(image))
                    # logger.info(" ")
                    bw = BioWriter(str(newfile), metadata=br.read_metadata())
                    bw.write_image(np.reshape(image, (br_y, br_x, br_z, 1, 1)))
                else:
                    raise ValueError("Function is not callable")
                i = i + 1
            imagenum = imagenum + 1
        logger.info('Closing the javabridge...')
        jutil.kill_vm()

    except Exception as e:
        logger.info(e)
        logger.info('Closing the javabridge...')
        jutil.kill_vm()
            # i = 1
            # for item in operations:
            #     if i == 1:
            #         if item == 'invertion':
            #             out_image = invert_binary(image)
            #             # newfile = Path(outDir).joinpath('inverted_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile, out_image.dtype)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'opening':
            #             if openkernel == None:
            #                 out_image = open_binary(image.squeeze(),kernel)
            #             else:
            #                 openkernel = np.ones((openkernel,openkernel), datatype) 
            #                 out_image = open_binary(image.squeeze(),openkernel)
            #             # newfile = Path(outDir).joinpath('open_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile, out_image.dtype)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'closing':
            #             if closekernel == None:
            #                 out_image = close_binary(image.squeeze(),kernel)
            #             else:
            #                 closekernel = np.ones((closekernel,closekernel), datatype) 
            #                 out_image = close_binary(image.squeeze(),closekernel)
            #             # newfile = Path(outDir).joinpath('close_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile, out_image.dtype)
            #             bw = BioWriter(newfile, xmetadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == "morphological_gradient":
            #             if morphkernel == None:
            #                 out_image = morphgradient_binary(image.squeeze(),kernel)
            #             else:
            #                 morphkernel = np.ones((morphkernel,morphkernel), datatype) 
            #                 out_image = morphgradient_binary(image.squeeze(),morphkernel)
            #             # newfile = Path(outDir).joinpath('morphgradient_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile, out_image.dtype)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'dilation':
            #             if dilatekernel == None:
            #                 out_image = dilate_binary(image.squeeze(), kernel, dilateby)
            #             else:
            #                 dilatekernel = np.ones((dilatekernel,dilatekernel), datatype) 
            #                 out_image = dilate_binary(image.squeeze(),dilatekernel, dilateby)
            #             # out_image = dilate_binary(image.squeeze(), kernel, dilateby)
            #             # newfile = Path(outDir).joinpath('dilated_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile, out_image.dtype)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'erosion':
            #             if erodekernel == None:
            #                 out_image = erode_binary(image.squeeze(),kernel, erodeby)
            #             else:
            #                 erodekernel = np.ones((erodekernel,erodekernel), datatype) 
            #                 out_image = erode_binary(image.squeeze(),erodekernel, erodeby)
            #             # out_image = erode_binary(image.squeeze(), kernel,erodeby)
            #             # newfile = Path(outDir).joinpath('eroded_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'skeleton':
            #             if skeletonkernel == None:
            #                 out_image = skeleton_binary(image.squeeze(), kernel)
            #             else:
            #                 skeletonkernel = np.ones((skeletonkernel,skeletonkernel), datatype) 
            #                 out_image = skeleton_binary(image.squeeze(),skeletonkernel, erodeby)
            #             # newfile = Path(outDir).joinpath('skeleton_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'fill_holes':
            #             image = holefilling_binary(image, intkernel)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, image=out_image, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(image, (br_y, br_x, br_z, 1, 1)))
            #         elif item =='hit_or_miss':
            #             out_image = hitormiss_binary(image.squeeze(),hitormiss)
            #             # newfile = Path(outDir).joinpath('hitormiss_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br.num_y(), br.num_x(), br.num_z(), 1, 1)))
            #         elif item == 'filter_area':
            #             if areafilterkernel == None:
            #                 out_image = areafiltering_binary(image.squeeze(), intkernel)
            #             else:
            #                 out_image = areafiltering_binary(image.squeeze(), areafilterkernel)
            #             # newfile = Path(outDir).joinpath('areafiltered_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'top_hat':
            #             if tophatkernel == None:
            #                 out_image = tophat_binary(image.squeeze(), kernel)
            #             else:
            #                 tophatkernel = np.ones((tophatkernel,tophatkernel), datatype) 
            #                 out_image = tophat_binary(image.squeeze(),tophatkernel)
            #             # out_image = tophat_binary(image.squeeze(), kernel)
            #             # newfile = Path(outDir).joinpath('tophat_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'black_hat':
            #             if blackhatkernel == None:
            #                 out_image = blackhat_binary(image.squeeze(), kernel)
            #             else:
            #                 blackhatkernel = np.ones((blackhatkernel,blackhatkernel), datatype) 
            #                 out_image = blackhat_binary(image.squeeze(),blackhatkernel)
            #             # out_image = blackhat_binary(image.squeeze(), kernel)
            #             # newfile = Path(outDir).joinpath('blackhat_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         else:
            #             raise ValueError("Operation value is incorrect")
            #     else:
            #         if item == 'invertion':
            #             out_image = invert_binary(out_image)
            #             # newfile = Path(outDir).joinpath('inverted_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'opening':
            #             if openkernel == None:
            #                 out_image = open_binary(out_image.squeeze(),kernel)
            #             else:
            #                 openkernel = np.ones((openkernel,openkernel), datatype) 
            #                 out_image = open_binary(image.squeeze(),openkernel)
            #             # newfile = Path(outDir).joinpath('open_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'closing':
            #             if closekernel == None:
            #                 out_image = close_binary(out_image.squeeze(),kernel)
            #             else:
            #                 closekernel = np.ones((closekernel,closekernel), datatype) 
            #                 out_image = close_binary(image.squeeze(),closekernel)
            #             # newfile = Path(outDir).joinpath('close_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, xmetadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == "morphological_gradient":
            #             if morphkernel == None:
            #                 out_image = morphgradient_binary(out_image.squeeze(),kernel)
            #             else:
            #                 morphkernel = np.ones((morphkernel,morphkernel), datatype) 
            #                 out_image = morphgradient_binary(out_image.squeeze(),morphkernel)
            #             # newfile = Path(outDir).joinpath('morphgradient_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'dilation':
            #             # out_image = None
            #             # if dilatekernel == None:
            #             out_image = dilate_binary(out_image.squeeze(), kernel, dilateby)
            #             # logger.info(out_image.dtype)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #             # else:
            #             #     dilatekernel = np.ones((dilatekernel,dilatekernel), datatype) 
            #             #     out_image = dilate_binary(out_image.squeeze(),dilatekernel, dilateby)
            #             # # out_image = dilate_binary(image.squeeze(), kernel, dilateby)
            #             # # newfile = Path(outDir).joinpath('dilated_'+f)
            #             #     newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             #     logger.info(newfile, out_image.dtype)
            #             #     bw = BioWriter(newfile, metadata=br.read_metadata())
            #             #     bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'erosion':
            #             if erodekernel == None:
            #                 out_image = erode_binary(out_image.squeeze(),kernel, erodeby)
            #             else:
            #                 erodekernel = np.ones((erodekernel,erodekernel), datatype) 
            #                 out_image = erode_binary(out_image.squeeze(),erodekernel, erodeby)
            #             # out_image = erode_binary(image.squeeze(), kernel,erodeby)
            #             # newfile = Path(outDir).joinpath('eroded_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'skeleton':
            #             if skeletonkernel == None:
            #                 out_image = skeleton_binary(out_image.squeeze(), kernel)
            #             else:
            #                 skeletonkernel = np.ones((skeletonkernel,skeletonkernel), datatype) 
            #                 out_image = skeleton_binary(out_image.squeeze(),skeletonkernel, erodeby)
            #             # newfile = Path(outDir).joinpath('skeleton_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'fill_holes':
            #             out_image = holefilling_binary(out_image, intkernel)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, image=out_image, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item =='hit_or_miss':
            #             out_image = hitormiss_binary(out_image.squeeze(),hitormiss)
            #             # newfile = Path(outDir).joinpath('hitormiss_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br.num_y(), br.num_x(), br.num_z(), 1, 1)))
            #         elif item == 'filter_area':
            #             if areafilterkernel == None:
            #                 out_image = areafiltering_binary(out_image.squeeze(), intkernel)
            #             else:
            #                 out_image = areafiltering_binary(out_image.squeeze(), areafilterkernel)
            #             # newfile = Path(outDir).joinpath('areafiltered_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'top_hat':
            #             if tophatkernel == None:
            #                 out_image = tophat_binary(out_image.squeeze(), kernel)
            #             else:
            #                 tophatkernel = np.ones((tophatkernel,tophatkernel), datatype) 
            #                 out_image = tophat_binary(out_image.squeeze(),tophatkernel)
            #             # out_image = tophat_binary(image.squeeze(), kernel)
            #             # newfile = Path(outDir).joinpath('tophat_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         elif item == 'black_hat':
            #             if blackhatkernel == None:
            #                 out_image = blackhat_binary(out_image.squeeze(), kernel)
            #             else:
            #                 blackhatkernel = np.ones((blackhatkernel,blackhatkernel), datatype) 
            #                 out_image = blackhat_binary(out_image.squeeze(),blackhatkernel)
            #             # out_image = blackhat_binary(image.squeeze(), kernel)
            #             # newfile = Path(outDir).joinpath('blackhat_'+f)
            #             newfile = str(Path(outDir).joinpath('image' + str(imagenum) + '_op'+ str(i+1) + '.ome.tiff'))
            #             logger.info(newfile)
            #             bw = BioWriter(newfile, metadata=br.read_metadata())
            #             bw.write_image(np.reshape(out_image, (br_y, br_x, br_z, 1, 1)))
            #         else:
            #             raise ValueError("Operation value is incorrect")
            #     i = i + 1
            

        # Write the output
        # bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
        # bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z,1,1)))
    
    
        

# %%


# %%
