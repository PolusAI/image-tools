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

Tile_Size = 256

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
    try:
        imagenum = 0
        for f in inpDir_files:
            # Load an image
            
            image = Path(inpDir).joinpath(f)

            br = BioReader(str(image.absolute()))
            # br = BioReader(image.absolute(),max_workers=max([cpu_count()-1,2]))

            br_y, br_x, br_z = br.num_y(), br.num_x(), br.num_z()
            datatype = br.pixel_type()
            kernel = np.ones((intkernel,intkernel), datatype)

            newfile = Path(outDir).joinpath(f)
            bw = BioWriter(file_path=str(newfile), metadata=br.read_metadata())

            strides = Tile_Size - (2*intkernel)
            # batch_size = br.maximum_batch_size(tile_stride=[strides, strides],tile_size=[Tile_Size,Tile_Size])
            # batch_size = br.maximum_batch_size(tile_size=[Tile_Size + 2*(intkernel),Tile_Size + 2*(intkernel)],tile_stride=[Tile_Size,Tile_Size])
            readerator = br.iterate(tile_stride=[Tile_Size, Tile_Size],tile_size=[Tile_Size + (2*intkernel), Tile_Size + (2*intkernel)], batch_size=1)
            writerator = bw.writerate(tile_size=[Tile_Size, Tile_Size], tile_stride=[Tile_Size, Tile_Size], batch_size=1)
            next(writerator)

            for images,indices in readerator:
                if indices[0][0][0] == br_x - intkernel:
                    continue
                if indices[1][0][0] == br_y - intkernel:
                    continue
                logger.info(indices)
                # images = images[0,5:-5,5:-5,0]
                images = images[0, :, :, 0]

                function = dispatch[operations]
                if callable(function):
                    trans_image = function(images, kernel=kernel, intk=intkernel, n=dict_n_args[operations])
                reshape_img = np.reshape(trans_image[intkernel:-intkernel,intkernel:-intkernel], (1, Tile_Size, Tile_Size, 1))
                writerator.send(reshape_img)
            bw.close_image()

            # def left_trim(im):
            #     im = im[:-intkernel,intkernel:-intkernel]
            #     return im

            # def right_trim(im):
            #     im = im[intkernel:,intkernel:-intkernel]
            #     return im

            # def up_trim(im):
            #     im = im[intkernel:-intkernel,:-intkernel]
            #     return im

            # def down_trim(im):
            #     im = im[intkernel:-intkernel, intkernel:]
            #     return im

            # for images,indices in readerator:
            #     images = images[0, :, :, 0]
            #     # logger.info("editing {}".format(indices[1][0][0]))
            #     if indices[0][0][0] == br_x - intkernel:
            #         continue
            #     if indices[1][0][0] == br_y - intkernel:
            #         continue

            #     # logger.info("before: {}, {} -- {} ".format(indices[0], indices[1], images.shape))
            #     x1, x2, y1, y2 = [0, 0, 0, 0]
            #     edges = []
            #     if indices[0][0][0] < 0:
            #         start = abs(indices[0][0][0])
            #         indices[0][0][0] = 0
            #         images = images[start:,:]
            #         edges.append("left")
            #         x1 = start
            #     if indices[1][0][0] < 0:
            #         start = abs(indices[1][0][0])
            #         indices[1][0][0] = 0
            #         images = images[:,start:]
            #         edges.append("up")
            #         y1 = start

            #     if indices[0][0][1] > 1024:
            #         end = 1024 - indices[0][0][1]
            #         indices[0][0][1] = 1024
            #         images = images[:end, :]
            #         edges.append("right")
            #         x2 = end
            #     if indices[1][0][1] > 1024:
            #         end = 1024 - indices[1][0][1]
            #         indices[1][0][1] = 1024
            #         images = images[:, :end]
            #         edges.append("down")
            #         y2 = end
                
            #     logger.info("indices: {}, {} -- {}".format(indices[0], indices[1], images.shape))
            #     logger.info("Edge? {}".format(edges))
            #     newindex = [(indices[0][0][0], indices[0][0][1]), (indices[1][0][0], indices[1][0][1])]
            #     logger.info(newindex)
            #     function = dispatch[operations]
            #     if callable(function):
            #         trans_image = function(images, kernel=kernel, intk=intkernel, n=dict_n_args[operations])
                
            #     if len(edges) == 2:
            #         # upper left
            #         if newindex[0][0] == 0 and newindex[1][0] == 0:
            #             trans_image = trans_image[:-intkernel, :-intkernel]
            #         # lower left
            #         if newindex[0][0] == 0 and newindex[1][1] == 1024:
            #             trans_image = trans_image[:-intkernel, intkernel:]
            #         # upper right
            #         if newindex[0][1] == 1024 and newindex[1][0] == 0:
            #             trans_image = trans_image[intkernel:, :-intkernel]
            #         # lower right
            #         if newindex[0][1] == 1024 and newindex[1][1] == 1024:
            #             trans_image = trans_image[intkernel:, intkernel:]
            #         # else:
            #         #     raise ValueError("Not a corner")
            #     elif len(edges) == 1:
            #         if edges[0] == "left":
            #             trans_image = left_trim(trans_image)
            #         elif edges[0] == "right":
            #             trans_image = right_trim(trans_image)
            #         elif edges[0] == "up":
            #             trans_image = up_trim(trans_image)
            #         elif edges[0] == "down":
            #             trans_image = down_trim(trans_image)
            #         else:
            #             raise ValueError("Not an edge")
            #     else:
            #         trans_image = trans_image[intkernel:-intkernel, intkernel:-intkernel]
            #     logger.info(trans_image.shape)

            #     # for item in edges:
            #     #     if item == "left":
            #     #         shrink = left_trim(trans_image)
            #     #     elif item == "right":
            #     #         shrink = right_trim(trans_image)
            #     #     elif item == "up":
            #     #         shrink = up_trim(trans_image)
            #     #     elif item == "down":
            #     #         shrink = down_trim(trans_image)
            #     #     else:
            #     #         logger.info("NOT AN EDGE")

            #     # trans_image = trans_image[0:256, 0:256]
            #     # logger.info("reshaped images {} ".format(trans_image.shape))
            #     reshaped_image = np.reshape(trans_image, (1,256, 256, 1)).astype('uint16')
            #     writerator.send(reshaped_image)
            #     # logger.info("difference {}, {}".format(X_afterindex - X_beforeindex, Y_afterindex-Y_beforeindex))
            #     logger.info("")
            # bw.close_image()
                # writerator.send(images)

            # x = Tile_Size
            # xsub = 0
            
            # while (xsub + x) <= br_x:
            #     writex = [xsub, xsub+x]
            #     lowerx = xsub - intkernel
            #     upperx = xsub+x+intkernel
            #     if lowerx < 0:
            #         lowerx = 0
            #     if upperx > br_x:
            #         upperx = br_x
            #     xvals = [lowerx, upperx]
            #     xsub = xsub + x
            #     y = Tile_Size
            #     ysub = 0
            #     while (ysub + y) <= br_y:
            #         writey = [ysub, ysub+y]
            #         lowery = ysub - intkernel
            #         uppery = ysub+y+intkernel
            #         if lowery < 0:
            #             lowery = 0
            #         if uppery > br_y:
            #             uppery = br_y
            #         yvals = [lowery, uppery]
            #         ysub = ysub + y

            #         logger.info("Reading {},{} --> Writing {},{}".format(xvals, yvals, writex, writey))

            #         image = br.read_image(X=xvals, Y=yvals)
            #         # image = imagetiling(br,xvals,yvals)
            #         # logger.info("Image Shape {}".format(image.shape))
            #         logger.info("Recording {}, {}".format(np.subtract(writex,xvals), np.subtract(writey,yvals)))

            #         difx = np.subtract(writex, xvals)
            #         dify = np.subtract(writey, yvals)

            #         sqimage = np.squeeze(image)

            #         function = dispatch[operations]

            #         if callable(function):
            #             transformed_image = function(sqimage, kernel=kernel, intk=intkernel, n=dict_n_args[operations])
            #         logger.info("{}:{}, {}:{}".format(dify[0], dify[1], difx[0], difx[1]))
            #         logger.info("before transformed image shape {}".format(transformed_image.shape))

            #         # Edge cases 
            #         if dify[1] == 0 and difx[1] != 0:
            #             trans_image = transformed_image[dify[0]:, difx[0]:difx[1]]
            #         elif difx[1] == 0 and dify[1] != 0:
            #             trans_image = transformed_image[dify[0]:dify[1], difx[0]:]
            #         elif difx[1] == 0 and dify[1] == 0:
            #             trans_image = transformed_image[dify[0]:, difx[0]:]
            #         else:
            #             trans_image = transformed_image[dify[0]:dify[1], difx[0]:difx[1]]

            #         logger.info("after transformed image shape {}".format(trans_image.shape))
            #         logger.info("Writing at {},{}".format(writey[0], writex[0]))
            #         # for item in trans_image:
            #         #     for item2 in item:
            #         #         print(item2, type(item2))
                    
                    
            #         writerator.send(trans_image[:][:])
            #         # reshaped_image = np.reshape(trans_image, (writey[1]-writey[0], writex[1]-writex[0], 1, 1, 1)).astype('uint16')
            #         logger.info("Reshaped Image Shape {}".format(trans_image.shape))
                    
            #         if writex[0] == 0 and writey[0]!= 0:
            #             bw._buffer_supertile(X=write[0],writey[0])
            #             bw.write_image(image=reshaped_image, X=[writex[0]], Y=[writey[0]])
            #         elif writex[0] != 0 and writey[0]== 0:
            #             bw.write_image(image=reshaped_image, X=[writex[0]], Y=[writey[0]])
            #         elif writex[0] == 0 and writey[0]== 0:
            #             bw.write_image(image=reshaped_image, X=[writex[0]], Y=[writey[0]])
            #         else:
            #             bw.write_image(image=reshaped_image, X=[writex[0]], Y=[writey[0]])
            # bw.close_image()
            # # logger.info("CHECK {}".format(bw._xyzct))

            image3 = np.squeeze(br.read_image())
            function = dispatch[operations]
            if callable(function):
                image3 = function(image3, kernel=kernel, intk=intkernel, n=dict_n_args[operations])
                new = Path(outDir).joinpath('checkimage.ome.tif')
                logger.info("{}: {}, {}".format(operations, new.name, type(image3[0][0])))
                bw2 = BioWriter(str(new), metadata=br.read_metadata())
                bw2.write_image(np.reshape(image3, (br_y, br_x, br_z, 1, 1)))
            bw2.close_image()


        # logger.info("SHAPE OF Y X Z: {}, {}, {}".format(br_y, br_x, br_z))
        # image = np.squeeze(br.read_image())
        # global image

        
        
        
        # while xsize+intkernel <= br_x:
        #     xsub = xsub + xsize+intkernel
        # image = br.read_image(X=[0,259], Y=[0,259])
        
        # image = image.astype('uint8')
        # ok_image = np.reshape(image, (1024, 1024, 1, 1, 1)).astype('uint8')

        

        # logger.info(image)

        
        
        # logger.info("orginal: {}".format(newfile.name))
        # # bw.close_image()

        # image = image.squeeze()
        # global out_image
        # out_image = np.zeros(image.shape, datatype)

        # brcheck = BioReader(str(newfile))
        # brcheck = brcheck.read_metadata()

        # logger.info(dict_intk)

        # i = 1
        # # for op in operations:
        
        
        # if image == image3:
        #     logger.info("iTS ALL GOOD")
        # else:
        #     logger.info("NO")
        # # else:
        #     raise ValueError("Function is not callable")
        # i = i + 1

    finally:
        logger.info("DONE")
        # logger.info(e)
        # logger.info('Closing the javabridge...')
        # jutil.kill_vm()
           