from bfio.bfio import BioReader, BioWriter
import numpy as np
import os
from multiprocessing import cpu_count
import logging

# x,y size of the 3d image chunk to be loaded into memory
tile_size = 1024 

# depth of the 3d image chunk
tile_size_z = 1024

def max_projection(inpDir, outDir):
    """ This function calculates the maximum intensity 
    projection along the z-direction(depth) for every 
    3d image in the input directory.

    Args:
        inpDir : path to the input directory
        outDir : path to the output directory
    """

    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("max_projection")
    logger.setLevel(logging.INFO)

    # iterate over the images in the input directory
    inpDir_files = os.listdir(inpDir)
    inpDir_files = [filename for filename in inpDir_files if filename.endswith('.ome.tif')]
    for image_name in inpDir_files:
        logger.info('---- Processing image: {} ----'.format(image_name))
        
        # initalize biowriter and bioreader
        with BioReader(os.path.join(inpDir, image_name)) as br, \
            BioWriter(os.path.join(outDir, image_name),metadata=br.metadata) as bw:

            # iterate along the x,y,z direction
            for x in range(0,br.X,tile_size):
                x_max = min([br.X,x+tile_size])

                for y in range(0,br.Y,tile_size):
                    y_max = min([br.Y,y+tile_size])
                    
                    for ind, z in enumerate(range(0,br.Z,tile_size_z)):
                        z_max = min([br.Z,z+tile_size_z])

                        logger.info('processing tile x: {}-{} y: {}-{} z: {}-{}'.format(x, x_max, y, y_max, z, z_max))
                        if ind == 0:
                            out_image = np.max(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)
                        else:
                            out_image = np.dstack((out_image, np.max(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)))

                    if out_image.shape[2] > 1:
                        out_image = np.max(out_image, axis=2)
                    
                    # write output
                    bw.Z = 1
                    bw[y:y_max,x:x_max,0:1,0,0] = out_image
                    
def min_projection(inpDir, outDir):
    """ This function calculates the minimum intensity 
    projection along the z-direction (depth)for every 
    3d image in the input directory.

    Args:
        inpDir : path to the input directory
        outDir : path to the output directory
    """
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("min_projection")
    logger.setLevel(logging.INFO)

    # iterate over the images in the input directory
    inpDir_files = os.listdir(inpDir)
    inpDir_files = [filename for filename in inpDir_files if filename.endswith('.ome.tif')]
    for image_name in inpDir_files:
        logger.info('---- Processing image: {} ----'.format(image_name))

        # initalize biowriter and bioreader
        with BioReader(os.path.join(inpDir, image_name)) as br, \
            BioWriter(os.path.join(outDir, image_name),metadata=br.metadata) as bw:

            # iterate along the x,y,z direction
            for x in range(0,br.X,tile_size):
                x_max = min([br.X,x+tile_size])

                for y in range(0,br.Y,tile_size):
                    y_max = min([br.Y,y+tile_size])
                    
                    for ind, z in enumerate(range(0,br.Z,tile_size_z)):
                        z_max = min([br.Z,z+tile_size_z])

                        logger.info('processing tile x: {}-{} y: {}-{} z: {}-{}'.format(x, x_max, y, y_max, z, z_max))
                        if ind == 0:
                            out_image = np.min(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)
                        else:
                            out_image = np.dstack((out_image, np.min(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)))

                    if out_image.shape[2] > 1:
                        out_image = np.min(out_image, axis=2)

                    # write output
                    bw.Z = 1
                    bw[y:y_max,x:x_max,0:1,0,0] = out_image

def mean_projection(inpDir, outDir):
    """ This function calculates the mean intensity 
    projection along the z-direction (depth)for every 
    3d image in the input directory.

    Args:
        inpDir : path to the input directory
        outDir : path to the output directory
    """
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("min_projection")
    logger.setLevel(logging.INFO)

    # iterate over the images in the input directory
    inpDir_files = os.listdir(inpDir)
    inpDir_files = [filename for filename in inpDir_files if filename.endswith('.ome.tif')]
    for image_name in inpDir_files:
        logger.info('---- Processing image: {} ----'.format(image_name))

        # initalize biowriter and bioreader
        with BioReader(os.path.join(inpDir, image_name)) as br, \
            BioWriter(os.path.join(outDir, image_name),metadata=br.metadata) as bw:

            # iterate along the x,y,z direction
            for x in range(0,br.X,tile_size):
                x_max = min([br.X,x+tile_size])

                for y in range(0,br.Y,tile_size):
                    y_max = min([br.Y,y+tile_size])
                    
                    for ind, z in enumerate(range(0,br.Z,tile_size_z)):
                        z_max = min([br.Z,z+tile_size_z])

                        logger.info('processing tile x: {}-{} y: {}-{} z: {}-{}'.format(x, x_max, y, y_max, z, z_max))
                        if ind == 0:
                            out_image = np.sum(br[y:y_max,x:x_max,z:z_max,0,0] / 1024, axis=2, dtype = np.float32)
                        else:
                            out_image = np.dstack((out_image, np.sum(br[y:y_max,x:x_max,z:z_max,0,0] / 1024, axis=2, dtype=np.float32)))

                    if out_image.shape[2] > 1:
                        out_image = np.sum(out_image, axis=2)/ br.Z 
                        out_image = np.array(out_image * 1024, br.dtype)
                    else:
                        out_image = out_image / br.Z 
                        out_image = np.array(out_image * 1024, br.dtype)

                    # write output
                    bw.Z = 1
                    bw[y:y_max,x:x_max,0:1,0,0] = out_image


    