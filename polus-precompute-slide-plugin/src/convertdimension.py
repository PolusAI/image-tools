import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import os
import numpy as np
from bfio.bfio import BioReader, BioWriter
import utils
from multiprocessing import cpu_count


def main():

    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format("image"),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)  

    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Generate a precomputed slice for Polus Volume Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)

    
    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))

    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir()]

    for image in images:
        logger.info(image)
        bf = BioReader(str(image.absolute()),max_workers=max([cpu_count()-1,2]))
        im = bf.read_image()
        xval = bf.num_x()
        yval = bf.num_y()
        zval = bf.num_z()
        cval = bf.num_c()
        tval = bf.num_t()
        logger.info("NUM_VAR Values ({}, {}, {}, {}, {})".format(bf.num_x(), bf.num_y(), bf.num_z(), bf.num_c(), bf.num_t()))
        if zval == 1 and (cval > 1 or tval > 1):
            if tval > 1:
                zval = bf.num_t()
                tval = bf.num_z()
            elif cval > 1:
                zval = bf.num_c()
                cval = bf.num_z()
            else:
                raise ValueError("Something is wrong in this logic")
        logger.info("Reordering ({}, {}, {}, {}, {})".format(xval, yval, zval, cval, tval))
        logger.info("IM SHAPE {}".format(im.shape))
        # logger.info("IMAGE SHAPE {}".format(im.shape))
        # logger.info(type(image1))
        # if (image1 == image2).all():
        #     logger.info("SAME?")
        # else:
        #     logger.info("NOT SAME, good :)")
        # logger.info("Image Read Shape: {}".format(image.shape))
        # # image = np.reshape(image, (xval, yval, zval, cval, tval))
        # logger.info("(CHECK IT OUT: {}, {}, {}, {}, {})".format(bf.num_x(), bf.num_y(), bf.num_z(), bf.num_c(), bf.num_t()))
        

    


    # if imagepattern or image type specified then images get stacked by the stack_by variables

if __name__ == "__main__":
    main()
