from multiprocessing import cpu_count
import logging, argparse, traceback
from bfio.bfio import BioReader, BioWriter

import utils
import os

import numpy as np

from neurogen import info as nginfo
from neurogen import volume as ngvol

if __name__=="__main__":
    try:
        # Setup the Argument parsing
        parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

        parser.add_argument('--inpDir', dest='input_dir', type=str,
                            help='Path to folder with CZI files', required=True)
        parser.add_argument('--outDir', dest='output_dir', type=str,
                            help='The output directory for ome.tif files', required=True)
        parser.add_argument('--imagetype', dest='image_type', type=str,
                            help='The type of image: image or segmentation', required=True)

        args = parser.parse_args()
        input_image = args.input_dir
        output_dir = args.output_dir
        imagetype = args.image_type
        image = os.path.basename(input_image)

        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("build_pyramid")
        logger.setLevel(logging.INFO) 

        logger.info("Starting to build...")

        # Create the BioReader object
        logger.info('Getting the BioReader...')
        
        bf = BioReader(input_image, max_workers=max([cpu_count()-1,2]))
        getimageshape = bf.shape
        logger.info("Image Shape {}".format(getimageshape))
        datatype = np.dtype(bf.dtype)
        chunk_size = [256,256,256]
        resolution = utils.get_resolution(phys_y=bf.physical_size_y, 
                                          phys_x=bf.physical_size_x, 
                                          phys_z=bf.physical_size_z)

        if imagetype == "segmentation":
            file_info = nginfo.info_segmentation(directory=output_dir,
                                                dtype=datatype,
                                                chunk_size = [256,256,256],
                                                size=getimageshape[:3],
                                                resolution=resolution)
            encodedvolume = ngvol.generate_recursive_chunked_representation(volume=bf,info=file_info, dtype=datatype, directory=output_dir)
        elif imagetype == "image":
            file_info = nginfo.info_image(directory=output_dir,
                                                dtype=datatype,
                                                chunk_size = [256,256,256],
                                                size=getimageshape[:3],
                                                resolution=resolution)
            encodedvolume = ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_dir, blurring_method='average')
        else:
            raise ValueError("Image Type was not properly specified")
        
        logger.info("Data Type: {}".format(file_info['data_type']))
        logger.info("Number of Channels: {}".format(file_info['num_channels']))
        logger.info("Number of Scales: {}".format(len(file_info['scales'])))
        logger.info("Image Type: {}".format(file_info['type']))

    except Exception as e:
        traceback.print_exc()


