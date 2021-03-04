import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS
import bioformats
import javabridge as jutil

import numpy as np
import os, shutil
from pathlib import Path
import tempfile

from concurrent.futures import ThreadPoolExecutor
import traceback

from neurogen import info as nginfo

import utils

if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_image = args.output_dir
    image = os.path.basename(output_image)
    
    try:
        # Initialize the logger    
        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("build_pyramid")
        logger.setLevel(logging.INFO) 

        logger.info("Starting to build...")
        # logger.info("Values of xyz in stack are {}".format(valsinstack))

        # Initialize the javabridge
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

        # Create the BioReader object
        logger.info('Getting the BioReader...')
        logger.info(image)

        pathbf = os.path.join(input_dir, image)
        bf = BioReader(pathbf)
        bfshape = bf.shape
        chunk_size = [256, 256, 256]
        bit_depth = 16
        resolution = utils.get_resolution(phys_y=bf.physical_size_y, 
                                    phys_x=bf.physical_size_x, 
                                    phys_z=bf.physical_size_z)

        ysplits = list(np.arange(0, bfshape[0], chunk_size[0]))
        ysplits.append(bfshape[0])
        xsplits = list(np.arange(0, bfshape[1], chunk_size[1]))
        xsplits.append(bfshape[1])
        zsplits = list(np.arange(0, bfshape[2], chunk_size[2]))
        zsplits.append(bfshape[2])

        all_identities = np.array([])
        totalbytes = {}
        temp_dir = os.path.join(output_image, "tempdir")

        with tempfile.TemporaryDirectory() as temp_dir:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            for y in range(len(ysplits)-1):
                for x in range(len(xsplits)-1):
                    for z in range(len(zsplits)-1):
                        start_y, end_y = (ysplits[y], ysplits[y+1])
                        start_x, end_x = (xsplits[x], xsplits[x+1])
                        start_z, end_z = (zsplits[z], zsplits[z+1])
                        

                        volume = bf[start_y:end_y,start_x:end_x,start_z:end_z]
                        volume = volume.reshape(volume.shape[:3])
                        logger.info("Loaded subvolume (YXZ) {}-{}__{}-{}__{}-{}".format(start_y, end_y,
                                                                                        start_x, end_x,
                                                                                        start_z, end_z))
                        ids = np.unique(volume)
                        if (ids == [0]).all():
                            continue
                        else:
                            ids = np.delete(ids, np.where(ids==0))
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                executor.submit(utils.create_plyfiles(subvolume = volume,
                                                                ids=ids,
                                                                temp_dir=temp_dir,
                                                                start_y=start_y,
                                                                start_x=start_x,
                                                                start_z=start_z,
                                                                totalbytes=totalbytes))
                                all_identities = np.append(all_identities, ids)

            executor.shutdown(wait=True)

            all_identities = np.unique(all_identities).astype('int')
            file_info = nginfo.info_mesh(directory=output_image,
                                        chunk_size=chunk_size,
                                        size=bf.shape[:3],
                                        dtype=np.dtype(bf.dtype).name,
                                        ids=all_identities,
                                        resolution=resolution,
                                        segmentation_subdirectory="segment_properties",
                                        bit_depth=bit_depth)

            logger.info("Image Shape {}".format(bf.shape))
            numscales = len(file_info['scales'])
            logger.info("Number of scales: {}".format(numscales))
            logger.info("Data type: {}".format(file_info['data_type']))
            logger.info("Number of channels: {}".format(file_info['num_channels']))
            logger.info("Number of scales: {}".format(numscales))
            logger.info("Resolution {}".format(resolution))
            logger.info("Image type: {}".format(file_info['type']))

            logger.info("Labelled Image Values: {}".format(all_identities))

            with ThreadPoolExecutor(max_workers=4) as executor:
                futuresvariable = [executor.submit(utils.concatenate_and_generate_meshes, 
                                                ide, temp_dir, output_image, bit_depth, chunk_size) 
                                                for ide in all_identities]
            executor.shutdown(wait=True)
        

    except Exception as e:
        traceback.print_exc()
    finally:
        jutil.kill_vm()
