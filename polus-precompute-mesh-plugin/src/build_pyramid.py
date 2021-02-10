# from multiprocessing import cpu_count
import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS
import bioformats
import javabridge as jutil
from pathlib import Path
import utils
import filepattern
from filepattern import FilePattern as fp
import itertools
import numpy as np
import os
import traceback
import json
from os import listdir
from os.path import isfile, join
import trimesh
import fulloctree_multires as decomposition
import ast
from concurrent.futures import ThreadPoolExecutor


bit_depth = 16
Tile_Size = (256,256,256)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--imageNum', dest='image_num', type=str,
                        help='Image number, will be stored as a timeframe', required=True)
    parser.add_argument('--image', dest='image', type=str,
                        help='The image to turn into a pyramid', required=True)
    parser.add_argument('--imagetype', dest='image_type', type=str,
                        help='The type of image: image or segmentation', required=True)
    parser.add_argument('--meshes', dest='meshes', type=str2bool, nargs='?',const=True,
                        default=False,help='True or False')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_num = args.image_num 
    image = args.image
    imagetype = args.image_type
    boolmesh = args.meshes

    try:
        # Initialize the logger    
        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image_num),
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
        pathbf = str((Path(input_dir).joinpath(image)))
        bf = BioReader(pathbf)
        getimageshape = (bf.x, bf.y, bf.z, bf.c, bf.t)
        stackheight = bf.z
        logger.info("Image Shape {}".format(getimageshape))

        # Make the output directory
        out_dir = Path(output_dir).joinpath(image)
        out_dir.mkdir()
        out_dir = str(out_dir.absolute())

        # Create the output path and info file
        file_info = utils.neuroglancer_info_file(bfio_reader=bf,outPath=out_dir,stackheight=stackheight,imagetype=imagetype, meshes=boolmesh)

        numscales = len(file_info['scales'])
        logger.info("data_type: {}".format(file_info['data_type']))
        logger.info("num_channels: {}".format(file_info['num_channels']))
        logger.info("number of scales: {}".format(numscales))
        logger.info("type: {}".format(file_info['type']))

        # Create the classes needed to generate a precomputed slice
        logger.info("Creating encoder and file writer...")
        encoder = utils.NeuroglancerChunkEncoder(file_info)
        file_writer = utils.NeuroglancerWriter(out_dir)

        ids = []
        
        # Create the stacked/pyramid images, will also create meshes if True
        outDir_mesh = Path(out_dir).joinpath("meshdir") # Initialize directory with all the mesh information
        utils._get_higher_res(S=0, bfio_reader=bf,slide_writer=file_writer,encoder=encoder,ids=ids,meshes=boolmesh, imagetype = imagetype, outDir_mesh=outDir_mesh)
        logger.info("Finished precomputing ")

        # Neuroglancer requires additional documentation to the segmentation and meshes
        if imagetype == "segmentation":
            utils.infodir_files(encoder,ids,out_dir)
            logger.info(ids)
            logger.info("Finished Segmentation Information File")

        # Polygon meshes must be converted to Draco meshes
        if boolmesh == True:
            utils.meshdir_files(outDir_mesh, encoder)
            temp_dir = str(outDir_mesh.joinpath('temp_drc'))
            chunkfiles = [f for f in listdir(temp_dir) if isfile(join(temp_dir, f))]

            # Each mesh segment is concatenated with its own thread
            ids.remove(0)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futuresvariable = [executor.submit(utils.progressive_meshes, ide, chunkfiles, temp_dir, out_dir, bit_depth) for ide in ids]
                
    except Exception as e:
        traceback.print_exc()
    finally:
        jutil.kill_vm()
        
