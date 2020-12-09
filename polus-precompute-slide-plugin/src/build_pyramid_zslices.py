import logging, argparse
from bfio import BioReader, BioWriter, LOG4J, JARS
import javabridge
import requests
from pathlib import Path
import utils
import numpy as np
from multiprocessing import cpu_count


if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--image', dest='image', type=str,
                        help='The image to turn into a pyramid', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)
    parser.add_argument('--imageNum', dest='image_num', type=str,
                        help='Image number, will be stored as a timeframe', required=True)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='Image type: either image or segmentation', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    image = args.image
    pyramid_type = args.pyramid_type
    image_num = args.image_num
    imagetype = args.image_type

    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(image_num)
    logger.setLevel(logging.INFO)  

    logger.info("Starting to build Z Slices or Z Stack ...")
    
    # Make the output directory
    image = Path(input_dir).joinpath(image)
    if pyramid_type == "Neuroglancer":
        out_dir = Path(output_dir).joinpath(image.name)
    elif pyramid_type == "DeepZoom":
        out_dir = Path(output_dir).joinpath('{}_files'.format(image_num))
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())
    
    # Create the BioReader object
    logger.info('Getting the BioReader...')

    try:
        javabridge.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],
                        class_path=JARS,
                        run_headless=True)

        bf = BioReader(str(image.absolute()),backend = 'java', max_workers=max([cpu_count()-1,2]))
        depth = bf.Z
        logger.info("XYZCT ({}, {}, {}, {}, {}), respectively".format(bf.X, bf.Y, bf.Z, bf.C, bf.T))

        if depth == 1:
            raise ValueError("This is not a Z stack")

        # Images have height (rows), width (columns), and depth (z-slices) - NJS
        logger.info('Depth {}'.format(depth))

        # Create the output path and info file
        if pyramid_type == "Neuroglancer":
            file_info = utils.neuroglancer_info_file(bf,out_dir, depth, imagetype)
        elif pyramid_type == "DeepZoom":
            file_info = utils.dzi_file(bf,out_dir,image_num, depth, imagetype)
        else:
            ValueError("pyramid_type must be Neuroglancer or DeepZoom")
        logger.info("data_type: {}".format(file_info['data_type']))
        logger.info("num_channels: {}".format(file_info['num_channels']))
        logger.info("number of scales: {}".format(len(file_info['scales'])))
        logger.info("type: {}".format(file_info['type']))
        
        logger.info("Creating encoder and file writer...")
        if pyramid_type == "Neuroglancer":
            encoder = utils.NeuroglancerChunkEncoder(file_info)
            file_writer = utils.NeuroglancerWriter(out_dir)
        elif pyramid_type == "DeepZoom":
            encoder = utils.DeepZoomChunkEncoder(file_info)
            file_writer = utils.DeepZoomWriter(out_dir)

        ids = []

        # Go through all the slices in the stack
        for i in range(0, depth):
            utils._get_higher_res(0, i, bf,file_writer,encoder, imageType = imagetype, ids=ids, slices=[i,i+1])
            logger.info("Finished Level {}/{} in stack process".format(i+1, depth))
        logger.info("Finished precomputing.")

        if imagetype == "segmentation":
            out_seginfo = utils.segmentinfo(encoder,ids,out_dir)
            logger.info("Finished Segmentation Information File")
    finally:
        javabridge.kill_vm()