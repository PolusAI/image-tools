import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import utils
import filepattern


if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)
    parser.add_argument('--imageNum', dest='image_num', type=str,
                        help='Image number, will be stored as a timeframe', required=True)
    parser.add_argument('--stackheight', dest='stack_height', type=int,
                        help='The Height of the Stack', required=True)
    parser.add_argument('--stackby', dest='stack_by', type=str,
                        help='Variable that the images get stacked by', required=True)
    parser.add_argument('--varsinstack', dest='vars_instack', type=str,
                        help='Variables that the stack shares', required=True)
    parser.add_argument('--valsinstack', dest='vals_instack', type=int, nargs='+',
                        help='Values of variables that the stack shares', required=True)
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    # image = args.image
    pyramid_type = args.pyramid_type
    image_num = args.image_num
    stackheight = args.stack_height
    stackby = args.stack_by
    varsinstack = args.vars_instack
    valsinstack = args.vals_instack
    imagepattern = args.image_pattern

    # Get images that are stacked together
    filesbuild = filepattern.parse_directory(input_dir, pattern=imagepattern, var_order=varsinstack)
    channels, channelvals = utils.recursivefiles(filesbuild[0], varsinstack, valsinstack, stackby, stackheight, pattern=imagepattern)
    image = channels[0]

     
    # Initialize the logger    
    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO)  

    logger.info("Starting to build...")
    logger.info("{} values in stack {}, respectively".format(varsinstack, valsinstack))
    logger.info("{} values in stack: {}".format(stackby, channelvals))
    logger.info("Images in stack: {}".format(channels))
    

    # Initialize the javabridge
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)


    # Make the output directory
    if pyramid_type == "Neuroglancer":
        out_dir = Path(output_dir).joinpath(image.name)
    elif pyramid_type == "DeepZoom":
        out_dir = Path(output_dir).joinpath('{}_files'.format(image_num))
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())
    
    # Create the BioReader object
    logger.info('Getting the BioReader...')
    bf = BioReader(str(image.absolute()))
    
    # Create the output path and info file
    if pyramid_type == "Neuroglancer":
        file_info = utils.neuroglancer_info_file(bf,out_dir,stackheight)
    elif pyramid_type == "DeepZoom":
        file_info = utils.dzi_file(bf,out_dir,image_num)
    else:
        ValueError("pyramid_type must be Neuroglancer or DeepZoom")
    logger.info("data_type: {}".format(file_info['data_type']))
    logger.info("num_channels: {}".format(file_info['num_channels']))
    logger.info("number of scales: {}".format(len(file_info['scales'])))
    logger.info("type: {}".format(file_info['type']))
    
    # Create the classes needed to generate a precomputed slice
    logger.info("Creating encoder and file writer...")
    if pyramid_type == "Neuroglancer":
        encoder = utils.NeuroglancerChunkEncoder(file_info)
        file_writer = utils.NeuroglancerWriter(out_dir)
    elif pyramid_type == "DeepZoom":
        encoder = utils.DeepZoomChunkEncoder(file_info)
        file_writer = utils.DeepZoomWriter(out_dir)
    
    # Create the stacked images
    if pyramid_type == "Neuroglancer":
        logger.info("Stack contains {} Levels (Stack's height)".format(stackheight))
        for i in range(0, stackheight):
            if i == 0:
                utils._get_higher_res(0, channelvals[i], bf,file_writer,encoder)
            else:
                bf = BioReader(str(channels[i].absolute()))
                utils._get_higher_res(0, channelvals[i], bf,file_writer,encoder)
            logger.info("Finished Level {} in Stack".format(channelvals[i]))
    
    logger.info("Finished precomputing. Closing the javabridge and exiting...")
    jutil.kill_vm()
