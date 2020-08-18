import logging, argparse
# import bioformats - NJS
# import javabridge as jutil - NJS
from bfio.bfio import BioReader, BioWriter
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

    # Try/Catch/Finally not needed with bfio Python backend
    # try:
    # Initialize the logger    
    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(image_num)
    logger.setLevel(logging.INFO)  

    logger.info("Starting to build Z Slices...")
    
    # Initialize the javabridge
    # Removed because new Python backend doesn't require it - NJS
    # logger.info('Initializing the javabridge...')
    # log_config = Path(__file__).parent.joinpath("log4j.properties")
    # jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

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
    
    # Since this code block is already surround by try/catch, this is redundant - NJS
    # try:
    #     bf = BioReader(str(image.absolute()))
    # except:
    #     jutil.kill_vm()
    #     exit
    bf = BioReader(str(image.absolute()),max_workers=max([cpu_count()-1,2])) # NJS
    
    # This is not scalable, since it requires loading an entire image - NJS
    # imageread = bf.read_image()
    # height = imageread.shape[2]
    
    # This is an efficient method to getting an image dimension
    depth = bf.num_z() # NJS

    # Images have height (rows), width (columns), and depth (z-slices) - NJS
    logger.info('Depth {}'.format(depth))

    # Create the output path and info file
    if pyramid_type == "Neuroglancer":
        file_info = utils.neuroglancer_info_file(bf,out_dir, depth, imagetype)
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
        if imagetype == "segmentation":
            # This should be replaced with something more scalable
            imageread = bf.read_image()
            mesh = np.transpose(imageread, (0,2,1,3,4))
            ids = [int(i) for i in np.unique(mesh[:])]
            out_seginfo = utils.segmentinfo(encoder,ids,out_dir)
    elif pyramid_type == "DeepZoom":
        encoder = utils.DeepZoomChunkEncoder(file_info)
        file_writer = utils.DeepZoomWriter(out_dir)

    # Go through all the slices in the stack
    for i in range(0, depth):
        utils._get_higher_res(0, i, bf,file_writer,encoder, imageType = imagetype, slices=[i,i+1])
        logger.info("Finished Level {}/{} in stack process".format(i, depth))
    
    logger.info("Finished precomputing.")
        
        # This should be removed, since the finally block will cause this to close - NJS
        # jutil.kill_vm()
        
    # This block isn't necessary if the finally block exists, unless you want to catch a specific exception - NJS
    # except Exception as e:
    #     jutil.kill_vm()
    #     raise
    
    # This will always get executed regardless of what happens above - NJS
    # finally:
    #     jutil.kill_vm()
