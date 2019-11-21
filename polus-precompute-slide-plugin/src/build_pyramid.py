import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
from utils import ChunkEncoder, SlideWriter, CHUNK_SIZE, bfio_metadata_to_slide_info, _get_higher_res

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)        

if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--image', dest='image', type=str,
                        help='The image to turn into a pyramid', required=True)


    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    image = args.image
    logger.info(" - {} - Starting to build...".format(image))
    
    logger.info(' - {} - Initializing the javabridge...'.format(image))
    jutil.start_vm(class_path=bioformats.JARS)

    image = Path(input_dir).joinpath(image)
    out_dir = Path(output_dir).joinpath(image.name)
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())
    
    # Create the BioReader object
    logger.info(' - {} - Getting the BioReader...'.format(image.name))
    bf = BioReader(str(image.absolute()))
    
    # Create the output path and info file
    file_info = bfio_metadata_to_slide_info(bf,out_dir)
    logger.info(" - {} - data_type: {}".format(image.name,file_info['data_type']))
    logger.info(" - {} - num_channels: {}".format(image.name,file_info['num_channels']))
    logger.info(" - {} - number of scales: {}".format(image.name,len(file_info['scales'])))
    logger.info(" - {} - type: {}".format(image.name,file_info['type']))
    
    # Create the classes needed to generate a precomputed slice
    logger.info(" - {} - Creating encoder...".format(image.name))
    encoder = ChunkEncoder(file_info)
    logger.info(" - {} - Creating file_writer...".format(image.name))
    file_writer = SlideWriter(out_dir)

    _get_higher_res(0,bf,file_writer,encoder)
    
    logger.info(" - {} - Finished precomputing. Closing the javabridge and exiting...".format(image.name))
    jutil.kill_vm()
    