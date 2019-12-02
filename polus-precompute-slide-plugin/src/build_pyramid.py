import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
from utils import ChunkEncoder, SlideWriter, CHUNK_SIZE, bfio_metadata_to_slide_info, _get_higher_res      

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

    # Initialize the logger    
    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO)  

    logger.info("Starting to build...")
    
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    image = Path(input_dir).joinpath(image)
    out_dir = Path(output_dir).joinpath(image.name)
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())
    
    # Create the BioReader object
    logger.info('Getting the BioReader...')
    bf = BioReader(str(image.absolute()))
    
    # Create the output path and info file
    file_info = bfio_metadata_to_slide_info(bf,out_dir)
    logger.info("data_type: {}".format(file_info['data_type']))
    logger.info("num_channels: {}".format(file_info['num_channels']))
    logger.info("number of scales: {}".format(len(file_info['scales'])))
    logger.info("type: {}".format(file_info['type']))
    
    # Create the classes needed to generate a precomputed slice
    logger.info("Creating encoder...")
    encoder = ChunkEncoder(file_info)
    logger.info("Creating file_writer...")
    file_writer = SlideWriter(out_dir)

    _get_higher_res(0,bf,file_writer,encoder)
    
    logger.info("Finished precomputing. Closing the javabridge and exiting...")
    jutil.kill_vm()
    