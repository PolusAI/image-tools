from multiprocessing import cpu_count
import logging, argparse
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import utils
from filepattern import FilePattern as fp

if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--imageNum', dest='image_num', type=str,
                        help='Image number, will be stored as a timeframe', required=True)
    parser.add_argument('--stackby', dest='stack_by', type=str,
                        help='Variable that the images get stacked by', required=True)
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)
    parser.add_argument('--image', dest='image', type=str,
                        help='The image to turn into a pyramid', required=True)
    parser.add_argument('--imagetype', dest='image_type', type=str,
                        help='The type of image: image or segmentation', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    image_num = args.image_num 
    stackby = args.stack_by
    imagepattern = args.image_pattern
    image = args.image
    imagetype = args.image_type

    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image_num),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO) 

    logger.info("Starting to build...")

    # Create the BioReader object
    logger.info('Getting the BioReader...')
    logger.info(image)
    bf = BioReader(Path(input_dir).joinpath(image),max_workers=max([cpu_count()-1,2]))
    getimageshape = bf.shape
    stackheight = bf.z

    # Make the output directory
    out_dir = Path(output_dir).joinpath(image)
    out_dir.mkdir()
    out_dir = str(out_dir.absolute())

    # Create the output path and info file
    file_info = utils.neuroglancer_info_file(bf,out_dir,stackheight, imagetype)

    logger.info("data_type: {}".format(file_info['data_type']))
    logger.info("num_channels: {}".format(file_info['num_channels']))
    logger.info("number of scales: {}".format(len(file_info['scales'])))
    logger.info("type: {}".format(file_info['type']))
    
    # Create the classes needed to generate a precomputed slice
    logger.info("Creating encoder and file writer...")
    encoder = utils.NeuroglancerChunkEncoder(file_info)
    file_writer = utils.NeuroglancerWriter(out_dir)
    
    logger.info("Getting Higher Res...")
    
    # Create the stacked images
    utils._get_higher_res(0, bf,file_writer,encoder)