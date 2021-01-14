from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing, sys
import numpy as np
from pathlib import Path

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate volumetric intensity projections')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--projectionType', dest='projectionType', type=str,
                        help='Type of volumetric intensity projection', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    projectionType = args.projectionType
    logger.info('projectionType = {}'.format(projectionType))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
        
        
        # Loop through files in inpDir image collection and process
        for i,f in enumerate(inpDir_files):
            # Load an image
            br = BioReader(Path(inpDir).joinpath(f))
            image = np.squeeze(br.read_image())

            # initialize the output
            out_image = np.zeros(image.shape,dtype=br._pix['type'])

            """ Do some math and science - you should replace this """
            logger.info('Processing image ({}/{}): {}'.format(i,len(inpDir_files),f))
            out_image = awesome_math_and_science_function(image)

            # Write the output
            bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
            bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z(),1,1)))
        
    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the javabridge')
        jutil.kill_vm()
        
        # Exit the program
        sys.exit()