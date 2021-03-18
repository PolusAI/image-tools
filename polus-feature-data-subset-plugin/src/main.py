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
    parser = argparse.ArgumentParser(prog='main', description='Subset data using a given feature')
    
    # Input arguments
    parser.add_argument('--csvDir', dest='csvDir', type=str,
                        help='CSV collection containing features', required=True)
    parser.add_argument('--delay', dest='delay', type=str,
                        help='Number of images to capture outside the cutoff', required=True)
    parser.add_argument('--feature', dest='feature', type=str,
                        help='Feature to use to subset data', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--groupVar', dest='groupVar', type=str,
                        help='variables to group by in a section', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--percentile', dest='percentile', type=str,
                        help='Percentile to remove', required=True)
    parser.add_argument('--removeDirection', dest='removeDirection', type=str,
                        help='remove direction above or below percentile', required=True)
    parser.add_argument('--sectionVar', dest='sectionVar', type=str,
                        help='variables to divide larger sections', required=True)
    parser.add_argument('--writeOutput', dest='writeOutput', type=str,
                        help='write output image collection or not', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    csvDir = args.csvDir
    if (Path.is_dir(Path(args.csvDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.csvDir).joinpath('images').absolute())
    logger.info('csvDir = {}'.format(csvDir))
    delay = args.delay
    logger.info('delay = {}'.format(delay))
    feature = args.feature
    logger.info('feature = {}'.format(feature))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    groupVar = args.groupVar
    logger.info('groupVar = {}'.format(groupVar))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    percentile = args.percentile
    logger.info('percentile = {}'.format(percentile))
    removeDirection = args.removeDirection
    logger.info('removeDirection = {}'.format(removeDirection))
    sectionVar = args.sectionVar
    logger.info('sectionVar = {}'.format(sectionVar))
    writeOutput = args.writeOutput == 'true'
    logger.info('writeOutput = {}'.format(writeOutput))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
        # Get all file names in csvDir image collection
        csvDir_files = [f.name for f in Path(csvDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
        
        
        
        
        
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
        
        
        
        
        
        # Loop through files in csvDir image collection and process
        for i,f in enumerate(csvDir_files):
            # Load an image
            br = BioReader(Path(csvDir).joinpath(f))
            image = np.squeeze(br.read_image())

            # initialize the output
            out_image = np.zeros(image.shape,dtype=br._pix['type'])

            """ Do some math and science - you should replace this """
            logger.info('Processing image ({}/{}): {}'.format(i,len(csvDir_files),f))
            out_image = awesome_math_and_science_function(image)

            # Write the output
            bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
            bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z(),1,1)))# Loop through files in inpDir image collection and process
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