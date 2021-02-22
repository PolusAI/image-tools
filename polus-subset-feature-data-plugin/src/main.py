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
    parser.add_argument('--background', dest='background', type=str,
                        help='The background of images in the collection', required=True)
    parser.add_argument('--csvDir', dest='csvDir', type=str,
                        help='CSV collection containing features', required=True)
    parser.add_argument('--feature', dest='feature', type=str,
                        help='Feature to subset data', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--timeseries', dest='timeseries', type=str,
                        help='Variables', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    background = args.background
    logger.info('background = {}'.format(background))
    csvDir = args.csvDir
    logger.info('csvDir = {}'.format(csvDir))
    feature = args.feature
    logger.info('feature = {}'.format(feature))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    timeseries = args.timeseries
    logger.info('timeseries = {}'.format(timeseries))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:
        
        # Get all file names in csvDir csv image collection
        csvDir_files = [f.name for f in Path(csvDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.csv']
        
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
        
        for i,f in enumerate(inpDir_files):


        
    finally:
        # Exit the program
        sys.exit()