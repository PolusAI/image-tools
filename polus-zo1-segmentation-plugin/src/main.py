from bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing, os
import numpy as np
from pathlib import Path

# Set the environment variable to prevent odd warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import tensorflow as tf

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segment epithelial cell borders labeled for ZO1 tight junction protein.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file()]
    
    # Loop through files in inpDir image collection and process
    batch_size = 50
    for f in range(0,len(inpDir_files),batch_size):
        
        images = inpDir_files[f:min([f+batch_size,len(inpDir_files)])]
        
        process = subprocess.Popen('python3 segment.py --inpDir {} --outDir {} --images {}'.format(inpDir,
                                                                                                   outDir,
                                                                                                   ','.join(images)),
                                    shell=True)
        
        process.wait()
