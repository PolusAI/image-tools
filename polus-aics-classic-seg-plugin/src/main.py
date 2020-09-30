#from bfio.bfio import BioReader, BioWriter
#import bioformats
#import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing, sys
import numpy as np
from pathlib import Path
import os
import json
from Workflows import Playground_CurvyLinear
from Workflows import Playground_dots
from Workflows import Playground_gja1
from Workflows import Playground_lamp1
import cv2
from aicsimageio import AICSImage
import traceback

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='The plugin integrates the allen cell structure segmenter into WIPP')
    
    # Input arguments
    parser.add_argument('--configFile', dest='configFile', type=str,
                        help='Configuration file for the workflow', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    configFile = args.configFile
    logger.info('configFile = {}'.format(configFile))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))

    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # load config file
    """
    config_file_path = os.path.join(configFile[:-7],'metadata_files')
    metafiles=os.listdir(config_file_path)
    with open(os.path.join(config_file_path,metafiles[0])) as json_file:
        config_data = json.load(json_file)

    """
    config_data = {
        "workflow_name": "Playground4_Curvi",
        "intensity_scaling_param": [
            3.5,
            15
        ],
        "gaussian_smoothing_sigma": 1,
        "preprocessing_function": "image_smoothing_gaussian_3d",
        "f2_param": [
            [
                1.5,
                0.16
            ]
        ],
        "minArea": 5
    }
    

    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        #jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
 
        
        if config_data['workflow_name'] == 'Playground4_Curvi':
            logger.info('executing {}'.format(config_data['workflow_name'] ))
            Playground_CurvyLinear.segment_images(inpDir, outDir, config_data)
        elif config_data['workflow_name'] == 'Playground_dots':
            logger.info('executing {}'.format(config_data['workflow_name'] ))
            Playground_dots.segment_images(inpDir, outDir, config_data)
        elif config_data['workflow_name'] == 'Playground_gja1':
            logger.info('executing {}'.format(config_data['workflow_name'] ))
            Playground_gja1.segment_images(inpDir, outDir, config_data)
        elif config_data['workflow_name'] == 'Playground_lamp1':
            logger.info('executing {}'.format(config_data['workflow_name'] ))
            Playground_lamp1.segment_images(inpDir, outDir, config_data)
    
    except Exception:
        traceback.print_exc()

    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the javabridge')
        #jutil.kill_vm()
        
        # Exit the program
        sys.exit()