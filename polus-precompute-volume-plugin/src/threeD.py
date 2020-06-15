import logging, argparse, bioformats
import javabridge as jutil
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import os
import sys

def main():

    logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format("test"),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("build_pyramid")
    logger.setLevel(logging.INFO) 

    listoffiles = os.listdir(sys.argv[1])
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    logger.info('Done Initializing')
    
    logger.info("Going into for loop")
    i = 0
    for item in listoffiles:
        fullpath = sys.argv[1] + '/' + item
        br = BioReader(fullpath)
        image = br.read_image().squeeze()
        logger.info("{}: {}".format(item, image.shape))
        i = i + 1

    jutil.kill_vm()
    exit()


main()
