import argparse
import time
from pathlib import Path
from bfio import czi2tif
import javabridge as jutil
import bioformats
import logging

def main():
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)


    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    
    logger.info('Initializing the javabridge...')
    jutil.start_vm(class_path=bioformats.JARS)

    logger.info('Extracting tiffs and saving as ome.tif...')
    files = [f for f in Path(input_dir).iterdir() if f.is_file() and f.suffix=='.czi']
    if not files:
        logger.error('No CZI files found.')
        ValueError('No CZI files found.')
    
    for file in files:
        start_time = time.time()
        logger.info('Starting extraction from ' + str(file.absolute()) + '...')
        czi2tif.write_ome_tiffs(str(file.absolute()),output_dir)
        logger.info('Finished in {}s!'.format(time.time()-start_time))
        
    logger.info('Finished extracting files. Closing the javabridge and exiting...')
    jutil.kill_vm()

if __name__ == "__main__":
    main()