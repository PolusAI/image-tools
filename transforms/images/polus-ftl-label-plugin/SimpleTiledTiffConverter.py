from bfio import BioReader, BioWriter, LOG4J, JARS
import javabridge
from pathlib import Path
import argparse
import logging

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Convert images from *.tif to *.ome.tif tiled tif.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))


    javabridge.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],
                        class_path=JARS,
                        run_headless=True)

    # Get all file names in inpDir image collection
    files = [f for f in inpDir.iterdir() if f.is_file() and f.name.endswith('.tif')]

    try:
        for file in files:
            # Set up the BioReader
            with BioReader(inpDir / file,backend='java') as br, \
                BioWriter(outDir / f'{file.stem}.ome.tif',metadata=br.metadata,backend='python') as bw:
            
                # Print off some information about the image before loading it
                print('br.shape: {}'.format(br.shape))
                print('br.dtype: {}'.format(br.dtype))
                
                # Read in the original image, then save
                original_image = br[:]
                bw[:] = original_image
        
    finally:
        # Close the javabridge. Since this is in the finally block, it is always run
        javabridge.kill_vm()