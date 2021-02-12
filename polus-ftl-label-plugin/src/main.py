import argparse, logging, ftl, bfio
from pathlib import Path
from preadator import ProcessManager
import numpy as np

def label_thread(input_path,output_path,connectivity):

    with ProcessManager.thread() as active_threads:
        with bfio.BioReader(input_path,max_workers=2) as br:
            with bfio.BioWriter(output_path,max_workers=2,metadata=br.metadata) as bw:
                # Load an image and convert to binary
                image = br[...,0,0]>0

                # Run the labeling algorithm
                labels = ftl.label_nd(image.squeeze(),connectivity)

                # Save the image
                bw.dtype = labels.dtype
                bw[:] = labels

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Label objects in a 2d or 3d binary image.')
    parser.add_argument('--connectivity', dest='connectivity', type=str,
                        help='City block connectivity, must be less than or equal to the number of dimensions', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    connectivity = int(args.connectivity)
    logger.info('connectivity = {}'.format(connectivity))
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))

    # We only need a thread manager since labeling and image reading/writing
    # release the gil
    ProcessManager.init_threads()

    # Get all file names in inpDir image collection
    files = [f for f in inpDir.iterdir() if f.is_file() and f.name.endswith('.ome.tif')]

    for file in files:

        ProcessManager.submit_thread(label_thread,
                                     file,outDir.joinpath(file.name),connectivity)

    ProcessManager.join_threads()

