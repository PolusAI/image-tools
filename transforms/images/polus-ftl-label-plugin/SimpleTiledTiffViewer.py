from bfio import BioReader
from pathlib import Path
import argparse
import logging
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='View *.ome.tif images and labels from FTL plugin.')
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

    # Get all file names in inpDir image collection
    files = [f for f in inpDir.iterdir() if f.is_file() and f.name.endswith('.tif')]

    for file in files:
        # Set up the BioReader
        with BioReader(inpDir / file.name) as br_in:
            img_in = br_in[:]

        with BioReader(outDir / file.name) as br_out:
            img_out = br_out[:]

        fig, ax = plt.subplots(1, 2, figsize=(16,8))
        ax[0].imshow(img_in), ax[0].set_title("Original Image")
        ax[1].imshow(img_out), ax[1].set_title("Labelled Image")
        fig.suptitle(file.name)
        plt.show()
        # Use savefig if you are on a headless machine, i.e. AWS EC2 instance
        # plt.savefig(outDir / (file.stem.split('.ome')[0] + '.png'))
        plt.close()

