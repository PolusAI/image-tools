import argparse, logging, time, sys, os, traceback
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import numpy as np

# x,y size of the 3d image chunk to be loaded into memory
tile_size = 1024

# depth of the 3d image chunk
tile_size_z = 128

def max_min_projection(br, x_range, y_range, **kwargs):
    """ This function calculates the max or min intensity
    projection of a section (specified by x_range and
    y_range) of the input image.

    Args:
        br (BioReader object):
        x_range (tuple): x-range of the img to be processed
        y_range (tuple): y-range of the img to be processed

    Returns:
        image array : Max IP of the input volume
    """

    # set projection method
    if not 'method' in kwargs:
        method = np.max
    else:
        method = kwargs['method']

    # x,y range of the volume
    x, x_max = x_range
    y, y_max = y_range

    # iterate over depth
    for ind, z in enumerate(range(0,br.Z,tile_size_z)):
        z_max = min([br.Z,z+tile_size_z])
        if ind == 0:
            out_image = method(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)
        else:
            out_image = np.dstack((out_image, method(br[y:y_max,x:x_max,z:z_max,0,0], axis=2)))

    # output image
    out_image = method(out_image, axis=2)
    return out_image

def mean_projection(br, x_range, y_range, **kwargs):
    """ Calculate the mean intensity projection

    Args:
        br (BioReader object):
        x_range (tuple): x-range of the img to be processed
        y_range (tuple): y-range of the img to be processed

    Returns:
        image array : Mean IP of the input volume
    """
    # x,y range of the volume
    x, x_max = x_range
    y, y_max = y_range

    # iterate over depth
    out_image = np.zeros((y_max-y,x_max-x),dtype=np.float64)
    for ind, z in enumerate(range(0,br.Z,tile_size_z)):
        z_max = min([br.Z,z+tile_size_z])

        out_image += np.sum(br[y:y_max,x:x_max,z:z_max,...].astype(np.float64),axis=2).squeeze()

    # output image
    out_image /= br.Z
    return out_image.astype(br.dtype)

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

    # initialize projection function
    if projectionType == 'max':
        projection = max_min_projection
        method = np.max
    elif projectionType == 'min':
        projection = max_min_projection
        method = np.min
    elif projectionType == 'mean':
        projection = mean_projection
        method = None

    # images in the input directory
    inpDir_files = os.listdir(inpDir)
    inpDir_files = [filename for filename in inpDir_files if filename.endswith('.ome.tif')]

    # Surround with try/finally for proper error catching
    try:
        for image_name in inpDir_files:
            logger.info('---- Processing image: {} ----'.format(image_name))

            # initalize biowriter and bioreader
            with BioReader(os.path.join(inpDir, image_name)) as br, \
                BioWriter(os.path.join(outDir, image_name),metadata=br.metadata) as bw:

                # output image is 2d
                bw.Z = 1

                # iterate along the x,y direction
                for x in range(0,br.X,tile_size):
                    x_max = min([br.X,x+tile_size])

                    for y in range(0,br.Y,tile_size):
                        y_max = min([br.Y,y+tile_size])
                        logger.info('Processing volume x: {}-{}, y: {}-{}'.format(x,x_max,y,y_max))

                        # write output
                        bw[y:y_max,x:x_max,0:1,0,0] = projection(br, (x, x_max), (y, y_max), method=method)

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        logger.info('Exiting the workflow..')
        sys.exit()

