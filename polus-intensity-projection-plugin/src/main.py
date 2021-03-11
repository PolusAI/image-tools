import argparse, logging, time, sys, os, traceback
from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import numpy as np
from preadator import ProcessManager

# x,y size of the 3d image chunk to be loaded into memory
tile_size = 1024

# depth of the 3d image chunk
tile_size_z = 128

def max_min_projection(br, bw, x_range, y_range, **kwargs):
    """ Calculate the max or min intensity
    projection of a section of the input image.

    Args:
        br (BioReader object): input file object
        bw (BioWriter object): output file object
        x_range (tuple): x-range of the img to be processed
        y_range (tuple): y-range of the img to be processed

    Returns:
        image array : Max IP of the input volume
    """
    with ProcessManager.thread():
        br.max_workers = ProcessManager._active_threads
        bw.max_workers = ProcessManager._active_threads

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
        bw[y:y_max,x:x_max,0:1,0,0] = method(out_image, axis=2)



def mean_projection(br, bw, x_range, y_range, **kwargs):
    """ Calculate the mean intensity projection

    Args:
        br (BioReader object): input file object
        bw (BioWriter object): output file object
        x_range (tuple): x-range of the img to be processed
        y_range (tuple): y-range of the img to be processed

    Returns:
        image array : Mean IP of the input volume
    """
    with ProcessManager.thread():
        br.max_workers = ProcessManager._active_threads
        bw.max_workers = ProcessManager._active_threads

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
        bw[y:y_max,x:x_max,0:1,0,0] = out_image.astype(br.dtype)


def process_image(input_img_path, output_img_path, projection, method):

    # Grab a free process
    with ProcessManager.process():

        # initalize biowriter and bioreader
        with BioReader(input_img_path, max_workers=ProcessManager._active_threads) as br, \
            BioWriter(output_img_path, metadata=br.metadata, max_workers=ProcessManager._active_threads) as bw:

            # output image is 2d
            bw.Z = 1

            # iterate along the x,y direction
            for x in range(0,br.X,tile_size):
                x_max = min([br.X,x+tile_size])

                for y in range(0,br.Y,tile_size):
                    y_max = min([br.Y,y+tile_size])

                    ProcessManager.submit_thread(projection,br,bw,(x, x_max),(y, y_max),method=method)

            ProcessManager.join_threads()


def main(inpDir, outDir, projection, method):

    # images in the input directory
    inpDir_files = os.listdir(inpDir)
    inpDir_files = [filename for filename in inpDir_files if filename.endswith('.ome.tif')]

    # Surround with try/finally for proper error catching
    try:
        for image_name in inpDir_files:

            input_img_path = os.path.join(inpDir, image_name)
            output_img_path = os.path.join(outDir, image_name)

            ProcessManager.submit_process(process_image, input_img_path, output_img_path, projection, method)

        ProcessManager.join_processes()

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        logger.info('Exiting the workflow..')
        sys.exit()
    
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
    
    ProcessManager.init_processes('main','intensity')

    main(inpDir, outDir, projection, method)



    



