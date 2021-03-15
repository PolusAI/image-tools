from bfio.bfio import BioReader
import argparse, logging, sys
import numpy as np
from pathlib import Path
import dynamics
import zarr

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')

    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                         help='Input image collection to be processed by this plugin', required=True)
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
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Surround with try/finally for proper error catching
    try:
        # Start  logging
        logger.info('Initializing ...')
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif' ]
        if Path(outDir).joinpath('flow.zarr').exists():
           raise FileExistsError('Zarr file exists. Delete the existing file')
        root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
        for f in inpDir_files:
            logger.info('Processing image %s ',f)
        # Loop through files in inpDir image collection and process
            br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
            tile_size = min(1080, br.X)
            # Initializing a array to store the vector field
            flow_final = np.zeros((br.Y, br.X, br.Z, 3, 1)).astype(np.float32)
            # Iterating over Z dimension
            for z in range(br.Z):
                for x in range(0, br.X, tile_size):
                    x_max = min([br.X, x + tile_size])
                    for y in range(0, br.Y, tile_size):
                        y_max = min([br.Y, y + tile_size])
                        flow = dynamics.labels_to_flows(br[y:y_max, x:x_max,z:z+1, 0,0].squeeze())
                        flow=flow[:,:,:,np.newaxis,np.newaxis]
                        flow=flow.transpose((1,2,3,0,4))
                        flow_final[y:y_max, x:x_max,z:z+1,:,:] = flow

            # Saving  Vector in chunks
            cluster = root.create_group(f)
            init_cluster_1 = cluster.create_dataset('vector', shape=flow_final.shape, data=flow_final,chunks=(br.Y, br.X, br.Z, 3, 1))
            init_cluster_2 = cluster.create_dataset('lbl', shape=br.shape, data=br[:],chunks=(br.Y, br.X, br.Z, 1, 1))
            cluster.attrs['metadata'] = str(br.metadata)

    except FileExistsError:
        logger.info('Zarr file exists. Delete the existing file %r' % str((Path(outDir).joinpath('flow.zarr'))))
    finally:
        logger.info('Closing ')
        # Exit the program
        sys.exit()