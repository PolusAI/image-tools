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
            image = np.squeeze(br.read())
            if len(image.shape) ==2:
                flow,lbl= dynamics.labels_to_flows(image)
                flow= np.asarray(flow.transpose((1,2,0)))
                lbl = np.asarray(lbl)

            else:
            # Iterating over a Zstack
                flow_final=[]
                lbl_final=[]
                for i in range(image.shape[-1]):
                    flow, lbl = dynamics.labels_to_flows(image[:, :, i])
                    flow=flow.transpose((1,2,0))
                    flow_final.append(flow.tolist())
                    lbl_final.append(lbl.tolist())
                flow = np.asarray(flow_final)
                lbl = np.asarray(lbl_final)
            cluster = root.create_group(f)
            init_cluster_1 = cluster.create_dataset('vector', shape=flow.shape, data=flow)
            init_cluster_2 = cluster.create_dataset('lbl', shape=lbl.shape, data=lbl)
            cluster.attrs['metadata'] = str(br.metadata)
    except FileExistsError:
        logger.info('Zarr file exists. Delete the existing file %r' % str((Path(outDir).joinpath('flow.zarr'))))
    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing ')

        
        # Exit the program
        sys.exit()