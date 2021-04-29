from bfio.bfio import BioReader
import argparse, logging
import numpy as np
from pathlib import Path
import dynamics
import zarr
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

TILE_SIZE = 2048
TILE_OVERLAP = 1024

def flow_thread(input_path: Path,
                zfile: Path,
                x: int,
                y: int,
                z: int) -> bool:
    
    root = zarr.open(zfile)
    
    with BioReader(input_path) as br:
        
        x_min = max([0,x-TILE_OVERLAP])
        x_max = min([br.X,x+TILE_SIZE+TILE_OVERLAP])
        
        y_min = max([0,y-TILE_OVERLAP])
        y_max = min([br.Y,y+TILE_SIZE+TILE_OVERLAP])
    
        flow = dynamics.labels_to_flows(br[y_min:y_max, x_min:x_max,z:z+1, 0,0].squeeze())
        flow_final = flow[:,:,:,np.newaxis,np.newaxis].transpose(1,2,3,0,4)
        
        x_overlap = x - x_min
        x_min = x
        x_max = min([br.X,x+TILE_SIZE])
        
        y_overlap = y - y_min
        y_min = y
        y_max = min([br.Y,y+TILE_SIZE])
        
        zfile[f]['vector'][y_min:y_max, x_min:x_max,z:z+1,0:3,0:1] = flow_final[y_overlap:y_max-y_min+y_overlap,x_overlap:x_max-x_min+x_overlap,...]
        zfile[f]['lbl'][y_min:y_max, x_min:x_max,z:z+1, 0:1,0:1] = br[y_min:y_max, x_min:x_max,z:z+1, 0,0]
        
    return True

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
    
    num_threads = max([cpu_count()//2,1])

    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Start  logging
    logger.info('Initializing ...')
    # Get all file names in inpDir image collection
    inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif' ]
        
    root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
    # Loop through files in inpDir image collection and process
    processes = []
    with ProcessPoolExecutor(num_threads) as executor:
        for f in inpDir_files:
            logger.info('Processing image %s ',f)
            br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
            tile_size = min(2048, br.X)
            
            # Initialize the zarr group, create datasets
            cluster = root.create_group(f)
            init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y,br.X,br.Z,3,1),
                                                    chunks=(1024, 1024, 1, 3, 1),
                                                    dtype=np.float32)
            init_cluster_2 = cluster.create_dataset('lbl', shape=br.shape,
                                                    chunks=(1024, 1024, 1, 3, 1),
                                                    dtype=np.float32)
            cluster.attrs['metadata'] = str(br.metadata)
            
            for z in range(br.Z):
                for x in range(0, br.X, tile_size):
                    x_max = min([br.X, x + tile_size])
                    for y in range(0, br.Y, tile_size):
                        
                        processes.append(executor.submit(flow_thread,
                                                         str(Path(inpDir).joinpath(f).absolute()),
                                                         str(Path(outDir).joinpath('flow.zarr')),
                                                         x,y,z))
                        
            br.close()
            
        done, not_done = wait(processes,0)
            
        logger.info(f'Percent complete: {100*len(done)/len(processes):6.3f}%')
        
        while len(not_done) > 0:
            
            done, not_done = wait(processes,3)
            
            logger.info(f'Percent complete: {100*len(done)/len(processes):6.3f}%')
            
            