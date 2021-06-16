from os import cpu_count
from bfio import BioReader, BioWriter, OmeXml
import argparse, logging
import numpy as np
from pathlib import Path
from cellpose import dynamics, utils
import torch
from concurrent.futures import ThreadPoolExecutor, wait, Future
import typing

""" Plugin Constants """
TILE_SIZE = 2048    # Largest chunk of an image to process
TILE_OVERLAP = 256  # Amount of overlap between tiles
NITER = 200         # Number of iterations to run flow dynamics

# Use a gpu if it's available
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    DEV = torch.device("cuda")
else:
    DEV = torch.device("cpu")

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def overlap(previous_values: np.ndarray,
            current_values: np.ndarray,
            tile: np.ndarray
            ) -> np.ndarray:
    """Resolve label values between tiles
    
    This function takes a row/column from the previous tile and a row/column
    from the current tile and finds labels that that likely match. If labels
    in the current tile should be replaced with labels from the previous tile,
    the pixels in the current tile are removed from ``tile`` and the label value
    and pixel coordinates of the label are stored in ``labels`` and ``indices``
    respectively.

    Args:
        previous_values (np.ndarray): Previous tile edge values
        current_values (np.ndarray): Current tile edge values
        tile (np.ndarray): Current tile pixel values, flattened

    Returns:
        np.ndarray: Current tile pixel values, flattened
    """
    
    # Get a list of unique values in the previous and current tiles
    previous_labels = np.unique(previous_values)
    if previous_values[0] == 0:
        previous_labels = previous_values[1:]
    
    current_labels = np.unique(current_values)
    if current_labels[0] == 0:
        current_labels = current_labels[1:]
    
    # Initialize outputs
    indices = []
    labels = []
    
    if previous_labels.size != 0 and current_labels.size != 0:
    
        # Find overlapping indices
        for label in current_labels:
            
            new_labels,counts = np.unique(previous_values[current_values==label],return_counts=True)
            
            if new_labels.size == 0:
                continue
            
            if new_labels[0] == 0:
                new_labels = new_labels[1:]
                counts = counts[1:]
                
            if new_labels.size == 0:
                continue
                
            # Get the most frequently occuring overlapping label
            labels.append(new_labels[np.argmax(counts)])
            
            # Add indices to output, remove pixel values from the tile
            indices.append(np.argwhere(tile==label))
            tile[indices[-1]] = 0
            
    return tile, labels, indices

def mask_thread(coords: typing.Tuple[int,int,int],
                file_path: Path,
                bw: BioWriter,
                cellprob_threshold: float,
                flow_threshold: float,
                dependency1: typing.Optional[Future],
                dependency2: typing.Optional[Future]
                ) -> typing.Tuple[np.ndarray,np.ndarray,np.uint32]:
    """[summary]

    Args:
        coords: x,y,z starting coordinates of the tile to process
        file_path: Vector field file path
        bw: Output file
        cellprob_threshold: Cell probability threshold
        flow_threshold: Flow field threshold
        dependency1: Tile future to the left of the current tile
        dependency2: Tile future above the current tile

    Returns:
        typing.Tuple[np.ndarray,np.ndarray,list]: Returns the right column of
            the processed tile, bottom row of the processed tile, and largest
            label value.
    """   
    
    # Calculate indice for the tile
    x,y,z = coords
    with BioReader(file_path) as br:
        
        image_X = br.X
        image_Y = br.Y
        
        x_min = max([0, x - TILE_OVERLAP])
        x_max = min([image_X, x + TILE_SIZE + TILE_OVERLAP])

        y_min = max([0, y - TILE_OVERLAP])
        y_max = min([image_Y, y + TILE_SIZE + TILE_OVERLAP])

        tile = br[y_min:y_max, x_min:x_max, z:z + 1, :3, 0]

    logger.debug('Calculating flows and masks  for tile [{}:{},{}:{},{}:{}]'.format(y, y_max, x,
                 x_max, z, z + 1))

    # Get flows and probabilities
    cellprob = tile[:,:,0,0].squeeze()
    dP = tile[:,:,0,1:].squeeze().transpose(2,0,1)

    # Compute flows for the tile
    p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                                niter=NITER, interp=True,
                                use_gpu=USE_GPU,device=DEV)
    mask = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                              flows=dP, threshold=flow_threshold,
                              use_gpu=USE_GPU,device=DEV)
    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=15)

    # reshape mask based on tile
    x_overlap = x - x_min
    x_min = x
    x_max = min([image_X, x + TILE_SIZE])

    y_overlap = y - y_min
    y_min = y
    y_max = min([image_Y, y + TILE_SIZE])

    mask = mask[y_overlap:y_max - y_min + y_overlap,
                x_overlap:x_max - x_min + x_overlap,
                np.newaxis, np.newaxis, np.newaxis].astype(np.uint32)
    
    """ Fix tile conflicts if image is large enough to require tiling """
    # Get previously processed tiles if they exist
    dependency1 = None if dependency1 is None else dependency1.result()
    dependency2 = None if dependency2 is None else dependency2.result()
    
    # Get offset to make labels consistent between tiles
    offset = 0 if dependency1 is None else dependency1[2]
    
    current_x = mask[:,0].squeeze()
    current_y = mask[0,:].squeeze()
    shape = mask.shape
    mask = mask.reshape(-1)
    
    # Resolve label conflicts along the left border
    if x > 0:
        
        mask, labels_x, indices_x = overlap(dependency1[0].squeeze(),current_x,mask)
        
    if y > 0:
        
        mask, labels_y, indices_y = overlap(dependency2[1].squeeze(),current_y,mask)
    
    _, image = np.unique(mask, return_inverse=True)
    image = image.astype(np.uint32)
    image[image>0] = image[image>0] + offset
    
    if x > 0:
        for label,ind in zip(labels_x,indices_x):
            if ind.size==0:
                continue
            image[ind] = label
            
    if y > 0:
        for label,ind in zip(labels_y,indices_y):
            if ind.size==0:
                continue
            image[ind] = label

    image = image.reshape(shape)
    bw[y_min:y_max, x_min:x_max, z:z + 1, 0, 0] = image
    
    return image[:,-1],image[-1,:],image.max()

def close_thread(dependency: Future,
                 bw: BioWriter):
    """ Close an image once the final tile is written

    Args:
        dependency (Future): The final tile thread
        bw (BioWriter): The BioWriter to clsoe

    Returns:
        Returns True when completed
    """    
    
    dependency.result()
    
    bw.close()
    
    return True

def main(inpDir: Path,
         cellprob_threshold: float,
         flow_threshold: float,
         outDir: Path
         ) -> None:
    
    # Get the list of files in path
    files = [p for p in Path(inpDir).iterdir() if p.name.endswith('_flow.ome.zarr')]
    
    num_threads = max([cpu_count()//2,1])
    logger.info(f'Processing tiles with {num_threads} threads using {DEV}')
    
    if len(files) == 0:
        logger.critical('No flow files detected.')
        quit()
    
    processes = []
    with ThreadPoolExecutor(num_threads) as executor:

        # Loop through files in inpDir image collection and process
        for ind,fpath in enumerate(files):
        
            br = BioReader(fpath)
                
            threads = np.empty((br.shape[:3]),dtype=object)
                            
            logger.debug(
                'Processing image ({}/{}): {}'.format(ind, len(files),
                                                      fpath))

            # TODO: Hard coding to ome.tif for now, this should be changed later.
            path = Path(outDir).joinpath(fpath.name.replace('_flow.ome.zarr','.ome.tif'))

            bw = BioWriter(file_path=Path(path), metadata=br.metadata)
            bw.dtype=np.dtype(np.uint32)
            bw.C = 1
            bw.channel_names = ['label']

            for z in range(0, br.Z, 1):
                
                y_ind = None
                dependency1 = None

                for y in range(0, br.Y, TILE_SIZE):

                    for x in range(0, br.X, TILE_SIZE):
                        
                        dependency2 = None if y_ind is None else threads[y_ind,x//TILE_SIZE,z]

                        processes.append(executor.submit(mask_thread,
                                                         (x,y,z),
                                                         fpath,bw,
                                                         cellprob_threshold,flow_threshold,
                                                         dependency1,dependency2))
                        dependency1 = processes[-1]
                        threads[y//TILE_SIZE,x//TILE_SIZE,z] = dependency1
                    
                    y_ind = y//TILE_SIZE
            
            executor.submit(close_thread,dependency1,bw)
                        
        done, not_done = wait(processes, 0)

        logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')

        while len(not_done) > 0:
            for r in done:
                r.result()
            done, not_done = wait(processes, 15)
            logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')

if __name__ == '__main__':
    
    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--flowThreshold', required=False,
                        default=0.8, type=float, help='flow error threshold, 0 turns off this optional QC step')
    parser.add_argument('--cellprobThreshold', required=False,
                        default=0.0, type=float, help='cell probability threshold, centered at 0.0')
    
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    cellprob_threshold = args.cellprobThreshold
    logger.info('cellprobThreshold = {}'.format(cellprob_threshold))
    flow_threshold= args.flowThreshold
    logger.info('flowThreshold = {}'.format(flow_threshold))
    
    main(inpDir,
         cellprob_threshold,
         flow_threshold,
         outDir)