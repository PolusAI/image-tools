import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count
from pathlib import Path

import filepattern
import numpy as np
import zarr
from bfio.bfio import BioReader

import dynamics

TILE_SIZE = 2048
TILE_OVERLAP = 512

def flow_thread(input_path: Path,
                zfile: Path,
                x: int,
                y: int,
                z: int) -> bool:
    """ Converts labels to flows
        Args:
            input_path(path): Path of input image collection
            zfile(path): Path  where output zarr file will be saved
            x(int): start index of the tile  in x dimension of image
            y(int): start index of the tile  in y dimension of image
            z(int): z slice of the  image

        """
    root = zarr.open(str(zfile))
  #  print('xyz',x,y,z)
    with BioReader(input_path) as br:
        x_min = max([0, x - TILE_OVERLAP])
        x_max = min([br.X, x + TILE_SIZE + TILE_OVERLAP])

        y_min = max([0, y - TILE_OVERLAP])
        y_max = min([br.Y, y + TILE_SIZE + TILE_OVERLAP])

        flow = dynamics.labels_to_flows(br[y_min:y_max, x_min:x_max, z:z + 1, 0, 0].squeeze())
        logger.info('tsedfs %d %d %d ',br.Y,br.X,br.Z)
        flow_final = flow[:, :, :, np.newaxis, np.newaxis].transpose(1, 2, 3, 0, 4)

        x_overlap = x - x_min
        x_min = x
        x_max = min([br.X, x + TILE_SIZE])

        y_overlap = y - y_min
        y_min = y
        y_max = min([br.Y, y + TILE_SIZE])

        root[f]['vector'][y_min:y_max, x_min:x_max, z:z + 1, 0:3, 0:1] = flow_final[
                                                                          y_overlap:y_max - y_min + y_overlap,
                                                                          x_overlap:x_max - x_min + x_overlap,
                                                                          ...]
        root[f]['lbl'][y_min:y_max, x_min:x_max, z:z + 1, 0:1, 0:1] = br[y_min:y_max, x_min:x_max,
                                                                       z:z + 1, 0, 0]

    return True


if __name__ == "__main__":
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
    parser.add_argument('--inpRegex', dest='inpRegex', type=str,
                        help='Input file name pattern.', required=False)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    num_threads = max([cpu_count() // 2, 1])

    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # Switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    inpRegex = args.inpRegex
    logger.info('FIle pattern = {}'.format(inpRegex))
    outDir = args.outDir

    logger.info('outDir = {}'.format(outDir))

    # Start  logging
    logger.info('Initializing ...')
    # Get all file names in inpDir image collection based on input pattern
    if inpRegex:
        fp = filepattern.FilePattern(inpDir, inpRegex)
        inpDir_files = [file[0]['file'].name for file in fp()]
        logger.info('Processing %d labels based on filepattern ' % (len(inpDir_files)))
    else:
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if
                        f.is_file() and str(f).endswith('.ome.tif')]
        logger.info('Processing %d labels in the directory ' % (len(inpDir_files)))

    root = zarr.group(store=str(Path(outDir).joinpath('flow.zarr')))
    # Loop through files in inpDir image collection and process
    # processes = []
    # with ProcessPoolExecutor(num_threads) as executor:
    for f in inpDir_files:
        logger.info('Processing image %s ', f)
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()),backend ='python')

        # Initialize the zarr group, create datasets
        cluster = root.create_group(f)
        init_cluster_1 = cluster.create_dataset('vector', shape=(br.Y, br.X, br.Z, 3, 1),
                                                    chunks=(TILE_SIZE, TILE_SIZE, 1, 3, 1),
                                                    dtype=np.float32)
        init_cluster_2 = cluster.create_dataset('lbl', shape=br.shape,
                                                    chunks=(TILE_SIZE, TILE_SIZE, 1, 3, 1),
                                                    dtype=np.float32)
        cluster.attrs['metadata'] = str(br.metadata)

        for z in range(br.Z):
            logger.info('Processing slice  %d', z)
            for x in range(0, br.X, TILE_SIZE):
                for y in range(0, br.Y, TILE_SIZE):
                        # processes.append(executor.submit(flow_thread,
                        #                                  Path(inpDir).joinpath(f).absolute(),
                        #                                  Path(outDir).joinpath('flow.zarr'),
                        #                                  x, y, z))
                    logger.info('entryyxz %d %d %d', y, x, z)
                    test=flow_thread(Path(inpDir).joinpath(f).absolute(),
                                                          Path(outDir).joinpath('flow.zarr'),
                                                          x, y, z)

        br.close()

        # done, not_done = wait(processes, 0)
        #
        # logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
        #
        # while len(not_done) > 0:
        #     done, not_done = wait(processes, 3)
        #     logger.info(f'Percent complete: {100 * len(done) / len(processes):6.3f}%')
