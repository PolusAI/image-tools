from bfio.bfio import BioReader, BioWriter
from preadator import ProcessManager

import argparse, logging
import numpy as np
import typing
from pathlib import Path

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

TILE_SIZE = 2**13

def image_to_zarr(inp_image: Path,
                  out_dir: Path):
    
    with ProcessManager.process():
    
        with BioReader(inp_image) as br:
            
            # Loop through timepoints
            for t in range(br.T):

                # Loop through channels
                for c in range(br.C):
            
                    extension = ''.join(inp_image.suffixes)
                    
                    out_path = out_dir.joinpath(inp_image.name.replace(extension,'.ome.zarr'))
                    if br.C > 1:
                        out_path = out_dir.joinpath(inp_image.name.replace(extension,f'_c{c}.ome.zarr'))
                    if br.T > 1:
                        out_path = out_dir.joinpath(inp_image.name.replace(extension,f'_t{c}.ome.zarr'))
                    
                    with BioWriter(out_path,max_workers=ProcessManager._active_threads,metadata=br.metadata) as bw:
                        
                        bw.C = 1
                        bw.T = 1
                        bw.channel_names = [br.channel_names[c]]
                        
                        # Loop through z-slices
                        for z in range(br.Z):
                            
                            # Loop across the length of the image
                            for y in range(0,br.Y,TILE_SIZE):
                                y_max = min([br.Y,y+TILE_SIZE])

                                # Loop across the depth of the image
                                for x in range(0,br.X,TILE_SIZE):
                                    x_max = min([br.X,x+TILE_SIZE])
                                    
                                    bw[y:y_max,x:x_max,z:z+1,0,0] = br[y:y_max,x:x_max,z:z+1,c,t]

def main(inpDir: Path,
         outDir: Path,
         ) -> None:
    
    ProcessManager.init_processes("main","zarr")
        
    files = list(inpDir.iterdir())
    
    for file in inpDir.iterdir():
        ProcessManager.submit_process(image_to_zarr,file,outDir)
    
    ProcessManager.join_processes()

if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Convert Bioformats supported format to OME Zarr.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input generic data collection to be processed by this plugin', required=True)
    
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    
    main(inpDir=inpDir,
         outDir=outDir)