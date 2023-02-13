from bfio.bfio import BioReader, BioWriter
import filepattern
import argparse, logging
import numpy as np
import typing, os
from pathlib import Path
import numpy as np
from preadator import ProcessManager

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

CHUNK_SIZE = 4096

OPERATIONS = {
    'multiply': np.multiply,
    'divide': np.divide,
    'add': np.add,
    'subtract': np.subtract,
    'and': np.bitwise_and,
    'or': np.bitwise_or,
    'xor': np.bitwise_xor
}

def process_chunk(brp: BioReader,
                  brs: BioReader,
                  x,x_max,
                  y,y_max,
                  z,
                  bw: BioWriter,
                  operation: str):
    
    with ProcessManager.thread():
        bw[y:y_max,x:x_max,z] = OPERATIONS[operation](
            brp[y:y_max,x:x_max,z],brs[y:y_max,x:x_max,z]
        )
        
    return

def process_image(file: str,
                   sfile: str,
                   outDir: str,
                   operation: str
                   ) -> np.ndarray:
    """Awesome function (actually just a template)
    
    This function should do something, but for now just returns the input.

    Args:
        input_data: A numpy array.
        
    Returns:
        np.ndarray: Returns the input image.
    """
    
    with ProcessManager.process(file['file'].name):
        # Load the input image
        with BioReader(file['file']) as brp, BioReader(sfile['file']) as brs:
            
            input_extension = ''.join([s for s in file['file'].suffixes[-2:] if len(s) < 6])
            out_name = file['file'].name.replace(input_extension,POLUS_EXT)
            out_path = outDir.joinpath(out_name)
            
            # Initialize the output image
            with BioWriter(out_path,metadata=brp.metadata) as bw:
                
                for z in range(brp.Z):
                    for x in range(0,brp.X,CHUNK_SIZE):
                        x_max = min([x+CHUNK_SIZE,brp.X])
                        for y in range(0,brp.Y,CHUNK_SIZE):
                            y_max = min([y+CHUNK_SIZE,brp.Y])
                            
                            ProcessManager.submit_thread(
                                process_chunk,
                                brp,brs,
                                x,x_max,
                                y,y_max,
                                z,
                                bw,
                                operation
                            )
        
                ProcessManager.join_threads()
                
    return
    
def main(primaryDir: Path,
         primaryPattern: str,
         operator: str,
         secondaryDir: Path,
         secondaryPattern: str,
         outDir: Path,
         ) -> None:
    
    assert operator in OPERATIONS.keys()
    
    primaryPattern = primaryPattern if primaryPattern is not None else '.*'
    secondaryPattern = secondaryPattern if secondaryPattern is not None else '.*'
    
    fp_primary = filepattern.FilePattern(primaryDir,primaryPattern)
    fp_secondary = filepattern.FilePattern(secondaryDir,secondaryPattern)
    
    ProcessManager.init_processes()
    
    for files in fp_primary:
        # get the first file
        file = files.pop()
        
        logger.info(f'Processing image: {file["file"]}')
        
        matches = fp_secondary.get_matching(**{k.upper():v for k,v in file.items() if k is not 'file'})
        if len(matches) > 1:
            logger.warning(f'Found multiple secondary images to match the primary image: {file.name}')
        sfile = matches.pop()
        
        ProcessManager.submit_process(process_image,file,sfile,outDir,operator)
        
    ProcessManager.join_processes()
        
if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Perform simple mathematical operations on images.')
    
    # Input arguments
    parser.add_argument('--primaryDir', dest='primaryDir', type=str,
                        help='The first set of images', required=True)
    parser.add_argument('--primaryPattern', dest='primaryPattern', type=str,
                        help='Filename pattern used to separate data', required=False)
    parser.add_argument('--operator', dest='operator', type=str,
                        help='The operation to perform', required=True)
    parser.add_argument('--secondaryDir', dest='secondaryDir', type=str,
                        help='The second set of images', required=True)
    parser.add_argument('--secondaryPattern', dest='secondaryPattern', type=str,
                        help='Filename pattern used to separate data', required=False)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    primaryDir = Path(args.primaryDir)
    if (primaryDir.joinpath('images').is_dir()):
        # switch to images folder if present
        primaryDir = primaryDir.joinpath('images').absolute()
    logger.info('primaryDir = {}'.format(primaryDir))
    
    primaryPattern = args.primaryPattern
    logger.info('primaryPattern = {}'.format(primaryPattern))
    
    operator = args.operator
    logger.info('operator = {}'.format(operator))
    
    secondaryDir = Path(args.secondaryDir)
    if (secondaryDir.joinpath('images').is_dir()):
        # switch to images folder if present
        secondaryDir = secondaryDir.joinpath('images').absolute()
    logger.info('secondaryDir = {}'.format(secondaryDir))
    
    secondaryPattern = args.secondaryPattern
    logger.info('secondaryPattern = {}'.format(secondaryPattern))
    
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    
    main(primaryDir=primaryDir,
         primaryPattern=primaryPattern,
         operator=operator,
         secondaryDir=secondaryDir,
         secondaryPattern=secondaryPattern,
         outDir=outDir)