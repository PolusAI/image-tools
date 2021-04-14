from bfio.bfio import BioReader
import filepattern
import argparse, logging
import shutil
import typing
from multiprocessing import cpu_count
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def validate_and_copy(file: Path,
                      outDir: Path,
                      ) -> None:
    
    # Enter context manager to verify the file is a tiled tiff
    with BioReader(file['file'],backend='python') as br:
        
        shutil.copy2(file['file'],outDir.joinpath(file['file'].name))

def main(inpDir: Path,
         outDir: Path,
         ) -> None:
    """ Main execution function
    
    All functions in your code must have docstrings attached to them, and the
    docstrings must follow the Google Python Style:
    https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
    """
    
    
    pattern = ".*.ome.tif"
    fp = filepattern.FilePattern(inpDir,pattern)
    
    files = [f[0] for f in fp]
    
    threads = []
    
    with ThreadPoolExecutor(cpu_count()) as executor:
    
        for ind,file in enumerate(files):
            
            threads.append(executor.submit(validate_and_copy,file,outDir))
           
        done, not_done = wait(threads,timeout=0)
        
        logger.info('Copy progress: {:6.2f}%'.format(100*len(done)/len(threads)))
        
        while len(not_done) > 0:
            
            done, not_done = wait(threads,timeout=5)
            
            logger.info('Copy progress: {:6.2f}%'.format(100*len(done)/len(threads)))

if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Copies .ome.tif files with proper tile format from a generic data type to an image collection.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    fpath = Path(args.inpDir)
    if (fpath.joinpath('images').is_dir()):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    
    main(inpDir=inpDir,
         outDir=outDir)