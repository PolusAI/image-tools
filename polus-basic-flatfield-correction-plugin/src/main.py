import argparse, logging, multiprocessing, subprocess, time
from pathlib import Path
from filepattern import FilePattern
import basic
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count, Queue

# Global variable to scale number of processing threads dynamically
max_threads = max([cpu_count()//2,1])

# Set logger delay times
process_delay = 30     # Delay between updates within _merge_layers

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main():
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate flatfield information from an image collection.')

    """ Define the arguments """
    parser.add_argument('--inpDir',
                        dest='inpDir',
                        type=str,
                        help='Path to input images.',
                        required=True)
    parser.add_argument('--darkfield',
                        dest='darkfield',
                        type=str,
                        help='If true, calculate darkfield contribution.',
                        required=False)
    parser.add_argument('--photobleach',
                        dest='photobleach',
                        type=str,
                        help='If true, calculates a photobleaching scalar.',
                        required=False)
    parser.add_argument('--filePattern',
                        dest='file_pattern',
                        type=str,
                        help='Input file name pattern.',
                        required=False)
    parser.add_argument('--outDir',
                        dest='output_dir',
                        type=str,
                        help='The output directory for the flatfield images.',
                        required=True)

    """ Get the input arguments """
    args = parser.parse_args()
    fpath = args.inpDir
    """Checking if there is images subdirectory"""
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        fpath=Path(args.inpDir).joinpath('images')
    get_darkfield = str(args.darkfield).lower() == 'true'
    output_dir = Path(args.output_dir).joinpath('images')
    output_dir.mkdir(exist_ok=True)
    metadata_dir = Path(args.output_dir).joinpath('metadata_files')
    metadata_dir.mkdir(exist_ok=True)
    file_pattern = args.file_pattern
    get_photobleach = str(args.photobleach).lower() == 'true'

    logger.info('input_dir = {}'.format(fpath))
    logger.info('get_darkfield = {}'.format(get_darkfield))
    logger.info('get_photobleach = {}'.format(get_photobleach))
    logger.info('inp_regex = {}'.format(file_pattern))
    logger.info('output_dir = {}'.format(output_dir))

    fp = FilePattern(fpath,file_pattern)
    
    processes = []
    with ProcessPoolExecutor(max_threads) as executor:
    
        for files in fp(group_by='xyp'):
            
            processes.append(executor.submit(basic.basic,files,output_dir,get_darkfield,get_photobleach))
        
        logger.info(f'max_threads = {max_threads}')
        logger.info(f'len(processes) = {len(processes)}')
        done, not_done = wait(processes,timeout=0)

        while len(not_done) > 0:

            logger.info('Total Progress: {:6.2f}%'.format(100*len(done)/len(processes)))

            done, not_done = wait(processes,timeout=process_delay)

    logger.info('Progress: {:6.3f}%'.format(100))

if __name__ == "__main__":
    main()