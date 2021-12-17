import argparse, subprocess, logging, time
from pathlib import Path

BATCH_SIZE = 20

def main():
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Parse the inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    args = parser.parse_args()
    
    # Input and output directory
    input_dir = args.input_directory
    logger.info("input_dir: {}".format(input_dir))
    output_dir = args.output_directory
    logger.info("output_dir: {}".format(output_dir))
    
    # Get a list of images
    files = [str(f.absolute()) for f in Path(input_dir).iterdir()]
    
    # Loop over images, 20 at a time
    for ind in range(0,len(files),BATCH_SIZE):
        logger.info('{:.2f}% complete...'.format(100*ind/len(files)))
        batch = ','.join(files[ind:min([ind+BATCH_SIZE,len(files)])])
        process = subprocess.Popen("python3 segment.py --batch {} --outDir {}".format(batch,output_dir),shell=True)
        while process.poll() == None:
            time.sleep(1) # Put the main process to sleep inbetween checks, free up some processing power
    logger.info('100% complete...')

if __name__ == "__main__":
    main()
    