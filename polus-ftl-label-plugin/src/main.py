import argparse, logging, subprocess, time, multiprocessing
from pathlib import Path

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Label objects in a 2d or 3d binary image.')
    parser.add_argument('--connectivity', dest='connectivity', type=str,
                        help='City block connectivity, must be less than or equal to the number of dimensions', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    connectivity = int(args.connectivity)
    logger.info('connectivity = {}'.format(connectivity))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Get all file names in inpDir image collection
    inpDir_files = [f for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    
    # Set up multiprocessing
    max_processes = max([multiprocessing.cpu_count()//3,1])
    processes = []
    
    # Loop through files in inpDir image collection and process
    for ind in range(0,len(inpDir_files),20):
        images = ','.join([str(f.absolute()) for f in inpDir_files[ind:max([ind+20,len(inpDir_files)])]])
        
        # Wait for a process to finish if we have too many running
        if len(processes) >= max_processes:
            free_process = -1
            while free_process<0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                # Only check intermittently to free up processing power
                if free_process<0:
                    time.sleep(3)
            del processes[free_process]
        
        # Create the inputs for the process
        inputs = [connectivity,
                  images,
                  ind//20+1,
                  outDir]
         
        # Start the process       
        processes.append(
            subprocess.Popen(
                "python3 label_images.py --connectivity {} --images {} --process {} --outDir {}".format(*inputs),
                shell=True
            )
        )
        
    while len(processes)>0:
        for process in range(len(processes)):
            if processes[process].poll() is not None:
                del processes[process]
                break
    