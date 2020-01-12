import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import javabridge as jutil
from utils import _get_higher_res, ChunkEncoder, CHUNK_SIZE

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main():
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Generate a precomputed slice for Polus Volume Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    
    # Get a list of all images in a directory
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    images.sort()
    
    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each pyramid is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    for image in images:
        if len(processes) >= multiprocessing.cpu_count()-1 and len(processes)>0:
            free_process = -1
            while free_process<0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                time.sleep(3)
                
            pnum += 1
            logger.info("Finished process {} of {} in {}s!".format(pnum,len(images),time.time() - process_timer[free_process]))
            del processes[free_process]
            del process_timer[free_process]
        processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir {} --outDir {} --image {}".format(input_dir,
                                                                                                               output_dir,
                                                                                                               image.name),
                                                                                                               shell=True))
        process_timer.append(time.time())
    
    # Wait for all processes to finish
    while len(processes)>1:
        free_process = -1
        while free_process<0:
            for process in range(len(processes)):
                if processes[process].poll() is not None:
                    free_process = process
                    break
            time.sleep(3)
        pnum += 1
        logger.info("Finished process {} of {} in {}s!".format(pnum,len(images),time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    
    logger.info("Finished process {} of {} in {}s!".format(len(images),len(images),time.time() - process_timer[0]))
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()