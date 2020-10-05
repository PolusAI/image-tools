import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import filepattern
import os
from filepattern import FilePattern as fp
import numpy as np


# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='The type of image, image or segmentation', required=True)
    parser.add_argument('--meshes', dest='meshes', type=str2bool, nargs='?',const=True,
                        default=False,help='True or False')

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.image_pattern
    imagetype = args.image_type
    boolmesh = args.meshes

    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image pattern = {}'.format(imagepattern))
    logger.info('meshes = {}'.format(boolmesh))
    # logger.info('images are stacked by variable(s) {}'.format(stack_by))
    
    # Get list of images that we are going to through
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".Labels.ome.tif"]
    images.sort()

    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each stack is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    stack_count = 1
    im_count = 1
    for image in images:
        # val_instack = vals_instack[stack_count-1]
        # val_ofstack = vals_stackby[stack_count-1]
        # heightofstack = organizedheights[stack_count-1]
        # newoutput = Path(output_dir + directoryfiles[stack_count - 1])
        if len(processes) >= multiprocessing.cpu_count()-1 and len(processes)>0:
            free_process = -1
            while free_process<0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                time.sleep(3)
                
            pnum += 1
            logger.info("Finished Z stack process {} of {} in {}s!".format(pnum,len(images),time.time() - process_timer[free_process]))
            del processes[free_process]
            del process_timer[free_process]
        try:
            processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir '{}' --outDir '{}' --pyramidType '{}' --imageNum '{}' --imagepattern '{}' --image '{}' --imagetype {} --meshes {}".format(input_dir,
                                                                                                                                                output_dir,
                                                                                                                                                pyramid_type,
                                                                                                                                                im_count,
                                                                                                                                                imagepattern,
                                                                                                                                                image.name,
                                                                                                                                                imagetype,
                                                                                                                                                boolmesh),
                                                                                                                                            shell=True))
        except:
            raise Exception("Previous process in build-pyramid.py input is wrong")
            exit()
        im_count += 1
        process_timer.append(time.time())
        stack_count = stack_count + 1
    
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
        logger.info("Finished stack process {} of {} in {}s!".format(pnum,len(images),time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
