
import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import filepattern
import itertools
import os


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
    parser.add_argument('--imagepattern', dest='image_pattern', type=str,
                        help='Filepattern of the images in input', required=True)
    parser.add_argument('--stackby', dest='stack_by', type=str,
                        help='Variable that the images get stacked by', required=True)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.image_pattern
    stack_by = args.stack_by

    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image pattern = {}'.format(imagepattern))
    logger.info('images are stacked by variable {}'.format(stack_by))
    
    # Get a list of all images in a directory
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    images.sort()

    # Get the Height of the Stack
    regex = filepattern.get_regex(pattern = imagepattern)
    regexzero = regex[0]
    regexone = regex[1]
    vars_instack = ''
    for item in regexone:
        if item == stack_by:
            continue
        else:
            vars_instack = vars_instack + item
    
    allfiles = filepattern.parse_directory(input_dir, pattern=imagepattern, var_order=vars_instack+stack_by)
    all_varlists = [allfiles[1][item] for item in allfiles[1]]
    all_combos = list(itertools.product(*all_varlists))
    
    commonfiles = filepattern.parse_directory(input_dir, pattern=imagepattern, var_order=vars_instack)
    common_varlists = [commonfiles[1][item] for item in commonfiles[1]]
    common_combos = list(itertools.product(*common_varlists))

    heightofstack = int(len(all_combos)/len(common_combos))

    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each pyramid is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    stack_count = 1
    im_count = 1
    for commontups in common_combos:
        vals_instack = ''
        for tup in commontups:
            if vals_instack == '':
                vals_instack = str(tup)
            else:
                vals_instack = vals_instack + " " + str(tup)

        if len(processes) >= multiprocessing.cpu_count()-1 and len(processes)>0:
            free_process = -1
            while free_process<0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                time.sleep(3)
                
            pnum += 1
            logger.info("Finished stack process {} of {} in {}s (Stacked {} images out of {} images)!".
                        format(pnum,len(common_combos),time.time() - process_timer[free_process], im_count, len(images)))
            del processes[free_process]
            del process_timer[free_process]
            
        processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir {} --outDir {} --pyramidType {} --imageNum {} --stackheight {} --stackby {} --varsinstack {} --valsinstack {} --imagepattern {}".format(input_dir,
                                                                                                                                            output_dir,
                                                                                                                                            pyramid_type,
                                                                                                                                            im_count,
                                                                                                                                            heightofstack,
                                                                                                                                            stack_by,
                                                                                                                                            vars_instack,
                                                                                                                                            vals_instack,
                                                                                                                                            imagepattern),
                                                                                                                                        shell=True))
        im_count = (stack_count)*(heightofstack) 
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
        logger.info("Finished stack process {} of {} in {}s!".format(pnum,len(common_combos),time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    
    logger.info("Finished stack process {} of {} in {}s!".format(len(common_combos),len(common_combos),time.time() - process_timer[0]))
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()