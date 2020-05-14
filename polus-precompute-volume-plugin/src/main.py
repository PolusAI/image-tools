import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import filepattern
import os
from filepattern import FilePattern as fp


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

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.image_pattern

    # Getting all images in directory
    # image_path = Path(input_dir)
    # images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    # images.sort()

    # Get Variables of Images
    regex = filepattern.get_regex(pattern = imagepattern)
    regexzero = regex[0]
    regexone = regex[1]
    vars_instack = ''
    stack_by = ''
    for item in regexone:
        if (item == 'x' or item == 'y') or item == 'z':
            vars_instack = vars_instack + item
        else:
            stack_by = stack_by + item


    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image pattern = {}'.format(imagepattern))
    logger.info('images are stacked by variable(s) {}'.format(stack_by))
    
    # Get list of images that we are going to through
    logger.info('Getting the images...')

    fpobject = fp(input_dir, pattern=imagepattern)

    organizedheights = []
    vals_instack = []
    vals_stackby = []
    directoryfiles = []

    for item in fp.iterate(fpobject, group_by=vars_instack):
        organizedheights.append(len(item))
        vals_inonestack = [[]] * len(item)
        i = 0
        for filesitem in item:
            vals_inonestack[i] = ""
            vals_stackone = ""
            directoryname = ""
            for char in vars_instack:
                if vals_inonestack[i] == "":
                    vals_inonestack[i] = str(filesitem[char])
                else:
                    vals_inonestack[i] = vals_inonestack[i] + " " + str(filesitem[char])
            for char in stack_by:
                if vals_stackone == "":
                    vals_stackone = str(filesitem[char])
                    directoryname = char + str(filesitem[char])
                else:
                    vals_stackone = vals_stackone + " " + str(filesitem[char])
                    directoryname = directoryname + "_" + char + str(filesitem[char])
            i = i + 1
        vals_instack.append(vals_inonestack)
        vals_stackby.append(vals_stackone)
        directoryfiles.append(directoryname)
        newdirectory = Path(output_dir).joinpath(directoryname)
        newdirectory.mkdir()
            
    numberofstacks = len(organizedheights)

    logger.info("Directory Name: {}".format(directoryfiles))
    logger.info("Height of the {} Stacks: {}".format(len(organizedheights), organizedheights))
    # heightofstack = int(len(all_combos)/len(common_combos))
    logger.info("Different Stack Variables of {}: {}".format(vars_instack, vals_instack))

    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each stack is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    stack_count = 1
    im_count = 1
    for iterate in fp.iterate(fpobject, group_by=vars_instack):
        val_instack = vals_instack[stack_count-1]
        heightofstack = organizedheights[stack_count-1]
        logger.info("{} stack values: {}".format(stack_count, val_instack))
        logger.info("{} stack height: {}".format(stack_count, heightofstack))
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
                        format(pnum,numberofstacks,time.time() - process_timer[free_process], im_count, sum(organizedheights)))
            del processes[free_process]
            del process_timer[free_process]
            
        processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir {} --outDir {} --pyramidType {} --imageNum {} --stackheight {} --stackby {} --varsinstack {} --valinstack {} --imagepattern {} --stackcount {} >>textfile.txt".format(input_dir,
                                                                                                                                            output_dir,
                                                                                                                                            pyramid_type,
                                                                                                                                            im_count,
                                                                                                                                            heightofstack,
                                                                                                                                            stack_by,
                                                                                                                                            vars_instack,
                                                                                                                                            val_instack,
                                                                                                                                            imagepattern,
                                                                                                                                            stack_count - 1),
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
        logger.info("Finished stack process {} of {} in {}s!".format(pnum,numberofstacks,time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    
    logger.info("Finished stack process {} of {} in {}s!".format(numberofstacks,numberofstacks,time.time() - process_timer[0]))
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
