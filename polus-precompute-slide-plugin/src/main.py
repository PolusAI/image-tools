import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import filepattern
import os
from filepattern import FilePattern as fp
import numpy as np

def checkprocesslen(processes, process_timer, pnum):
    if len(processes) >= multiprocessing.cpu_count()-1 and len(processes)>0:
        free_process = -1
        while free_process<0:
            for process in range(len(processes)):
                if processes[process].poll() is not None:
                    free_process = process
                    break
            time.sleep(3)
        
        pnum += 1
        logger.info("Finished process {} in {}s!".format(pnum, time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

def whilewaitprocess(processes, process_timer, pnum):
    while len(processes)>1:
        free_process = -1
        while free_process<0:
            for process in range(len(processes)):
                if processes[process].poll() is not None:
                    free_process = process
                    break
            time.sleep(3)
        pnum += 1
        logger.info("Finished process {} in {}s!".format(pnum, time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]
    

def zslicefunction(input_dir, output_dir, pyramid_type, imagetype):

    # Get a list of all images in a directory
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir()]
    images.sort()
    
    return images

def stackbyfunction(input_dir, stack_by, imagepattern):

    imagepattern = str(imagepattern)
    regex = filepattern.get_regex(pattern = imagepattern)
    regexzero = regex[0]
    regexone = regex[1]
    vars_instack = ''
    for item in regexone:
        if item == stack_by:
            continue
        else:
            vars_instack = vars_instack + item
    
    # Get list of images that we are going to through
    logger.info('Getting the images...')

    fpobject = fp(input_dir, imagepattern, var_order=vars_instack)
    organizedheights = []
    vals_instack = []
    for item in fp.iterate(fpobject, group_by=stack_by):
        organizedheights.append(len(item))
        vals_instack.append("")
        for filesitem in item:
            for char in vars_instack:
                if vals_instack[-1] == "":
                    vals_instack[-1] = str(filesitem[char])
                else:
                    vals_instack[-1] = vals_instack[-1] + " " + str(filesitem[char])
            break

    logger.info("Height of the {} Stacks: {}".format(len(organizedheights), organizedheights))
    
    # heightofstack = int(len(all_combos)/len(common_combos))
    logger.info("Different Stack Variables of {}: {}".format(vars_instack, vals_instack))

    if len(organizedheights) == 0:
        raise ValueError("There are no images to stack (image pattern may be incorrect)")

    return fpobject, vals_instack, vars_instack, organizedheights


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
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--stackby', dest='stack_by', type=str,
                        help='Variable that the images get stacked by', required=False)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='Either a image or a segmentation', required=True)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.image_pattern
    stack_by = args.stack_by
    image_type = args.image_type

    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image_type = {}'.format(image_type))
    logger.info('image pattern = {} ({})'.format(imagepattern, type(imagepattern)))
    logger.info('images are stacked by variable {}'.format(stack_by))

    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0

    # variables to keep track of processes
    im_count = 1
    stack_count = 1

    processstring = " --inpDir '{}' --outDir '{}' --pyramidType '{}' --imageType '{}'".format(input_dir, 
                                                                                            output_dir,
                                                                                            pyramid_type,
                                                                                            image_type)

    # if imagepattern or image type specified then images get stacked by the stack_by variables
    if stack_by != None or imagepattern != None:
        fpobject, vals_instack, vars_instack, organizedheights = stackbyfunction(input_dir, stack_by, imagepattern)
        numberofstacks = len(organizedheights)
        processstring = "python3 build_pyramid_stitched.py" + processstring + " --stackby " + stack_by + " --imagepattern '" + imagepattern + "' --varsinstack " + vars_instack

        # Build one pyramid for each image in the input directory
        # Each stack is built within its own process, with a maximum number of processes
        # equal to number of cpus - 1.
        for iterate in fp.iterate(fpobject, group_by=stack_by):
            val_instack = vals_instack[stack_count - 1]
            heightofstack = organizedheights[stack_count -1]
            checkprocesslen(processes, process_timer, pnum)
            processstring_plus = processstring + " --imageNum '{}' --stackheight '{}' --valinstack {} --stackcount '{}'".format(im_count,
                                                                                                                                heightofstack,
                                                                                                                                val_instack,
                                                                                                                                stack_count-1)

            print(processstring_plus)
            processes.append(subprocess.Popen(processstring_plus, shell=True))
            im_count = (stack_count)*(heightofstack) 
            process_timer.append(time.time())
            stack_count = stack_count + 1
        
        # Wait for all processes to finish
        whilewaitprocess(processes, process_timer, pnum)
        processes[0].wait()
        
        logger.info("Finished stack process {} of {} in {}s!".format(numberofstacks,numberofstacks,time.time() - process_timer[0]))
        logger.info("Finished all processes!")


    else:
        images = zslicefunction(input_dir, output_dir, pyramid_type, image_type)
        processstring = "python3 build_pyramid_zslices.py" + processstring

        for image in images:

            appendtoprocess = ' --image {} --imageNum {}'.format('"' + image.name + '"', im_count)
            processstring_plus = processstring + appendtoprocess

            checkprocesslen(processes, process_timer, pnum)
                
            processes.append(subprocess.Popen(processstring_plus, shell=True))
            im_count += 1
            process_timer.append(time.time())
        
        whilewaitprocess(processes, process_timer, pnum)
        processes[0].wait()
        
        logger.info("Finished process {} of {} in {}s!".format(len(images),len(images),time.time() - process_timer[0]))
        logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
