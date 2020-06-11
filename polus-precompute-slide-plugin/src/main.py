import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import filepattern
import os
from filepattern import FilePattern as fp
import pandas
import numpy as np

def stitchingfunction(input_dir, output_dir, pyramid_type, stitchingvectors):

    df = {}
    inputlocations = []
    stacks = {}
    stitchingfiles = sorted(os.listdir(stitchingvectors))
    numberofstacks = len(stitchingfiles)
    i = 0
    for file in stitchingfiles:
        with open(stitchingvectors + file) as readfile:
            stitchedvector = []
            for line in readfile:
                line = (line.split(";"))[0]
                line = (line.split(":"))[1]
                line = line.strip()
                stitchedvector.append(line)
            df[i] = stitchedvector
        readfile.close()
        i = i + 1
    df = pandas.DataFrame.from_dict(df)

    inputfiles = sorted(os.listdir(input_dir))
    inputthreeD = [z for z in inputfiles]
    inputthreeD = np.asarray(inputthreeD)
    for item in inputfiles:
        position = df[df==item].stack().index.tolist()[0]
        colpos = position[1]
        if colpos in stacks:
            stacks[colpos].append(item)
        else:
            stacks[colpos] = [item]
        inputlocations.append(position)
    numberofstacks = len(stacks)

    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each stack is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    stack_count = 1
    im_count = 1
    for i in range(0, numberofstacks):
        heightofstack = len(stacks[i])
        stackofinterest = " ".join(stacks[i])
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
                        format(pnum,numberofstacks,time.time() - process_timer[free_process], im_count, sum(stacks[i])))
            del processes[free_process]
            del process_timer[free_process]
            
        processes.append(subprocess.Popen("python3 build_pyramid_vectors.py --inpDir '{}' --outDir '{}' --pyramidType '{}' --imageNum '{}' --stackheight '{}' --stackcount '{}' --stackofinterest '{}'".format(input_dir,
                                                                                                                                            output_dir,
                                                                                                                                            pyramid_type,
                                                                                                                                            im_count,
                                                                                                                                            heightofstack,
                                                                                                                                            i,
                                                                                                                                            stackofinterest),
                                                                                                                                        shell=True))
        im_count = (i)*(heightofstack) 
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
        logger.info("Finished stack process {} of {} in {}s!".format(pnum,numberofstacks,time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    
    logger.info("Finished stack process {} of {} in {}s!".format(numberofstacks,numberofstacks,time.time() - process_timer[0]))
    logger.info("Finished all processes!")
            

def stackbyfunction(input_dir, output_dir, pyramid_type, stack_by, imagepattern):
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
    numberofstacks = len(organizedheights)

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
    for iterate in fp.iterate(fpobject, group_by=stack_by):
        val_instack = vals_instack[stack_count - 1]
        heightofstack = organizedheights[stack_count -1]
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
            
        processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir '{}' --outDir '{}' --pyramidType '{}' --imageNum '{}' --stackheight '{}' --stackby '{}' --varsinstack '{}' --valinstack {} --imagepattern '{}' --stackcount '{}'".format(input_dir,
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
    parser.add_argument('--stitchingVector', dest='stiching_vector', type=str,
                        help='Textfiles that describe which images should be grouped', required=False)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.image_pattern
    stack_by = args.stack_by
    stitchingvectors = args.stiching_vector

    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image pattern = {}'.format(imagepattern))
    logger.info('images are stacked by variable {}'.format(stack_by))
    logger.info('stitching vectors file: {}'.format(stitchingvectors))

    # Getting all images in directory
    # image_path = Path(input_dir)
    # images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    # images.sort()

    if stack_by != None:
        stackbyfunction(input_dir, output_dir, pyramid_type, stack_by, imagepattern)
        # Get Variables of Images
        
    if stitchingvectors != None:
        stitchingfunction(input_dir, output_dir, pyramid_type, stitchingvectors)

if __name__ == "__main__":
    main()
