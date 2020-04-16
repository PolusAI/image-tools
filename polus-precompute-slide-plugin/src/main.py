import logging, argparse, time, multiprocessing, subprocess
from pathlib import Path
import os
import filepattern

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
    parser.add_argument('--imagepattern', type=str, 
                        help='The pattern that images are formated', required=True)

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pyramid_type = args.pyramid_type
    imagepattern = args.imagepattern
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('pyramid_type = {}'.format(pyramid_type))
    logger.info('image patterns = {}'.format(imagepattern))

    # Get a list of all images in a directory
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    filename_images = [os.path.basename(i) for i in images]
    images.sort()

    #Getting imagepattern values
    getregex = filepattern.get_regex(imagepattern)
    variablepatterns = getregex[0]
    variablelist = getregex[1]
    lettervariables = ''.join(str(var) for var in getregex[1])

    stackbyvariable = 'c'
    dontstack = 'rz'
    groupimagesbyvariables = len(variablelist) * [0]
    rzuniques = []
    imageslist = []
    print("Number of Variables", groupimagesbyvariables)
    print("IMAGE PATTERN PARAMETERS")
    print("Pattern", variablepatterns)
    print("Variables", lettervariables)
    print("IMAGE NAMES")
    for origitem in images:
        item = os.path.basename(origitem)
        dictionaryvals = filepattern.parse_filename(item, pattern= imagepattern)
        # print(i, ") ", item)
        # print(i, ") ", dictionaryvals)
        dontstackvars = [dictionaryvals[char] for char in dontstack]
        if dontstackvars in rzuniques:
            idx = rzuniques.index(dontstackvars) 
            imageslist[idx].append(origitem)
        else:
            rzuniques.append(dontstackvars)
            imageslist.append([origitem])   
    for item in imageslist:
        print(item)

    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    
    # Build one pyramid for each image in the input directory
    # Each pyramid is built within its own process, with a maximum number of processes
    # equal to number of cpus - 1.
    
    for imglist in imageslist:
        stacknumber = 1
        im_count = 1
        firstimage = imglist[0]
        for image in imglist:
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
                
            processes.append(subprocess.Popen("python3 build_pyramid.py --inpDir {} --outDir {} --pyramidType {} --image {} --imageNum {} --stackNum {} --firstimage {}".format(input_dir,
                                                                                                                                                output_dir,
                                                                                                                                                pyramid_type,
                                                                                                                                                '"' + image.name + '"',
                                                                                                                                                im_count, 
                                                                                                                                                stacknumber, 
                                                                                                                                                firstimage),
                                                                                                                                                shell=True))
            im_count += 1
            process_timer.append(time.time())
        stacknumber += 1
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