import argparse, logging, subprocess, time, multiprocessing, re, imagesize

import numpy as np

from pathlib import Path

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
STITCH_LINE = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

def get_number(s):
    """ Check that s is number
    
    In this plugin, heatmaps are created only for columns that contain numbers. This
    function checks to make sure an input value is able to be converted into a number.
    Inputs:
        s - An input string or number
    Outputs:
        value - Either float(s) or False if s cannot be cast to float
    """
    try:
        return int(s)
    except ValueError:
        return s

def _parse_stitch(stitchPath,imagePath,timepointName=False):
    """ Load and parse image stitching vectors
    
    This function creates a list of file dictionaries that include the filename and
    pixel position and dimensions within a stitched image. It also determines the
    size of the final stitched image and the suggested name of the output image based
    on differences in file names in the stitching vector.

    Inputs:
        stitchPath - A path to stitching vectors
        imagePath - A path to tiled tiff images
        timepointName - Use the vector timeslice as the image name
    Outputs:
        out_dict - Dictionary with keys (width, height, name, filePos)
    """

    # Initialize the output
    out_dict = { 'width': int(0),
                 'height': int(0),
                 'name': '',
                 'filePos': []}

    # Set the regular expression used to parse each line of the stitching vector
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"

    # Get a list of all images in imagePath
    images = [p.name for p in Path(imagePath).iterdir()]

    # Open each stitching vector
    fpath = str(Path(stitchPath).absolute())
    name_pos = {}
    with open(fpath,'r') as fr:

        # Read the first line to get the filename for comparison to all other filenames
        line = fr.readline()
        stitch_groups = re.match(line_regex,line)
        stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}
        name = stitch_groups['file']
        name_ind = [i for i in range(len(name))]
        fr.seek(0) # reset to the first line

        # Read each line in the stitching vector
        for line in fr:
            # Read and parse values from the current line
            stitch_groups = re.match(line_regex,line)
            stitch_groups = {key:get_number(val) for key,val in zip(STITCH_VARS,stitch_groups.groups())}
            
            # If an image in the vector doesn't match an image in the collection, then skip it
            if stitch_groups['file'] not in images:
                continue

            # Get the image size
            stitch_groups['width'], stitch_groups['height'] = imagesize.get(str(Path(imagePath).joinpath(stitch_groups['file']).absolute()))
            if out_dict['width'] < stitch_groups['width']+stitch_groups['posX']:
                out_dict['width'] = stitch_groups['width']+stitch_groups['posX']
            if out_dict['height'] < stitch_groups['height']+stitch_groups['posY']:
                out_dict['height'] = stitch_groups['height']+stitch_groups['posY']

            # Set the stitching vector values in the file dictionary
            out_dict['filePos'].append(stitch_groups)

            # Determine the difference between first name and current name
            if not timepointName:
                for i in name_ind:
                    if name[i] != stitch_groups['file'][i]:
                        if i not in name_pos.keys():
                            name_pos[i] = set()
                            name_pos[i].update([get_number(stitch_groups['file'][i])])
                            name_pos[i].update([get_number(name[i])])
                        else:
                            name_pos[i].update([get_number(stitch_groups['file'][i])])
    
    # Generate the output file name
    # NOTE: This should be rewritten later to determine numeric values rather than position values.
    #       Output file names should be 
    indices = sorted(name_pos.keys())
    if timepointName:
        global_regex = ".*global-positions-([0-9]+).txt"
        name = re.match(global_regex,Path(stitchPath).name).groups()[0]
        name += '.ome.tif'
        out_dict['name'] = name
    elif len(indices) > 0:
        out_dict['name'] = name[0:indices[0]]
        minvals = []
        maxvals = []
        for v,i in enumerate(indices):
            if len(minvals)==0:
                out_dict['name'] += '<'
            minvals.append(min(name_pos[i]))
            maxvals.append(max(name_pos[i]))
            if i == indices[-1] or indices[v+1] - i > 1:
                out_dict['name'] += ''.join([str(ind) for ind in minvals])
                out_dict['name'] += '-'
                out_dict['name'] += ''.join([str(ind) for ind in maxvals])
                out_dict['name'] += '>'
                if i ==  indices[-1]:
                    out_dict['name'] += name[indices[-1]+1:]
                else:
                    out_dict['name'] += name[indices[v]+1:indices[v+1]]
                minvals = []
                maxvals = []
    else:
        out_dict['name'] = name

    return out_dict

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Assemble images from a single stitching vector.')
    parser.add_argument('--stitchPath', dest='stitchPath', type=str,
                        help='Complete path to a stitching vector', required=False)
    parser.add_argument('--imgPath', dest='imgPath', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--timesliceNaming', dest='timesliceNaming', type=str,
                        help='Use timeslice number as image name', required=False)
    parser.add_argument('--vectorInMetadata', dest='vectorInMetadata', type=str,
                        help='Use stitching vectors stored in the metadata', required=False)

    # Parse the arguments
    args = parser.parse_args()
    vectorInMetadata = args.vectorInMetadata == 'true'
    logger.info('vectorInMetadata: {}'.format(vectorInMetadata))
    imgPath = args.imgPath
    outDir = args.outDir
    logger.info('outDir: {}'.format(outDir))
    timesliceNaming = args.timesliceNaming == 'true'
    logger.info('timesliceNaming: {}'.format(timesliceNaming))
    if vectorInMetadata:
        stitchPath = str(Path(imgPath).parent.joinpath('metadata_files').absolute())
    else:
        stitchPath = args.stitchPath
        if stitchPath == None:
            ValueError('If vectorInMetadata==False, then stitchPath must be defined')

    # Get a list of stitching vectors
    try:
        vectors = [str(p.absolute()) for p in Path(stitchPath).iterdir() if p.is_file() and "".join(p.suffixes)=='.txt']
    except FileNotFoundError:
        # Workaround for WIPP bug
        if Path(stitchPath).name == 'metadata_files':
            stitchPath = str(Path(imgPath).joinpath('metadata_files').absolute())
            imgPath = str(Path(imgPath).joinpath('images').absolute())
            vectors = [str(p.absolute()) for p in Path(stitchPath).iterdir() if p.is_file() and "".join(p.suffixes)=='.txt']
        
    logger.info('imgPath: {}'.format(imgPath))
    logger.info('stichPath: {}'.format(stitchPath))
    vectors.sort()

    # Variables for image building processes
    img_processes = []
    img_paths = []

    for v in vectors:
        # Check to see if the file is a stitching vector
        if not Path(v).name.startswith('img-global-positions'):
            continue
        
        # Parse the stitching vector
        logger.info('Analyzing vector: {}'.format(Path(v).name))
        outvals = _parse_stitch(v,imgPath,timesliceNaming)
        logger.info('Building image: {}'.format(outvals['name']))
        logger.info('Output image size (width, height): {},{}'.format(outvals['width'],outvals['height']))

        # Variables for tile building processes
        processes = []
        running_processes = []
        completed_processes = []
        pnum = 0
        ptotal = np.ceil(outvals['width']/10240) * np.ceil(outvals['height']/10240)
        ptotal = 1/ptotal * 100

        # Assemble the images
        logger.info('Generating tiles...')
        for x in range(0, outvals['width'], 10240):
            X_range = min(x+10240,outvals['width']) # max x-pixel index in the assembled image
            for y in range(0, outvals['height'], 10240):
                Y_range = min(y+10240,outvals['height']) # max y-pixel index in the assembled image

                # If there are num_cores - 1 processes running, wait until one finishes
                if len(processes) >= multiprocessing.cpu_count()-1-len(img_processes) and len(processes)+len(img_processes) > 0:
                    free_process = -1
                    while free_process<0:
                        for process in range(len(processes)):
                            if processes[process].poll() is not None:
                                free_process = process
                                break

                        # Check the img_processes
                        for process in range(len(img_processes)):
                            if img_processes[process].poll() is not None:
                                Path(img_paths[process][0]).rmdir()
                                logger.info('Finished building image: {}'.format(Path(img_paths[process][1]).name))
                                del img_processes[process]
                                del img_paths[process]
                                free_process = -2
                                break
                        if free_process == -2:
                            break

                        # Only check intermittently to free up processing power
                        if free_process<0:
                            time.sleep(3)
                    if free_process >= 0:
                        pnum += 1
                        logger.info("{:.2f}% complete...".format(pnum*ptotal))
                        del processes[free_process]
                        completed_processes.append(running_processes[free_process])
                        del running_processes[free_process]

                # create the arg dictionary for the process that will be spawned
                regex = ".*-global-positions-([0-9]+).txt"
                timeframe = re.match(regex,v).groups()[0]
                tile_outdir = str(Path(outDir).joinpath(timeframe).absolute())
                Path(tile_outdir).mkdir(exist_ok=True)
                inputs = [imgPath,tile_outdir,v,x,y,X_range,Y_range]
                
                # Spawn a stack building process and record the starting time
                processes.append(
                    subprocess.Popen(
                        "python3 tile.py --imgPath {} --outDir {} --stitchPath {} --x {} --y {} --X_range {} --Y_range {}".format(*inputs),
                        shell=True
                    )
                )
                running_processes.append(inputs)

        # Wait for all processes to finish
        while len(processes)>0:
            free_process = -1
            while free_process<0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                # Only check intermittently to free up processing power
                if free_process<0:
                    time.sleep(3)
            pnum += 1
            logger.info("{:.2f}% complete...".format(pnum*ptotal))
            del processes[free_process]
            completed_processes.append(running_processes[free_process])
            del running_processes[free_process]
    
        # Start a process to build the output image
        refImg = str(Path(imgPath).joinpath(outvals['filePos'][0]['file']).absolute())
        outFile = '"' + str(Path(outDir).joinpath(outvals['name']).absolute()) + '"'
        inputs = [tile_outdir,outFile,refImg,outvals['width'],outvals['height']]
        img_processes.append(
            subprocess.Popen(
                "python3 assemble.py --imgPath {} --outFile {} --refImg {} --width {} --height {}".format(*inputs),
                shell=True
            )
        )
        img_paths.append([tile_outdir,outFile])

    # Wait for all image building processes to finish
    while len(img_processes)>0:
        for process in range(len(img_processes)):
            if img_processes[process].poll() is not None:
                Path(img_paths[process][0]).rmdir()
                logger.info('Finished building image: {}'.format(Path(img_paths[process][1]).name))
                del img_processes[process]
                del img_paths[process]
                break
    logger.info('Finished all processes successfully!')