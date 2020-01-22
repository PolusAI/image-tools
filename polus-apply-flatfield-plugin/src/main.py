from bfio.bfio import BioReader, BioWriter
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
from filepattern import get_regex,FilePattern,VARIABLES,val_variables

# Variables that will be grouped for the purposes of applying a flatfield
GROUPED = 'xyzp'

if __name__=="__main__":
    ''' Initialize the logger '''
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    # Initialize the argument parser
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Apply a flatfield algorithm to a collection of images.')
    parser.add_argument('--darkPattern', dest='darkPattern', type=str,
                        help='Filename pattern used to match darkfield files to image files', required=False)
    parser.add_argument('--ffDir', dest='ffDir', type=str,
                        help='Image collection containing brightfield and/or darkfield images', required=True)
    parser.add_argument('--brightPattern', dest='brightPattern', type=str,
                        help='Filename pattern used to match brightfield files to image files', required=True)
    parser.add_argument('--imgDir', dest='imgDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--imgPattern', dest='imgPattern', type=str,
                        help='Filename pattern used to separate data and match with flatfied files', required=True)
    parser.add_argument('--photoPattern', dest='photoPattern', type=str,
                        help='Filename pattern used to match photobleach files to image files', required=False)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    darkPattern = args.darkPattern
    logger.info('darkPattern = {}'.format(darkPattern))
    ffDir = args.ffDir
    # catch the case that ffDir is the output within a workflow
    if Path(ffDir).joinpath('images').is_dir():
        ffDir = str(Path(ffDir).joinpath('images').absolute())
    logger.info('ffDir = {}'.format(ffDir))
    brightPattern = args.brightPattern
    logger.info('brightPattern = {}'.format(brightPattern))
    imgDir = args.imgDir
    logger.info('imgDir = {}'.format(imgDir))
    imgPattern = args.imgPattern
    logger.info('imgPattern = {}'.format(imgPattern))
    photoPattern = args.photoPattern
    logger.info('photoPattern = {}'.format(photoPattern))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    ''' Argument validation and error checking'''
    # Get the variables from the file patterns, if they existed
    ff_regex, ff_variables = get_regex(brightPattern)
    img_regex, img_variables = get_regex(imgPattern)
    if darkPattern != None and darkPattern!='':
        dark_regex, dark_variables = get_regex(darkPattern)
    if photoPattern != None and photoPattern!='':
        photo_regex, photo_variables = get_regex(photoPattern)

    # Validate the variables
    val_variables(ff_variables)
    val_variables(img_variables)
    if darkPattern != None and darkPattern!='':
        val_variables(dark_variables)
    if photoPattern != None and photoPattern!='':
        val_variables(photo_variables)

    for v in VARIABLES:
        if v in GROUPED:
            # Check brightfield/darkfield/photobleach patterns do not contain xyzp
            assert v not in ff_variables, 'Variable {} cannot be in the brightfield pattern.'
            if darkPattern != None and darkPattern!='':
                assert v not in dark_variables, 'Variable {} cannot be in the darkfield pattern.'
            if photoPattern != None and photoPattern!='':
                assert v not in photo_variables, 'Variable {} cannot be in the photobleach pattern.'
            continue
        # If variables are specified in the img pattern, it must be present in all other patterns
        if (v in ff_variables) != (v in img_variables): # v must be in both or neither (xor)
            logger.error('Variable {} is not in both ffPattern and imgPattern.'.format(v))
        if darkPattern != None and darkPattern!='':
            if (v in dark_variables) != (v in img_variables): # v must be in both or neither (xor)
                logger.error('Variable {} is not in both darkPattern and imgPattern.'.format(v))
        if photoPattern != None and photoPattern!='':
            if (v in photo_variables) != (v in img_variables): # v must be in both or neither (xor)
                logger.error('Variable {} is not in both photoPattern and imgPattern.'.format(v))

    ''' Start a process for each set of brightfield/darkfield/photobleach patterns '''
    # Create the FilePattern objects to handle file access
    ff_files = FilePattern(ffDir,brightPattern)
    if darkPattern != None and darkPattern!='':
        dark_files = FilePattern(ffDir,darkPattern)
    if photoPattern != None and photoPattern!='':
        photo_files = FilePattern(str(Path(ffDir).parents[0].joinpath('metadata').absolute()),photoPattern)

    # Initialize variables for process management
    processes = []
    process_timer = []
    pnum = 0
    total_jobs = len([p for p in ff_files.iterate()])
    
    # Loop through files in ffDir image collection and process
    base_pstring = "python3 apply_flatfield.py --inpDir {} --outDir {} --filepattern {}".format(imgDir,
                                                                                                outDir,
                                                                                                imgPattern)
    for f in ff_files.iterate():
        # If there are num_cores - 1 processes running, wait until one finishes
        if len(processes) >= multiprocessing.cpu_count()-1 and len(processes)>0:
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
            logger.info("Finished process {} of {} in {}s!".format(pnum,total_jobs,time.time() - process_timer[free_process]))
            del processes[free_process]
            del process_timer[free_process]

        if len(f)>1: # There should only be one brightfield image matching the file pattern
            ValueError("More than one brightfield image matched the file pattern: {}".format([ff['file']+' ' for ff in f]))
        
        ffpath = f[0]['file']
        R=f[0]['r']
        T=f[0]['t']
        C=f[0]['c']

        pstring = base_pstring + ' --brightfield "{}" --R {} --T {} --C {}'.format(ffpath,
                                                                               R,
                                                                               T,
                                                                               C)

        if darkPattern != None and darkPattern!='':
            pstring += ' --darkfield "{}"'.format(dark_files.get_matching(R=R,T=T,C=C)[0]['file'])
        if photoPattern != None and photoPattern!='':
            pstring += ' --photobleach "{}"'.format(photo_files.get_matching(R=R,T=T,C=C)[0]['file'])

        # Start the process    
        logger.info("Starting process [R,T,C]: [{},{},{}]".format(R,T,C))
        process_timer.append(time.time())
        processes.append(subprocess.Popen(pstring,shell=True))

    # If the process generator finishes and no processes were generated, throw an error so an empty image collection isn't generated
    if pnum==0:
        ValueError("No processes were generated. This could mean no brightfield images were found using the specified filepattern.")

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
        logger.info("Finished process {} of {} in {}s!".format(pnum,total_jobs,time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    logger.info("Finished all processes, closing...")