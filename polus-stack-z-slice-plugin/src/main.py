import argparse, time, logging, subprocess, multiprocessing
from pathlib import Path
from utils import _parse_files_p,_parse_files_xy,_parse_fpattern

if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Compile individual tiled tiff images into a single volumetric tiled tiff.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with tiled tiff files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--filePattern', dest='file_pattern', type=str,
                        help='A filename pattern specifying variables in filenames.', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_pattern = args.file_pattern
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('file_pattern = {}'.format(file_pattern))
    
    # Parse the filename pattern
    regex,variables = _parse_fpattern(file_pattern)
    
    # Parse files based on regex
    if 'p' not in variables:
        logger.info('Using x and y as the position variable if present...')
        files = _parse_files_xy(input_dir,regex,variables)
    else:
        logger.info('Using p as the position variable...')
        files = _parse_files_p(input_dir,regex,variables)
    
    # Initialize variables for process management
    processes = []
    process_timer = []
    pnum = 0
    
    # Cycle through image replicate variables
    rs = [r for r in files.keys()]
    rs.sort() # sorted list of replicate values
    for r in rs:
        # Cycle through image timepoint variables
        ts = [t for t in files[r].keys()] 
        ts.sort() # sorted list of timepoint values
        for t in ts:
            # Cycle through image channel variables
            cs = [c for c in files[r][t].keys()] 
            cs.sort() # sorted list of channel values
            for c in cs:
                if 'p' not in variables:
                    # Cycle through image x positions
                    xs = [x for x in files[r][t][c].keys()] 
                    xs.sort() # sorted list of x-positions
                    for x in xs:
                        # Cycle through image y positions
                        ys = [y for y in files[r][t][c][x].keys()] 
                        ys.sort() # sorted list of y-positions
                        for y in ys:
                            # If there are num_cores - 1 processes running, wait until one finishes
                            if len(processes) >= multiprocessing.cpu_count()-1 and len(processes) > 0:
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
                                logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(xs)*len(ys),time.time() - process_timer[free_process]))
                                del processes[free_process]
                                del process_timer[free_process]
                            
                            # Spawn a stack building process and record the starting time
                            processes.append(subprocess.Popen("python3 merge_layers.py --inpDir {} --outDir {} --regex {} --X {} --Y {} --C {} --T {} --R {}".format(input_dir,
                                                                                                                                                                     output_dir,
                                                                                                                                                                     file_pattern,
                                                                                                                                                                     x,
                                                                                                                                                                     y,
                                                                                                                                                                     c,
                                                                                                                                                                     t,
                                                                                                                                                                     r),
                                                                                                                                                                     shell=True))
                            process_timer.append(time.time())
                else:
                    # Cycle through image sequence positions
                    ps = [p for p in files[r][t][c].keys()]
                    ps.sort()
                    for p in ps:
                        # If there are num_cores - 1 processes running, wait until one finishes
                        if len(processes) >= multiprocessing.cpu_count()-1 and len(processes) > 0:
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
                            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(ps),time.time() - process_timer[free_process]))
                            del processes[free_process]
                            del process_timer[free_process]
                        
                        # Spawn a stack building process and record the starting time
                        processes.append(subprocess.Popen("python3 merge_layers.py --inpDir {} --outDir {} --regex {} --P {} --C {} --T {} --R {}".format(input_dir,
                                                                                                                                                          output_dir,
                                                                                                                                                          file_pattern,
                                                                                                                                                          p,
                                                                                                                                                          c,
                                                                                                                                                          t,
                                                                                                                                                          r),
                                                                                                                                                          shell=True))
                        process_timer.append(time.time())
    
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
        if 'p' not in variables:
            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(xs)*len(ys),time.time() - process_timer[free_process]))
        else:
            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(ps),time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    logger.info("Finished all processes, closing...")
    