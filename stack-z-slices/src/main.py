import argparse, time, logging, subprocess, multiprocessing
from pathlib import Path
from utils import _parse_files_p,_parse_files_xy,_parse_regex

if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--filePattern', dest='file_pattern', type=str,
                        help='The output directory for ome.tif files', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_pattern = args.file_pattern
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('file_pattern = {}'.format(file_pattern))
    
    # Parse the filename pattern
    regex,variables = _parse_regex(file_pattern)
    
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
    
    # Cycle through image timepoint variables
    ts = [t for t in files.keys()] # sorted list of timepoint values
    ts.sort()
    for t in ts:
        # Cycle through image channel variables
        cs = [c for c in files[t].keys()] # sorted list of channel values
        cs.sort()
        for c in cs:
            if 'p' not in variables:
                # Cycle through image x positions
                xs = [x for x in files[t][c].keys()] # sorted list of x-positions
                xs.sort()
                for x in xs:
                    # Cycle through image y positions
                    ys = [y for y in files[t][c][x].keys()] # sorted list of y-positions
                    ys.sort()
                    for y in ys:
                        # If there are num_cores - 1 processes running, wait until one finishes
                        if len(processes) >= multiprocessing.cpu_count()-1:
                            free_process = False
                            while not free_process:
                                for process in range(len(processes)):
                                    if processes[process].poll() is not None:
                                        free_process = process
                            pnum += 1
                            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(xs)*len(ys),time.time() - process_timer[free_process]))
                            del processes[free_process]
                            del process_timer[free_process]
                        
                        # Spawn a pyramid building process and record the starting time
                        processes.append(subprocess.Popen("python3 merge_layers.py --inpDir {} --outDir {} --regex {} --X {} --Y {} --C {} --T {}".format(input_dir,
                                                                                                                                                          output_dir,
                                                                                                                                                          file_pattern,
                                                                                                                                                          x,
                                                                                                                                                          y,
                                                                                                                                                          c,
                                                                                                                                                          t),
                                                                                                                                                          shell=True))
                        process_timer.append(time.time())
            else:
                # Cycle through image sequence positions
                ps = [p for p in files[t][c].keys()]
                ps.sort()
                for p in ps:
                    # If there are num_cores - 1 processes running, wait until one finishes
                    if len(processes) >= multiprocessing.cpu_count()-1:
                        free_process = False
                        while not free_process:
                            for process in range(len(processes)):
                                if processes[process].poll() is not None:
                                    free_process = process
                        pnum += 1
                        logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(ps),time.time() - process_timer[free_process]))
                        del processes[free_process]
                        del process_timer[free_process]
                    
                    # Spawn a pyramid building process and record the starting time
                    processes.append(subprocess.Popen("python3 merge_layers.py --inpDir {} --outDir {} --regex {} --P {} --C {} --T {}".format(input_dir,
                                                                                                                                               output_dir,
                                                                                                                                               file_pattern,
                                                                                                                                               p,
                                                                                                                                               c,
                                                                                                                                               t),
                                                                                                                                               shell=True))
                    process_timer.append(time.time())
    
    while len(processes)>1:
        free_process = False
        while not free_process:
            for process in range(len(processes)):
                if processes[process].poll() is not None:
                    free_process = process
        pnum += 1
        if 'p' not in variables:
            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(xs)*len(ys),time.time() - process_timer[free_process]))
        else:
            logger.info("Finished process {} of {} in {}s!".format(pnum,len(ts)*len(cs)*len(ps),time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()
    logger.info("Finished all processes, closing...")
    