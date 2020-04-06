import argparse, logging, multiprocessing, subprocess, time,os
from pathlib import Path
from filepattern import FilePattern
# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def main():
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate flatfield information from an image collection.')

    """ Define the arguments """
    parser.add_argument('--inpDir',               # Name of the bucket
                        dest='inpDir',
                        type=str,
                        help='Path to input images.',
                        required=True)
    parser.add_argument('--darkfield',                  # Path to the data within the bucket
                        dest='darkfield',
                        type=str,
                        help='If true, calculate darkfield contribution.',
                        required=False)
    parser.add_argument('--photobleach',                  # Path to the data within the bucket
                        dest='photobleach',
                        type=str,
                        help='If true, calculates a photobleaching scalar.',
                        required=False)
    parser.add_argument('--inpRegex',                 # Output directory
                        dest='inp_regex',
                        type=str,
                        help='Input file name pattern.',
                        required=False)
    parser.add_argument('--outDir',                 # Output directory
                        dest='output_dir',
                        type=str,
                        help='The output directory for the flatfield images.',
                        required=True)

    """ Get the input arguments """
    args = parser.parse_args()
    fpath = args.inpDir
    if (os.path.isdir(Path(args.inpDir).joinpath('images'))):
        fpath=Path(args.inpDir).joinpath('images')
    get_darkfield = str(args.darkfield).lower() == 'true'
    output_dir = Path(args.output_dir).joinpath('images')
    output_dir.mkdir(exist_ok=True)
    metadata_dir = Path(args.output_dir).joinpath('metadata_files')
    metadata_dir.mkdir(exist_ok=True)
    inp_regex = args.inp_regex
    get_photobleach = str(args.photobleach).lower() == 'true'

    logger.info('input_dir = {}'.format(fpath))
    logger.info('get_darkfield = {}'.format(get_darkfield))
    logger.info('get_photobleach = {}'.format(get_photobleach))
    logger.info('inp_regex = {}'.format(inp_regex))
    logger.info('output_dir = {}'.format(output_dir))
    # Set up lists for tracking processes
    processes = []
    process_timer = []
    pnum = 0
    file = FilePattern(fpath, inp_regex)
    total_no = len(list(file.iterate(group_by='xyz')))
    for i in file.iterate(group_by='xyz'):
        if len(processes) >= multiprocessing.cpu_count() - 1:
            free_process = -1
            while free_process < 0:
                for process in range(len(processes)):
                    if processes[process].poll() is not None:
                        free_process = process
                        break
                # Wait between checks to free up some processing power
                time.sleep(3)
            pnum += 1
            logger.info("Finished process {} of {} in {}s!".format(pnum, total_no,
                                                                           time.time() - process_timer[free_process]))
            del processes[free_process]
            del process_timer[free_process]

        logger.info("Starting process [r,t,c]: [{},{},{}]".format(i[0]['r'], i[0]['t'], i[0]['c']))
        processes.append(subprocess.Popen(
                    "python3 basic.py --inpDir {} --outDir {} --darkfield {} --photobleach {} --inpRegex {} --R {} --T {} --C {}".format(
                        fpath,
                        args.output_dir,
                        get_darkfield,
                        get_photobleach,
                        inp_regex,
                        i[0]['r'],
                        i[0]['t'],
                        i[0]['c']),
                    shell=True))
        process_timer.append(time.time())
    while len(processes) > 1:
        free_process = -1
        while free_process < 0:
            for process in range(len(processes)):
                if processes[process].poll() is not None:
                    free_process = process
                    break
            # Wait between checks to free up some processing power
            time.sleep(3)
        pnum += 1
        logger.info("Finished process {} of {} in {}s!".format(pnum,total_no, time.time() - process_timer[free_process]))
        del processes[free_process]
        del process_timer[free_process]

    processes[0].wait()

    logger.info("Finished process {} of {} in {}s!".format(total_no,total_no, time.time() - process_timer[0]))
    logger.info("Finished all processes!")


if __name__ == "__main__":
    main()