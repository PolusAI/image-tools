import argparse, logging, math, filepattern, time, queue
from bfio import BioReader, BioWriter
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import Queue

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

# Global variable to scale number of processing threads dynamically
max_threads = max([cpu_count()//2,1])
available_threads = Queue(max_threads)
for _ in range(max_threads):
    available_threads.put(2)

# Set logger delay times
main_delay = 30        # Delay between updates for main process
process_delay = 10     # Delay between updates within _merge_layers

# length/width of the chunk each _merge_layers thread processes at once
# Number of useful threads is limited
chunk_size = 8192
useful_threads = (chunk_size // BioReader._TILE_SIZE) ** 2

def initialize_queue(processes):
    global available_threads
    available_threads = processes

def _merge_layers(input_files,output_path):
    global available_threads
    
    logger = logging.getLogger("merge")
    logger.setLevel(logging.INFO)

    # Grab some available threads
    active_threads = available_threads.get()
    logger.info(f'{output_path.name}: Starting with {active_threads} threads')

    # Get some basic info about the files to stack
    with BioReader(input_files[0]['file']) as br:

        # Get the physical z-distance if avaiable, set to physical x if not
        ps_z = br.ps_z
        if None in ps_z:
            ps_z = br.ps_x

        # Get the metadata
        metadata = br.metadata

        # Set the maximum number of available threads (don't hog if not needed)
        max_active_threads = math.ceil(br.x / br._TILE_SIZE) * math.ceil(br.y / br._TILE_SIZE)
        if max_active_threads > useful_threads:
            max_active_threads = useful_threads

    # Get the number of layers to stack
    z_size = 0
    for f in files:
        with BioReader(f['file']) as br:
            z_size += br.z

    # Create the output file within a context manager
    with BioWriter(output_path,metadata=metadata,max_workers=active_threads) as bw:

        # Update timer
        start = time.time()

        # Adjust the dimensions before writing
        bw.z = z_size
        bw.ps_z = ps_z

        # ZIndex tracking for the output file
        zi = 0

        # Threadpool to check for scaling opportunities during file IO
        write_thread = []
        with ThreadPoolExecutor(1) as executor:

            # Start stacking
            for file in input_files:

                with BioReader(file['file'],max_workers=active_threads) as br:

                    for z in range(br.z):

                        for xs in range(0,br.x,chunk_size):
                            xe = min([br.x,xs + chunk_size])

                            for ys in range(0,br.y,chunk_size):
                                ye = min([br.y,ys + chunk_size])

                                wait(write_thread)
                                write_thread = [executor.submit(bw.write,br[ys:ye,xs:xe,z:z+1],X=[xs],Y=[ys],Z=[zi])]

                        zi += 1

                now = time.time()
                if now - start > process_delay:
                    start = now
                    logger.info('{}: Progress {:6.2f}%'.format(output_path.name,100*zi/bw.z))

                if active_threads < max_active_threads and active_threads < useful_threads:
                    # See if more threads are available
                    new_threads = 0
                    try:
                        while active_threads + new_threads < useful_threads:
                            new_threads += available_threads.get(block=False)
                    except:
                        pass
                    
                    if new_threads > 0:
                        logger.info(f'{output_path.name}: Increasing threads from {active_threads} to {active_threads+new_threads}')
                        active_threads += new_threads
                        bw.max_workers = active_threads

    # Free the threads for other processes
    for _ in range(active_threads//2):
        available_threads.put(2)

if __name__ == "__main__":
    # Initialize the main thread logger
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_pattern = args.file_pattern
    logger.info(f'input_dir = {input_dir}')
    logger.info(f'output_dir = {output_dir}')
    logger.info(f'file_pattern = {file_pattern}')

    logger.info(f'max_threads: {max_threads}')

    # create the filepattern object
    fp = filepattern.FilePattern(input_dir,file_pattern)

    processes = []
    with ProcessPoolExecutor(max_threads,initializer=initialize_queue,initargs=(available_threads,)) as executor:
        count = 0
        for files in fp(group_by='z'):

            output_name = fp.output_name(files)
            output_file = output_dir.joinpath(output_name)

            processes.append(executor.submit(_merge_layers,files,output_file))

        done, not_done = wait(processes,timeout=0)

        while len(not_done):

            logger.info('Progress: {:7.3f}%'.format(100*len(done)/len(processes)))

            done, not_done = wait(processes,timeout=main_delay)

    logger.info('Progress: {:6.3f}%'.format(100))
