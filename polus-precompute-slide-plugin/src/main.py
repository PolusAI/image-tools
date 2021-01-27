import logging, argparse, filepattern, bfio, utils, pathlib
from multiprocessing import Queue, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait

# Global variable to scale number of processing threads dynamically
max_threads = max([cpu_count()//2+1,1])
available_threads = Queue(max_threads)

# Set logger delay times
process_delay = 30     # Delay between updates within _merge_layers

for _ in range(max_threads):
    available_threads.put(2)

def initialize_queue(processes,pyramid_writer):
    global available_threads
    global PyramidWriter
    available_threads = processes
    PyramidWriter = pyramid_writer

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

PyramidWriter = {
    'Neuroglancer': utils.NeuroglancerWriter,
    'DeepZoom': utils.DeepZoomWriter
}

def pyramid_process(pyramid_type: str,
                    image_type: str,
                    base_dir: pathlib.Path,
                    image_path: pathlib.Path,
                    image_depth: int,
                    output_depth: int,
                    max_output_depth: int):
    
    active_threads = available_threads.get()
    
    pyramid_writer = PyramidWriter[pyramid_type](base_dir,
                                                 image_path,
                                                 image_depth=image_depth,
                                                 output_depth=output_depth,
                                                 max_output_depth=max_output_depth,
                                                 image_type=image_type,
                                                 num_threads=active_threads)
    
    logger.info(f'{pyramid_writer.image_path.name}: Starting process with {active_threads} threads')
    
    with ThreadPoolExecutor(1) as executor:
        threads = [executor.submit(pyramid_writer.write_slide)]
    
        done, not_done = wait(threads,timeout=0)
        
        while len(not_done) > 0:
            
            # See if more threads are available
            new_threads = 0
            try:
                while True:
                    new_threads += available_threads.get(block=False)
            except:
                pass
            
            if new_threads > 0:
                logger.info(f'{pyramid_writer.image_path.name}: Increasing threads from {active_threads} to {active_threads+new_threads}')
                active_threads += new_threads
                pyramid_writer.add_threads(new_threads)
            
            done, not_done = wait(threads,timeout=process_delay)
            
    for _ in range(active_threads//2):
        available_threads.put(2)
        
    logger.info(f'{pyramid_writer.image_path.name}: Finished!')
        
    return threads[0].result()

def main():
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='main',
        description='Generate a precomputed slice for Polus Volume Viewer.'
    )

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)
    parser.add_argument('--filePattern', dest='file_pattern', type=str,
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='Either a image or a segmentation, defaults to image', required=False)

    '''Parse arguments'''
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    # Validate the pyramid type
    pyramid_type = args.pyramid_type
    assert pyramid_type in ['Neuroglancer','DeepZoom'], "pyramidType must be one of ['Neuroglancer','DeepZoom']"
    
    # Use a universal filepattern if none is provided
    file_pattern = args.file_pattern
    if file_pattern == None:
        file_pattern = '.*'
        
    # Default image_type to 'image'
    image_type = args.image_type
    if image_type == None:
        image_type = 'image'
    assert image_type in ['image','segmentation'], 'imageType must be one of ["image","segmentation"]'

    logger.info('input_dir = %s', input_dir)
    logger.info('output_dir = %s',output_dir)
    logger.info('pyramid_type = %s',pyramid_type)
    logger.info('image_type = %s', image_type)
    logger.info('file_pattern = %s', file_pattern)
    logger.info('max concurrent processes = %s', max_threads)

    # Parse the input file directory
    fp = filepattern.FilePattern(input_dir,file_pattern)
    if 'z' in fp.variables and pyramid_type == 'Neuroglancer':
        group_by = 'z'
        logger.info('Stacking images by z-dimension for Neuroglancer precomputed format.')
    elif 't' in fp.variables and pyramid_type == 'DeepZoom':
        group_by = 't'
        logger.info('Creating time slices by t-dimension for DeepZoom format.')
    else:
        group_by = ''
        logger.info(f'Creating one pyramid for each image in {pyramid_type} format.')
    
    depth = 0
    depth_max = 0
    depth_delta = 0
    image_dir = ''
    
    processes = []
    with ProcessPoolExecutor(max_threads,
                             initializer=initialize_queue,
                             initargs=(available_threads,PyramidWriter)) as executor:
    
        for files in fp(group_by=group_by):
            
            # Create the output name for Neuroglancer format
            if pyramid_type == 'Neuroglancer':
                try:
                    image_dir = fp.output_name([file for file in files])
                except:
                    image_dir = files[0]['file'].name
                
                # Reset the depth for every iteration of Neuroglancer files
                depth = 0
                d_depth = 1
            
            pyramid_writer = None
            
            for file in files:
                
                with bfio.BioReader(file['file'],max_workers=1) as br:
                    
                    d_z = br.z
                    
                depth_max += d_z
                    
                for z in range(d_z):
                    
                    pyramid_args = {
                        'base_dir': output_dir.joinpath(image_dir),
                        'image_path': file['file'],
                        'image_depth': z,
                        'output_depth': depth,
                        'max_output_depth': depth_max
                    }
                    
                    processes.append(executor.submit(pyramid_process,
                                                     pyramid_type,
                                                     image_type,
                                                     **pyramid_args))
                    
                    depth += 1
                    
                    if pyramid_type == 'DeepZoom':
                        utils.DeepZoomWriter(**pyramid_args).write_info()
                        
            
            if pyramid_type == 'Neuroglancer':
                utils.NeuroglancerWriter(**pyramid_args).write_info()

        done, not_done = wait(processes,timeout=0)
        
        while len(not_done) > 0:
            
            done, not_done = wait(processes,timeout=process_delay)
            
            logger.info('Total Progress: {:6.2f}%'.format(100*len(done)/len(processes)))

if __name__ == "__main__":
    main()
