import argparse, logging, math, filepattern, queue
from bfio import BioReader, BioWriter
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

import pprint

# Global variable to scale number of processing threads dynamically
free_threads = 0

def _merge_layers(input_files,output_path):
    
    global free_threads
    
    # Initialize the output file
    br = BioReader(input_files[0]['file'],max_workers)
    bw = BioWriter(str(Path(output_dir).joinpath(output_file).absolute()),metadata=br.read_metadata())
    bw.num_z(Z = len(zs))
    del br
    
    # Load each image and save to the volume file
    for z,i in zip(zs,range(len(zs))):
        br = BioReader(str(Path(input_dir).joinpath(input_files[z][0]).absolute()))
        bw.write_image(br.read_image(),Z=[i,i+1])
        del br
    
    # Close the output image and delete
    bw.close_image()
    del bw

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
    logger.info(f'input_dir = {input_dir}')
    logger.info(f'output_dir = {output_dir}')
    logger.info(f'file_pattern = {file_pattern}')
    
    max_threads = cpu_count()
    logger.info(f'max_threads: {max_threads}')
    
    # create the filepattern object
    fp = filepattern.FilePattern(input_dir,file_pattern)
    
    # Create thread handling variables
    
    for files in fp.iterate(group_by='z'):
        
        print(list(math.ceil(d/1024) for d in BioReader.image_size(files[0]['file'])))
        
        print(filepattern.output_name(file_pattern,files,{'p': files[0]['p']}))
        
        pprint.pprint(files)
        
        quit()
    