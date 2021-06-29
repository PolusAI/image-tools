from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, multiprocessing, filepattern
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from multiprocessing import cpu_count
import time

# size of chunks to process (chunk_size x chunk_size)
chunk_size = 4096

# Thread lock for file writing
lock = Lock()

def write_tile(br,bw,X,Y,Z,C):
    """Read and write one tile of a multichannel image
    
    This function is intended to be run inside of a thread. Since data is sent
    to the JVM and the GIL is released, reading images can be performed in
    parallel. Writing still requires a lock.

    Args:
        br ([bfio.BioReader]): BioReader to be read from
        bw ([bfio.BioWriter]): BioWriter to be written to
        X ([list]): Range of x-values to read/write
        Y ([list]): Range of y-values to grab/write
        C ([list]): Channel to write
    """
    
    # Connect to the JVM
    jutil.attach()
    
    # Read the image
    image = br.read_image(X=X,Y=Y,Z=Z)
    
    # Lock the thread and write the image
    with lock:
        bw.write_image(image,X=[X[0]],Y=[Y[0]],Z=[Z[0]],C=C)
    
    # Detach from the JVM
    jutil.detach()

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Create multichannel, ome-tif from an image collection.')
    
    # Input arguments
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--channelOrder', dest='channelOrder', type=str,
                        help='Order in which to place the channels (e.g. 1,3,2)', required=True)
    
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(Path(inpDir).resolve()))
    channelOrder = [int(c) for c in str(args.channelOrder).split(',')]
    logger.info('channelOrder = {}'.format(channelOrder))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
        
        # Create a filepattern object to auto sort images
        fp = filepattern.FilePattern(inpDir,filePattern)
        
        # Loop through files in inpDir image collection and process
        for files in fp.iterate(group_by='c'):
            
            # Get the filenames in the channel order
            paths = []
            for c in channelOrder:
                for file in files:
                    if file['c'] == c:
                        paths.append(file)
                        break
            
            # make sure that files were found in the current loop
            if len(paths)==0:
                continue
            
            # Initialize the output file
            br = BioReader(paths[0]['file'])
            file_name = filepattern.output_name(filePattern,
                                                paths,
                                                {c:paths[0][c] for c in fp.variables if c != 'c'})
            logger.info('Writing: {}'.format(file_name))
            bw = BioWriter(str(Path(outDir).joinpath(file_name)),
                           metadata=br.read_metadata())
            del br
            
            # Modify the metadata to make sure channels are written correctly
            bw.num_c(len(paths))
            bw._metadata.image().Pixels.channel_count = bw.num_c()
            
            # Process the data in tiles
            threads = []
            count = 0
            total = bw.num_c() * bw.num_z() * \
                    (bw.num_x()//chunk_size + 1) * (bw.num_y()//chunk_size + 1)
            with ThreadPoolExecutor(cpu_count()) as executor:
                for c,file in enumerate(paths):
                    br = BioReader(file['file'])
                    C = [c]
                    first_image = True
                    for z in range(br.num_z()):
                        Z = [z,z+1]
                        for x in range(0,br.num_x(),chunk_size):
                            X = [x,min([x+chunk_size,br.num_x()])]
                            for y in range(0,br.num_y(),chunk_size):
                                Y = [y,min([y+chunk_size,br.num_y()])]
                                threads.append(executor.submit(write_tile,br,bw,X,Y,Z,C))
                                
                                # Bioformats requires the first tile to be written
                                # before any other tile is written
                                if first_image or len(threads)==cpu_count():
                                    threads[0].result()
                                    del threads[0]
                                    first_image = False
                                    count += 1
                                    if count % 5 == 0:
                                        logger.info('{:.2f}% finished...'.format(100*count/total))

                        # Threads have to complete before moving to the next z
                        wait(threads)
                        count += len(threads)
                        logger.info('{:.2f}% finished...'.format(100*count/total))
                        threads = []
                        
            bw.close_image()
        
    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the javabridge')
        jutil.kill_vm()
