import argparse, logging, shutil
from pathlib import Path
from filepattern import FilePattern

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Create a new image collection that is a subset of an existing image collection.')
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    filePattern = args.filePattern
    if not filePattern.strip():
        filePattern = '.*'
    logger.info('filePattern = {}'.format(filePattern))
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Get all file names in inpDir image collection
    fp = FilePattern(inpDir,pattern=filePattern)

    # Loop through files in inpDir image collection and process
    for files in fp():
        if isinstance(files,list):
            # files is a list of file dictionaries
            for f in files:
                input_path = Path(f['file'])
                output_path = Path(outDir).joinpath(input_path.name)
                logger.info('Copying file: {}'.format(input_path.name))
                shutil.copy2(str(input_path.absolute()), str(output_path.absolute()))    
    logger.info('Finished copying all files!')
