from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, imagesize, re, difflib
import numpy as np
from pathlib import Path
from filepattern import FilePattern

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
STITCH_LINE = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

def _parse_stitch(stitchPath,imagePath):
    """ Load and parse image stitching vectors
    
    This function adds keys to the FilePattern object (fp) that indicate image positions
    extracted from the stitching vectors found at the stitchPath location.

    As the stitching vector is parsed, images in the stitching vector are analyzed to
    determine what the height and width of the assembled image should be.

    Inputs:
        fp - A FilePattern object
        stitchPath - A path to stitching vectors
    Outputs:
        width - Suggested width of the output image
        height - Suggested height of the output image
        name - Suggested output name based on input image names
    """

    # Set the regular expression used to parse each line of the stitching vector
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"

    # Get a list of all images in imagePath
    images = [p.name for p in Path(imagePath).iterdir()]

    # initialize the output values
    width = int(0)
    height = int(0)
    name_pos = set() # positions in the filename strings that are different in at least one files

    # Open each stitching vector
    fpath = str(Path(stitchPath).absolute())
    stitch_images = []
    with open(fpath,'r') as fr:

        # Read the first line to get the filename for comparison to all other filenames
        line = fr.readline()
        stitch_groups = re.match(line_regex,line)
        stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}
        name = stitch_groups['file']
        fr.seek(0) # reset to the first line

        # Read each line in the stitching vector
        for line in fr:
            # Read and parse values from the current line
            stitch_groups = re.match(line_regex,line)
            stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}
            
            # If an image in the vector doesn't match an image in the collection, then skip it
            if stitch_groups['file'] not in images:
                continue

            # Get the image size
            stitch_groups['width'], stitch_groups['height'] = imagesize.get(Path())
            if width < current_image['width']+current_image['posX']:
                width = current_image['width']+current_image['posX']
            if height < current_image['height']+current_image['posY']:
                height = current_image['height']+current_image['posY']

            # Set the stitching vector values in the file dictionary
            stitch_images.append(stitch_groups)

            # Determine the difference between first name and current name, update the index
            name_pos.update([i for i,d in enumerate(difflib.ndiff(name, stitch_groups['file'])) if d[0] not in '? '])

    return width,height,name

if __name__=="__main__":
    stitchPath = '/media/schaubnj/ExtraDrive1/SLAS_Demo/stitch_vectors/stitching-vector-5df4fa53172bfe0009b57fde-img-global-positions-1.txt'
    imgPath = '/media/schaubnj/ExtraDrive1/SLAS_Demo/p001'

    fp = FilePattern(imgPath,'.*.ome.tif')

    print(fp.files)

    # # Initialize the logger
    # logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    #                     datefmt='%d-%b-%y %H:%M:%S')
    # logger = logging.getLogger("main")
    # logger.setLevel(logging.INFO)

    # # Setup the argument parsing
    # logger.info("Parsing arguments...")
    # parser = argparse.ArgumentParser(prog='main', description='A scalable image assembling plugin.')
    # parser.add_argument('--filePattern', dest='filePattern', type=str,
    #                     help='Filename pattern used to separate data', required=True)
    # parser.add_argument('--inpDir', dest='inpDir', type=str,
    #                     help='Input image collection to be processed by this plugin', required=True)
    # parser.add_argument('--outDir', dest='outDir', type=str,
    #                     help='Output collection', required=True)
    
    # # Parse the arguments
    # args = parser.parse_args()
    # filePattern = args.filePattern
    # logger.info('filePattern = {}'.format(filePattern))
    # inpDir = args.inpDir
    # logger.info('inpDir = {}'.format(inpDir))
    # outDir = args.outDir
    # logger.info('outDir = {}'.format(outDir))
    
    
    # # Start the javabridge with proper java logging
    # logger.info('Initializing the javabridge...')
    # log_config = Path(__file__).parent.joinpath("log4j.properties")
    # jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    # # Get all file names in inpDir image collection
    # inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    # # Loop through files in inpDir image collection and process
    # for f in inpDir_files:
    #     # Load an image
    #     br = BioReader(Path(inpDir).joinpath(f))
    #     image = np.squeeze(br.read_image())

    #     # initialize the output
    #     out_image = np.zeros(image.shape,dtype=br._pix['type'])

    #     """ Do some math and science - you should replace this """
    #     out_image = awesome_math_and_science_function(image)

    #     # Write the output
    #     bw = BioWriter(Path(outDir).joinpath(f),metadata=br.read_metadata())
    #     bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z,1,1)))
    
    
    # # Close the javabridge
    # logger.info('Closing the javabridge...')
    # jutil.kill_vm()
    