import argparse, logging, os, time, filepattern
from pathlib import Path
import tensorflow as tf
from functions import *


def main(inpDir:Path, 
         pattern:str,
         groupBy:str,
         outDir:Path,         
         ):       
        starttime= time.time()
        if pattern is None:
            logger.info(
                    "No filepattern was provided so filepattern infer pattern from the input files"
                )
            try:
                pattern = filepattern.infer_pattern([f.name for f in inpDir.iterdir()])
            except ValueError:
                logger.error(
                    "Could not infer a filepattern from the input files, "
                    + "and no filepattern was provided."
                )
                raise

        assert inpDir.exists(), logger.info("Input directory does not exist")
        count=0
        fp = filepattern.FilePattern(inpDir,pattern)
        imagelist = len([f for f in fp])

        for f in fp(group_by=groupBy):
            count += 1
            file = f[0]['file'].name
            logger.info(f'Label image: {file}')
            db = Discard_borderobjects(inpDir, outDir, file)
            db.discard_borderobjects()
            relabel_img, _ = db.relabel_sequential()
            db.save_relabel_image(relabel_img)
            logger.info(f'Saving {count}/{imagelist} Relabelled image with discarded objects: {file}') 
            logger.info('Finished all processes')
            endtime = (time.time() - starttime)/60
            logger.info(f'Total time taken to process all images: {endtime}')

#Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
if tf.test.gpu_device_name():
    logger.info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    logger.info('Default Device is CPU')

# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Discard Border Objects Plugin')    
#   # Input arguments

parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True
    )
parser.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        help="Filepattern regex used to parse image files",
        required=False
    )
parser.add_argument(
    "--groupBy",
    dest="groupBy",
    type=str,
    help="Select a parameter to select images in specific order",
    required=False
)                    
#  # Output arguments
parser.add_argument('--outDir',
    dest='outDir',
    type=str,
    help='Output directory',
    required=True
    )   
# # Parse the arguments
args = parser.parse_args()
inpDir = Path(args.inpDir)

if (inpDir.joinpath('images').is_dir()):
    inputDir = inpDir.joinpath('images').absolute()
logger.info('inpDir = {}'.format(inpDir))
pattern = args.pattern
logger.info("pattern = {}".format(pattern))
groupBy = args.groupBy
logger.info("groupBy = {}".format(groupBy))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))

if __name__=="__main__":
    main(inpDir=inpDir,
         pattern=pattern,
         groupBy=groupBy,
         outDir=outDir
         )