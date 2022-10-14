import argparse
import logging
import os
import pathlib 
import time
import re
import numpy as np
import multiprocessing
from typing import Optional, List
from functools import partial
from func import nyxus_func


def main(inpDir:str, 
         segDir:str, 
         outDir:pathlib.Path,
         filePattern:str,
         features:Optional[List[str]],
         neighborDist:Optional[float]=5.0,
         pixelPerMicron:Optional[float]=1.0
         ):

    starttime = time.time()

    
    if '{' and '}' in filePattern:
        filePattern  = re.sub(r"{.*}", '([0-9]+)', filePattern)

    ## Extracting unique image replicates using filepattern
    
    replicate = np.unique([re.search(filePattern, f).groups() for f in os.listdir(inpDir)])

    logger.info(f"Total number of replicates found: {replicate}")

    assert len(replicate) is not None, f'Replicate plate images not found! Please check the filepattern again: {replicate}'
    
    groupfeatures= ["ALL_INTENSITY",
              "ALL_MORPHOLOGY",
              "BASIC_MORPHOLOGY",
              "ALL_GLCM",
              "ALL_GLRLM",
              "ALL_GLSZM",
              "ALL_GLDM",
              "ALL_NGTDM",
              "ALL_EASY",
              "ALL"]

    ## Adding * to the start and end of nyxus group features 

    ft_gp = [f'*{x}*' for x in features.split(',') if x in groupfeatures]
    ft_only = [f'{x}' for x in features.split(',') if x not in groupfeatures]
    features = [*ft_only , *ft_gp]

    num_workers = max(multiprocessing.cpu_count() // 2, 2)

    with multiprocessing.Pool(processes=num_workers) as executor:
      
        executor.map(partial(nyxus_func, 
                        inpDir, 
                        segDir,
                        outDir,
                        filePattern,
                        features,
                        neighborDist,
                        pixelPerMicron
                        ), replicate)

        executor.close()
        executor.join()
    endtime = round((time.time() - starttime)/60, 3)
    logger.info(f"Time taken to finish nyxus feature extraction: {endtime} minutes!!!")

    return


# #Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Scaled Nyxus')    
#   # Input arguments

parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Input image collection to be processed by this plugin",
        required=True
    )

parser.add_argument(
        "--segDir",
        dest="segDir",
        type=str,
        help="Input label images",
        required=True
    )

parser.add_argument(
        "--filePattern",
        dest="filePattern",
        type=str,
        help="Pattern use to parse image filenames",
        required=True
    )

parser.add_argument(
        "--features",
        dest="features",
        type=str,
        help="Nyxus features to be extracted",
        default="ALL",
        required=False
    )

parser.add_argument(
        "--neighborDist",
        dest="neighborDist",
        type=float,
        help="Number of Pixels between Neighboring cells",
        default=5.0,
        required=False
    )
parser.add_argument(
        "--pixelPerMicron",
        dest="pixelPerMicron",
        type=float,
        help="Number of pixels per micrometer",
        default=1.0,
        required=False
    )                 
#  # Output arguments
parser.add_argument('--outDir',
    dest='outDir',
    type=str,
    help='Output directory',
    required=True
    )   
# # # Parse the arguments
args = parser.parse_args()
inpDir = args.inpDir
logger.info('inpDir = {}'.format(inpDir))
assert pathlib.Path(inpDir).exists(), f'Path of intensity images directory not found: {inpDir}'
segDir = args.segDir
logger.info('segDir = {}'.format(segDir))
assert pathlib.Path(segDir).exists(), f'Path of Labelled images directory not found: {segDir}'
filePattern = args.filePattern
logger.info("filePattern = {}".format(filePattern))
features = args.features
assert len(re.findall(r'{.*?}', filePattern)) == 1, f'Incorrect filePattern: {filePattern}'
logger.info("features = {}".format(features))
neighborDist = args.neighborDist
logger.info("neighborDist = {}".format(neighborDist))
pixelPerMicron = args.pixelPerMicron
logger.info("pixelPerMicron = {}".format(pixelPerMicron))

outDir = pathlib.Path(args.outDir)
logger.info('outDir = {}'.format(outDir))

if __name__=="__main__":

    main(inpDir=inpDir,
         segDir=segDir,
         outDir=outDir,
         filePattern=filePattern, 
         features=features,
         neighborDist=neighborDist,
         pixelPerMicron=pixelPerMicron    
         )
