from bfio import BioReader, BioWriter
import argparse, logging, os
from pathlib import Path
from utils import *

#Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Implementation of Image Quality Metrics')    
#     # Input arguments
parser.add_argument('--inputDir', dest='inputDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
parser.add_argument('--minI', dest='minI', type=int,
                        help='Select the minimum Intensity value to normalized image', required=True)
parser.add_argument('--maxI', dest='maxI', type=int,
                        help='Select the maximum Intensity value to normalized image', required=True)
parser.add_argument('--scale', dest='scale', type=int,
                        help='Select the spatial scale to calculate the Focus score', required=True)
 # Output arguments
parser.add_argument('--filename', dest='filename', type=str,
                        help='Filename of the output CSV file', required=True)
parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output directory', required=True)   
# # Parse the arguments
args = parser.parse_args()
inputDir = Path(args.inputDir)

if (inputDir.joinpath('images').is_dir()):
    # switch to images folder if present
    inputDir = inputDir.joinpath('images').absolute()
logger.info('inputDir = {}'.format(inputDir))
minI = int(args.minI)
logger.info('minI= {}'.format(minI))
maxI= int(args.maxI)
logger.info('maxI = {}'.format(maxI))
scale= int(args.scale)
logger.info('scale = {}'.format(scale))
filename = str(args.filename)
logger.info('filename = {}'.format(filename))
outDir = Path(args.outDir)
logger.info('outDir = {}'.format(outDir))

def main(inputDir: Path,
         minI: int,
         maxI: int,
         scale: int,
         filename: str,
         outDir: Path,
         ) -> None:

        # Validate method
    if minI not in [0, -1]:
        logger.error('Invalid minI value!!')
    if maxI not in [0, -1]:
        logger.error('Invalid maxI value!!')
    if scale <=  1:
        logger.error('Selected Scale value should be greater than 1')

    for i, image in enumerate(os.listdir(inputDir)):
        logger.info(f'Processing image: {image}')
        imgpath = os.path.join(inputDir, image)
        if image.endswith('.ome.tif'):
            logger.debug(f'Initializing BioReader for {image}')
            br = BioReader(imgpath)
            img = br.read().squeeze()
            qc = Image_quality(inputDir,img, minI, maxI, scale)
            img = qc.normalize_intensities()
            labels, magnitude, powersum = qc.rps()
            powerlogslope = qc.power_spectrum() 
            imagepath, imagename = imgpath, image
            img = qc.normalize_intensities()
            focus_score = qc.Focus_score()
            median_LocalFocus, mean_LocalFocus =qc.local_Focus_score()
            maximum_percent, minimum_percent = qc.saturation_calculation()
            brisque_score = qc.brisque_calculation()
            sharpness_score = qc.sharpness_calculation()
            correlation_score, dissimilarity_score = qc.calculate_correlation_dissimilarity()
            header= ['Image_Path', 'Image_FileName', 
            'Image_FocusScore', 'Image_Median_LocalFocusScore', 'Image_Mean_LocalFocusScore',
            'Image_maximumI_percent', 'Image_minimumI_percent', 'Image_BrisqueScore', 
            'Image_Sharpness', 'Image_PowerLogLogSlope', 'Correlation', 'Dissimilarity']
            datalist = [str(imagepath), str(imagename),
            focus_score,  median_LocalFocus, mean_LocalFocus, maximum_percent, minimum_percent, 
            brisque_score, sharpness_score, powerlogslope, correlation_score, dissimilarity_score]
            os.chdir(outDir)
            logger.info('Saving Output CSV file')
            write_csv(header, datalist, filename)
            logger.info("Finished all processes!")                  
    return       
if __name__=="__main__":    
    main(inputDir=inputDir,
         minI=minI,
         maxI=maxI,
         scale=scale,
         filename=filename,
         outDir=outDir)





