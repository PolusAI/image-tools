from utility import ConvertImage, Df_Csv_single
import argparse, logging


 # Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a Feature Extraction plugin.')
    parser.add_argument('--features', dest='features', type=str,
                        help='Features to calculate', required=True)
    parser.add_argument('--csvfile', dest='csvfile', type=str,
                        help='Save csv as separate or single file', required=True)
    parser.add_argument('--units', dest='units', type=str,
                        help='Units for the features', required=True)
    parser.add_argument('--length_of_unit', dest='length_of_unit', type=str,
                        help='Length of the unit', required= False)
    parser.add_argument('--pixels_per_unit', dest='pixels_per_unit', type=int,
                        help='Pixels per unit', required= False)
    parser.add_argument('--labelimage', dest='labelimage', type=str,
                        help='Input image need to be labeled or not', required=True)
    parser.add_argument('--intDir', dest='intDir', type=str,
                        help='Intensity image collection', required=False)
    parser.add_argument('--pixelDistance', dest='pixelDistance', type=int,
                        help='Pixel distance to calculate the neighbors touching cells', required=False)
    parser.add_argument('--segDir', dest='segDir', type=str,
                        help='Segmented image collection', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    features = args.features.split(',')#features
    logger.info('features = {}'.format(features))
    csvfile = args.csvfile#csvfile
    logger.info('csvfile = {}'.format(csvfile))
    units = args.units#units
    logger.info('units = {}'.format(units))
    length_of_unit = args.length_of_unit#csvfile
    logger.info('length of unit = {}'.format(length_of_unit))
    pixels_per_unit = args.pixels_per_unit#csvfile
    logger.info('pixels per unit = {}'.format(pixels_per_unit))
    labelimage = args.labelimage#label image
    logger.info('labelimage = {}'.format(labelimage))
    intDir = args.intDir#intensity image
    logger.info('intDir = {}'.format(intDir))
    pixelDistance = args.pixelDistance#pixel distance to calculate neighbors
    logger.info('pixelDistance = {}'.format(pixelDistance))
    segDir = args.segDir#segmented image
    logger.info('segDir = {}'.format(segDir))
    outDir = args.outDir#directory to save output files
    logger.info('outDir = {}'.format(outDir))
    logger.info("Started")
    image_convert = ConvertImage(segDir,intDir)
    
    df,filenames= image_convert.convert_tiled_tiff(features, csvfile, labelimage, outDir, pixelDistance, units, pixels_per_unit, length_of_unit)
    #call csv function to save as a single file
    if csvfile == 'singlecsv':
        csv_file= Df_Csv_single(df, outDir)
        csv_final = csv_file.csvfilesave()
        del csv_final,df,filenames

    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
    
    