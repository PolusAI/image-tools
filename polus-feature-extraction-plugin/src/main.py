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
    parser.add_argument('--embeddedpixelsize', dest='embeddedpixelsize', type=str,
                        help='Embedded pixel size if present', required=False)
    parser.add_argument('--pixelsPerunit', dest='pixelsPerunit', type=float,
                        help='Pixels per unit', required= False)
    parser.add_argument('--unitLength', dest='unitLength', type=str,
                        help='Units of length', required= False)
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
    #features
    features = args.features.split(',')
    logger.info('features = {}'.format(features))
    #csvfile
    csvfile = args.csvfile
    logger.info('csvfile = {}'.format(csvfile))
    #embedded pixel size
    embeddedpixelsize = args.embeddedpixelsize
    logger.info('embeddedpixelsize = {}'.format(embeddedpixelsize))
    #unit length
    unitLength = args.unitLength
    logger.info('unitLength = {}'.format(unitLength))
    #pixels per unit
    pixelsPerunit = args.pixelsPerunit
    logger.info('pixels per unit = {}'.format(pixelsPerunit))
    #intensity image
    intDir = args.intDir
    logger.info('intDir = {}'.format(intDir))
    #pixel distance to calculate neighbors
    pixelDistance = args.pixelDistance
    logger.info('pixelDistance = {}'.format(pixelDistance))
    #Labeled image
    segDir = args.segDir
    logger.info('segDir = {}'.format(segDir))
    #directory to save output files
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    logger.info("Started")
    image_convert = ConvertImage(segDir,intDir)
    
    df,filenames= image_convert.convert_tiled_tiff(features, csvfile, outDir, pixelDistance, embeddedpixelsize, unitLength, pixelsPerunit)
    #call csv function to save as a single file
    if csvfile == 'singlecsv':
        csv_file= Df_Csv_single(df, outDir)
        csv_final = csv_file.csvfilesave()
        del csv_final,df,filenames

    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
    
    
