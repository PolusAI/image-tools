import argparse, logging
from features_single import runMain

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Plugin to create ROI image from tiff file.')
    # Input arguments
    parser.add_argument('--GTDir', dest='GTDir', type=str,
                        help='Ground truth feature collection to be processed by this plugin.', required=True)
    parser.add_argument('--PredDir', dest='PredDir', type=str,
                        help='Predicted feature collection to be processed by this plugin.', required=True)
    parser.add_argument('--outFileFormat', dest='outFileFormat', type=str, default="false",
                        help='Boolean to save output file as csv. Default is lz4.', required=False)
    parser.add_argument('--combineLabels', dest='combineLabels', type=str, default="false",
                        help='Boolean to calculate number of bins for histogram by combining GT and Predicted Labels. Default is using GT labels only.', required=False)
    parser.add_argument('--filePattern', dest='filePattern', type=str, default=".*",
                        help='Filename pattern to filter data.', required=False)
    parser.add_argument('--singleCSV', dest='singleCSV', type=str, default="true",
                        help='Boolean to save output file as a single csv. Default is true.', required=False)
    # Output arguments                    
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    GTDir = args.GTDir
    logger.info('GTDir = {}'.format(GTDir))
    PredDir = args.PredDir
    logger.info('PredDir = {}'.format(PredDir))
    outFileFormat = args.outFileFormat == 'true' or args.outFileFormat == 'True'
    logger.info('outFileFormat = {}'.format(outFileFormat))
    combineLabels = args.combineLabels == 'true' or args.combineLabels == 'True'
    logger.info('combineLabels = {}'.format(combineLabels))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    singleCSV = args.singleCSV == 'true' or args.singleCSV == 'True'
    logger.info('combineLabels = {}'.format(singleCSV))
    
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    runMain(GTDir, PredDir, outFileFormat, combineLabels, filePattern, singleCSV, outDir)