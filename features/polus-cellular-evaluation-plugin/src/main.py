import argparse, logging
from evaluate import evaluation

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
                        help='Ground truth input image collection to be processed by this plugin', required=True)
    parser.add_argument('--PredDir', dest='PredDir', type=str,
                        help='Predicted input image collection to be processed by this plugin', required=True)
    parser.add_argument('--inputClasses', dest='inputClasses', type=int,
                        help='Number of Classes', required=True)
    parser.add_argument('--individualData', dest='individualData', type=str, default="false",
                        help='Boolean to calculate individual image statistics. Default is false.', required=False)
    parser.add_argument('--individualSummary', dest='individualSummary', type=str, default="false",
                        help='Boolean to calculate summary of individual images. Default is false.', required=False)
    parser.add_argument('--totalStats', dest='totalStats', type=str, default="false",
                        help='Boolean to calculate overall statistics across all images. Default is false.', required=False)
    parser.add_argument('--totalSummary', dest='totalSummary', type=str, default="false",
                        help='Boolean to calculate summary across all images. Default is false.', required=False)
    parser.add_argument('--radiusFactor', dest='radiusFactor', type=str, default = 0.5,
                        help='Importance of radius/diameter to find centroid distance. Should be between (0,2].Default is 0.5.', required=False)
    parser.add_argument('--iouScore', dest='iouScore', type=str, default = 0,
                        help='IoU theshold. Default is 0.', required=False)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern to filter data.', required=True)

    # Output arguments                    
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    GTDir = args.GTDir
    logger.info('GTDir = {}'.format(GTDir))
    PredDir = args.PredDir
    logger.info('PredDir = {}'.format(PredDir))
    inputClasses = args.inputClasses
    logger.info('inputClasses = {}'.format(inputClasses))
    individualData = args.individualData == 'true' or args.individualData == 'True'
    logger.info('individualData = {}'.format(individualData))
    individualSummary = args.individualSummary == 'true' or args.individualData == 'True'
    logger.info('individualSummary = {}'.format(individualSummary))
    totalStats = args.totalStats == 'true' or args.individualData == 'True'
    logger.info('totalStats = {}'.format(totalStats))
    totalSummary = args.totalSummary == 'true' or args.individualData == 'True'
    logger.info('totalSummary = {}'.format(totalSummary))
    radiusFactor = float(args.radiusFactor)
    logger.info('radiusFactor = {}'.format(radiusFactor))
    iouScore = float(args.iouScore)
    logger.info('iouScore = {}'.format(iouScore))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    evaluation(GTDir, PredDir,inputClasses, outDir,individualData, individualSummary, \
        totalStats, totalSummary, radiusFactor,iouScore,filePattern)
