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
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern to filter data.', required=True)
    parser.add_argument('--individualStats', dest='individualStats', type=str,  default="false",
                        help='Boolean to create separate result file per image. Default is false.', required=False)
    parser.add_argument('--totalStats', dest='totalStats', type=str,  default="false",
                        help='Boolean to calculate overall statistics across all images. Default is false.', required=False)

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
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    individualStats = args.individualStats == 'true' or args.individualStats == 'True'
    logger.info('individualStats = {}'.format(individualStats))
    totalStats = args.totalStats == 'true' or args.totalStats == 'True'
    logger.info('totalStats = {}'.format(totalStats))

    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    evaluation(GTDir, PredDir,inputClasses, outDir, filePattern, individualStats, totalStats)