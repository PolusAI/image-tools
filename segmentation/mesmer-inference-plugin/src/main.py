import argparse, logging, subprocess, time, multiprocessing, sys
from padded import run
from pathlib import Path

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='WIPP plugin to train PanopticNet.')
    
    # Input arguments
    parser.add_argument('--testingImages', dest='testingImages', type=str,
                        help='Input testing image collection to be processed by this plugin', required=True)
    parser.add_argument('--testingLabels', dest='testingLabels', type=str,
                        help='Input testing label collection to be processed by this plugin', required=False)
    parser.add_argument('--tilesize', dest='tilesize', type=str, default="256",
                        help='Input image tile size. Default 256x256.', required=False)
    parser.add_argument('--modelPath', dest='modelPath', type=str,
                        help='Path to weights file.', required=False)
    parser.add_argument('--filePatternTest', dest='filePatternTest', type=str,
                        help='Filename pattern to filter data.', required=True)
    parser.add_argument('--filePatternWholeCell', dest='filePatternWholeCell', type=str,
                        help='Filename pattern to filter nuclear data.', required=False)

    parser.add_argument('--model', dest='model', type=str, default="mesmer",
                        help='Model type.', required=True)


    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    testingImages = args.testingImages
    if (Path.is_dir(Path(args.testingImages).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.testingImages).joinpath('images').absolute())
    logger.info('testingImages = {}'.format(testingImages))

    testingLabels = args.testingLabels
    logger.info('testingLabels = {}'.format(testingLabels))

    modelPath = args.modelPath
    logger.info('modelPath = {}'.format(modelPath))

    filePatternTest = args.filePatternTest
    logger.info('filePatternTest = {}'.format(filePatternTest))

    filePatternWholeCell = args.filePatternWholeCell
    logger.info('filePatternWholeCell = {}'.format(filePatternWholeCell))

    tilesize = args.tilesize
    logger.info('tilesize = {}'.format(tilesize))

    model = args.model
    logger.info('model = {}'.format(model))


    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    run(testingImages, testingLabels, tilesize, modelPath, filePatternTest, filePatternWholeCell, model, outDir)
    
