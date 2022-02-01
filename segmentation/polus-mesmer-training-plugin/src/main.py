import argparse, logging, subprocess, time, multiprocessing, sys
from train import run
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
    parser.add_argument('--trainingImages', dest='trainingImages', type=str,
                        help='Input training image collection to be processed by this plugin', required=True)
    parser.add_argument('--testingLabels', dest='testingLabels', type=str,
                        help='Input testing label collection to be processed by this plugin', required=True)
    parser.add_argument('--trainingLabels', dest='trainingLabels', type=str,
                        help='Input training label collection to be processed by this plugin', required=True)
    parser.add_argument('--tilesize', dest='tilesize', type=str, default="256",
                        help='Input image tile size. Default 256x256.', required=False)
    parser.add_argument('--iterations', dest='iterations', type=str, default="10",
                        help='Number of training iterations. Default is 10.', required=False)
    parser.add_argument('--batchSize', dest='batchSize', type=str, default="1",
                        help='Batch Size. Default is 1.', required=False)

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
    trainingImages = args.trainingImages
    if (Path.is_dir(Path(args.trainingImages).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.trainingImages).joinpath('images').absolute())
    logger.info('trainingImages = {}'.format(trainingImages))

    testingLabels = args.testingLabels
    if (Path.is_dir(Path(args.testingLabels).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.testingLabels).joinpath('images').absolute())
    logger.info('testingLabels = {}'.format(testingLabels))
    trainingLabels = args.trainingLabels
    if (Path.is_dir(Path(args.trainingLabels).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.trainingLabels).joinpath('images').absolute())
    logger.info('trainingLabels = {}'.format(trainingLabels))

    tilesize = args.tilesize
    logger.info('tilesize = {}'.format(tilesize))
    iterations = args.iterations
    logger.info('iterations = {}'.format(iterations))
    batchSize = args.batchSize
    logger.info('batchSize = {}'.format(batchSize))

    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    run(trainingImages, trainingLabels, testingImages, testingLabels, outDir, tilesize, iterations, batchSize)
    
