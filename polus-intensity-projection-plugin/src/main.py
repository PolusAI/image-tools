import argparse, logging, time, sys, os, traceback
import intensity_projection


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate volumetric intensity projections')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--projectionType', dest='projectionType', type=str,
                        help='Type of volumetric intensity projection', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    projectionType = args.projectionType
    logger.info('projectionType = {}'.format(projectionType))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:
        if projectionType == 'max':
            intensity_projection.max_projection(inpDir, outDir)
        elif projectionType == 'min':
            intensity_projection.min_projection(inpDir, outDir)
        elif projectionType == 'mean':
            intensity_projection.mean_projection(inpDir, outDir)
        elif projectionType == 'mode':
            intensity_projection.mode_projection(inpDir, outDir)

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        sys.exit()

