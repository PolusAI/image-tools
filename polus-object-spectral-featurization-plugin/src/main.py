import os
import mesh
import numpy as np
import pandas as pd
import argparse, logging
from pathlib import Path
from bfio import BioReader


if __name__=='__main__':
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info('Parsing arguments...')
    parser = argparse.ArgumentParser(prog='main', description='Spectral feature generation for segmented objects.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin.', required=True)
    parser.add_argument('--scaleInvariant', dest='scaleInvariant', type=str,
                        help='Calculate scale invariant features.', required=True)
    parser.add_argument('--numFeatures', dest='numFeatures', type=int, 
                        help='Number of spectral features to calculate.', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        inpDir = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    scaleInvariant = args.scaleInvariant.lower() == 'true'
    logger.info('scaleInvariant = {}'.format(scaleInvariant))
    numFeatures = args.numFeatures
    logger.info(f'numFeatures = {numFeatures}')
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and ''.join(f.suffixes[-2:]) == '.ome.tif']
    
    # Loop through files in inpDir image collection and process
    for i, f in enumerate(inpDir_files):
        # Load an image
        br = BioReader(Path(inpDir).joinpath(f))
        
        logger.info(f'Processing image ({i + 1}/{len(inpDir_files)}): {f}') 

        labels, features = mesh.mesh_and_featurize_image(br, num_features=numFeatures, scale_invariant=scaleInvariant)

        df = pd.DataFrame(data=labels, columns=['Label'])
        df_data = pd.DataFrame(data=features, columns=np.arange(numFeatures).tolist())

        df = pd.concat([df, df_data], axis=1)

        df.to_csv(os.path.join(outDir, f'{os.path.splitext(f)[0]}.csv'), index=False)