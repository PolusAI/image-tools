import argparse
from importlib.resources import path
import logging
import os
import pathlib 
import time
import re
import numpy as np
import multiprocessing
from typing import Optional, List, Union
from functools import partial
from thresholding import custom_fpr, n_sigma, otsu
import vaex
import json

# #Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
OUT_FORMAT = os.environ.get('FILE_EXT','csv')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='tabular-data-thresholding')    
#   # Input arguments

parser.add_argument(
        "--inpDir",
        dest="inpDir",
        type=str,
        help="Directory containing tabular data",
        required=True
    )

parser.add_argument(
        "--metaDir",
        dest="metaDir",
        type=str,
        help="Directory containing metadata information of tabular data",
        required=False
    )

parser.add_argument(
        "--mappingvariableName",
        dest="mappingvariableName",
        type=str,
        help="Common variableName between two CSVs and use to merge metadata and tabular data",
        required=False
    )

parser.add_argument(
        "--negControl",
        dest="negControl",
        type=str,
        help="FeatureName containing information about the position of non treated wells",
        required=True
    )

parser.add_argument(
        "--posControl",
        dest="posControl",
        type=str,
        help="FeatureName containing information about the position of wells with known treatment outcome",
        required=False
    )

parser.add_argument(
        "--variableName",
        dest="variableName",
        type=str,
        help="Name of the Variable for computing thresholds",
        required=True
    )

parser.add_argument(
        "--thresholdType",
        dest="thresholdType",
        type=str,
        help="Name of the threshold method",
        required=False
    )

parser.add_argument(
        "--falsePositiverate",
        dest="falsePositiverate",
        type=float,
        default=0.1,
        help="False positive rate threshold value",
        required=False
    )

parser.add_argument(
        "--numBins",
        dest="numBins",
        type=int,
        default=512,
        help="Number of Bins for otsu threshold",
        required=False
    )

parser.add_argument(
        "--n",
        dest="n",
        type=int,
        default=4,
        help="Number of Standard deviation",
        required=False
    )

parser.add_argument(
        "--outFormat",
        dest="outFormat",
        type=str,
        default='csv',
        help="Output format",
        required=False
    )
                
#  # Output arguments
parser.add_argument('--outDir',
    dest='outDir',
    type=str,
    help='Output directory',
    required=True
    )  


def thresholding_func(csvfile:str,
                    inpDir:pathlib.Path,
                    metaDir:pathlib.Path,
                    outDir:pathlib.Path,                  
                    mappingvariableName:str,
                    negControl:str,
                    posControl:str,
                    variableName:str,
                    thresholdType:str,
                    falsePositiverate:Optional[float]=0.1,
                    numBins:Optional[int]=512,
                    n:Optional[int]=4,
                    outFormat:Optional[str]='csv'):

        metafile = [f for f in os.listdir(metaDir) if f.endswith('.csv')]
        logger.info(f'Number of CSVs detected: {len(metafile)}, filenames: {metafile}')
        if metafile:
            assert len(metafile) > 0 and len(metafile) < 2, logger.info(f'There should be one metadata CSV used for merging: {metafile}')

        if metafile:
            if mappingvariableName is None:
                raise ValueError(logger.info(f'{mappingvariableName} Please define Variable Name to merge CSVs together'))

        data = vaex.from_csv(inpDir.joinpath(csvfile), convert=False)
        meta = vaex.from_csv(metaDir.joinpath(metafile[0]), convert=False)  
        assert f'{mappingvariableName}' in list(meta.columns), logger.info(f'{mappingvariableName} is not present in metadata CSV')
        df = data.join(meta,how='left',
                            left_on = mappingvariableName,
                            right_on= mappingvariableName,
                            allow_duplication=False) 

        collist = list(df.columns)
        collist2 = [negControl, posControl, variableName]
        columns = collist[:3] + collist2
        df = df[columns]

        if posControl is None:
            logger.info(f"Otsu threshold will not be computed as it requires information of both {negControl} & {posControl}")

        if posControl:
            if df[posControl].unique() != [0.0, 1.0]:
                raise ValueError(logger.info(f'{posControl} Positive controls are missing. Please check the data again'))           
           
            pos_controls = df[df[posControl] == 1][variableName].values
       
        if df[negControl].unique() != [0.0, 1.0]:
            raise ValueError(logger.info(f'{negControl} Negative controls are missing. Please check the data again'))
        neg_controls = df[df[negControl] == 1][variableName].values

        plate = re.match('\w+', csvfile).group(0)
        threshold_dict = {}
        threshold_dict['plate'] = plate

        if thresholdType == 'fpr':
            threshold = custom_fpr.find_threshold(neg_controls, false_positive_rate=falsePositiverate)
            threshold_dict[thresholdType] = threshold
            df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)         
        elif thresholdType == 'otsu':
            combine_array = np.append(neg_controls, pos_controls, axis=0)
            threshold = otsu.find_threshold(combine_array, num_bins=numBins, normalize_histogram = False)
            threshold_dict[thresholdType] = threshold
            df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)
        elif thresholdType == 'nsigma':
            threshold = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict[thresholdType] = threshold
            df[thresholdType] = df.func.where(df[variableName] <= threshold, 0, 1)
        elif thresholdType == 'all':
            fpr_thr = custom_fpr.find_threshold(neg_controls, false_positive_rate=falsePositiverate)
            combine_array = np.append(neg_controls, pos_controls, axis=0)
            otsu_thr = otsu.find_threshold(combine_array, num_bins=numBins, normalize_histogram = False)
            nsigma_thr = n_sigma.find_threshold(neg_controls, n=n)
            threshold_dict['fpr'] = fpr_thr
            threshold_dict['otsu'] = otsu_thr
            threshold_dict['nsigma'] = nsigma_thr
            df['fpr'] = df.func.where(df[variableName] <= fpr_thr, 0, 1)
            df['otsu'] = df.func.where(df[variableName] <= otsu_thr, 0, 1)
            df['nsigma'] = df.func.where(df[variableName] <= nsigma_thr, 0, 1)
   
        
        OUT_FORMAT = OUT_FORMAT if outFormat is None else outFormat
        if OUT_FORMAT == "feather":
            outname = outDir.joinpath(f'{plate}_binary.feather')
            df.export_feather(outname)
            logger.info(f"Saving f'{plate}_binary.feather")
        else:
            outname = outDir.joinpath(f'{plate}_binary.csv')
            df.export_csv(path=outname, chunk_size=10_000)
            logger.info(f"Saving f'{plate}_binary.csv")
        return 

# # # Parse the arguments
args = parser.parse_args()
def main(args):
    starttime = time.time()    
    inpDir = pathlib.Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    assert pathlib.Path(inpDir).exists(), f'Path of CSVs directory not found: {inpDir}'
    metaDir = pathlib.Path(args.metaDir)
    logger.info('metaDir = {}'.format(metaDir))
    mappingvariableName = args.mappingvariableName
    logger.info('mappingvariableName = {}'.format(mappingvariableName))
    outDir = pathlib.Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))  
    assert pathlib.Path(inpDir).exists(), f'Path of output directory not found: {outDir}' 
    negControl= args.negControl
    logger.info('negControl = {}'.format(negControl)) 
    posControl= args.posControl
    logger.info('posControl = {}'.format(posControl)) 
    variableName= args.variableName
    logger.info('variableName = {}'.format(variableName)) 
    thresholdType= args.thresholdType
    logger.info('thresholdType = {}'.format(thresholdType)) 
    falsePositiverate= args.falsePositiverate
    logger.info('falsePositiverate = {}'.format(falsePositiverate)) 
    numBins= int(args.numBins)
    logger.info('numBins = {}'.format(numBins)) 
    n= int(args.n)
    logger.info('n = {}'.format(n)) 
    outFormat= str(args.outFormat)
    logger.info('outFormat = {}'.format(outFormat)) 

    csvlist = sorted([f for f in os.listdir(inpDir) if f.endswith('.csv')])
    logger.info(f'Number of CSVs detected: {len(csvlist)}, filenames: {csvlist}')
    metalist = [f for f in os.listdir(metaDir) if f.endswith('.csv')]
    logger.info(f'Number of CSVs detected: {len(metalist)}, filenames: {metalist}')
    if metaDir:
        assert len(metalist) > 0 and len(metalist) < 2, logger.info(f'There should be one metadata CSV used for merging: {metaDir}')

    num_workers = max(multiprocessing.cpu_count() // 2, 2)

    with multiprocessing.Pool(processes=num_workers) as executor:     
        executor.map(partial(thresholding_func, 
                        inpDir=inpDir,
                        metaDir=metaDir, 
                        outDir=outDir,
                        mappingvariableName=mappingvariableName,
                        negControl=negControl,
                        posControl=posControl,
                        variableName=variableName,
                        thresholdType=thresholdType,
                        falsePositiverate=falsePositiverate,
                        numBins=numBins,
                        n=n,
                        outFormat=outFormat), csvlist)
        executor.close()
        executor.join()
    endtime = round((time.time() - starttime)/60, 3)
    logger.info(f"Time taken to finish nyxus feature extraction: {endtime} minutes!!!")
    return

if __name__=="__main__":
    main(args)