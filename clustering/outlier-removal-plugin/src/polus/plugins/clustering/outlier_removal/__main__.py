"""Outlier Removal Plugin."""

import argparse
import logging
import os
import fnmatch
import typing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def list_file(csv_directory):
    """List all the .csv files in the directory.
    
    Args:
        csv_directory (str): Path to the directory containing the csv files.
        
    Returns:
        The path to directory, list of names of the subdirectories in dirpath (if any) and the filenames of .csv files.
        
    """
    list_of_files = [os.path.join(dirpath, file_name)
                     for dirpath, dirnames, files in os.walk(csv_directory)
                     for file_name in fnmatch.filter(files, '*.csv')]
    return list_of_files

def isolationforest(data_set, types):
    """Detects outliers using Isolation Forest algorithm.
    
    Args:
        data_set (array): Input data.
        types (str): Type of outliers to be removed.
        
    Returns:
        ndarray whether or not the data point should be considered as an inlier according to the fitted model.
    
    """
    if types == 'Global':
        clf = IsolationForest(random_state=19,n_estimators=200)
    else:
        clf = IForest(random_state=10,n_estimators=200)
    clf.fit(data_set)
    pred = clf.predict(data_set)
    return pred
 
def main(inpdir: os.path,
         methods: typing.List[str],
         outdir: os.chdir,
         types: typing.Optional[str]=None
         ) -> None:
    
    
    #Get list of .csv files in the directory including sub folders for outlier removal
    inputcsv = list_file(inpdir)
    if not inputcsv:
        raise ValueError('No .csv files found.')
        
    if not methods:
       raise ValueError('Select methods for outlier detection.')
            
    #Dictionary of methods to detect outliers
    FEAT = {'IsolationForest': isolationforest}
       
    for inpfile in inputcsv:
        #Get the full path
        split_file = os.path.normpath(inpfile)
        
        #split to get only the filename
        inpfilename = os.path.split(split_file)
        file_name_csv = inpfilename[-1]
        file_name,file_name1 = file_name_csv.split('.', 1)
        
        #Read csv file
        logger.info('Started reading the file ' + file_name)
        df = pd.read_csv(inpfile)
        
        #Standardize the data
        data = StandardScaler().fit_transform(df)
        
        #Detect outliers
        logger.info('Detecting outliers using ' + methods)
        rem_out = FEAT[methods](data,types)
        df['anomaly']= rem_out
        if types == 'Global':
            inliers = df.loc[df['anomaly']==1]
            outliers = df.loc[df['anomaly']==-1]
        else:
            inliers = df.loc[df['anomaly']==0]
            outliers = df.loc[df['anomaly']==1]
        
        #Drop 'anomaly' column    
        inliers = inliers.drop('anomaly',axis=1)
        outliers = outliers.drop('anomaly',axis=1)

    	#Save dataframe into csv file
        os.chdir(outdir)
        logger.info('Saving csv file')
        export_inlier = inliers.to_csv (r'%s.csv'%file_name, index=None, header=True, encoding='utf-8-sig')
        export_outlier = outliers.to_csv (r'%s_outliers.csv'%file_name, index=None, header=True, encoding='utf-8-sig')
        logger.info("Finished all processes!")

if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Outlier removal plugin.')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection-Data that need outliers to be removed', required=True)
    parser.add_argument('--methods', dest='methods', type=str,
                        help='Select methods for outlier detection', required=True)
    parser.add_argument('--types', dest='types', type=str,
                        help='Select type of outliers to detect', required=False)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()  
    
    #Path to csvfile directory
    inpdir = args.inpdir
    logger.info('inpdir = {}'.format(inpdir))
    
    #Detect outliers using different methods
    methods = args.methods
    logger.info('methods = {}'.format(methods))
    
    #Detect outliers based on type selected
    types = args.types
    logger.info('types = {}'.format(types))
    
    #Path to save output csvfiles
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))
    
    main(inpdir,
         methods,
         outdir,
         types)