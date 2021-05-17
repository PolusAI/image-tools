from sklearn.preprocessing import StandardScaler
import argparse
import logging
import os
import fnmatch
import csv
import hdbscan
import pandas as pd
import numpy as np
import numpy.matlib

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
   

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='HDBSCAN clustering plugin')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection-Data need to be clustered', required=True)
    parser.add_argument('--minclustersize', dest='minclustersize', type=int,
                        help='Minimum cluster size', required=True)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()  
    
    #Path to csvfile directory
    inpdir = args.inpdir
    logger.info('inpdir = {}'.format(inpdir))
    
    #Minimum cluster size for clustering using HDBSCAN
    minclustersize = args.minclustersize
    logger.info('minclustersize = {}'.format(minclustersize))
    
    #Path to save output csvfiles
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))
    
    #Get list of .csv files in the directory including sub folders for clustering
    inputcsv = list_file(inpdir)
    if not inputcsv:
        raise ValueError('No .csv files found.')
            
    for inpfile in inputcsv:
        #Get the full path
        split_file = os.path.normpath(inpfile)
        #split to get only the filename
        inpfilename = os.path.split(split_file)
        file_name_csv = inpfilename[-1]
        file_name,file_name1 = file_name_csv.split('.', 1)
        logger.info('Reading the file ' + file_name)
        #Read csv file
        cluster_data = pd.read_csv(inpfile)
        #Get fields with datatype as object to concate after clustering
        data_obj = cluster_data.select_dtypes(include='object')
        #Exclude fields with datatype as object for clustering
        data_num = cluster_data.select_dtypes(exclude='object')
        obj_array = np.array(data_obj)

        #Get column names
        col_name = cluster_data.columns.values.tolist()
        num_array = np.array(data_num, dtype=np.float64)
        
        #Standardize the data
        data=StandardScaler().fit_transform(num_array)

        #Cluster data using HDBSCAN clustering
        logger.info('Clustering the data')
        hdbclus = hdbscan.HDBSCAN(min_cluster_size = minclustersize).fit(data)
        label_data = hdbclus.labels_
        classified_data = label_data.reshape(label_data.shape[0],-1)
        #Adding one to convert the instances with -1 as 0 for outliers
        classified_data = classified_data.astype(int) + 1

        #Get clusters as last column
        df_processed = tuple(np.hstack((obj_array, data_num, classified_data)))
        col_name.append('Cluster')
        col_name_sep = ","
        col_names = col_name_sep.join(col_name) 
        
        #Save dataframe into csv file
        os.chdir(outdir)
        logger.info('Saving csv file')
        export_csv = np.savetxt('%s.csv'%file_name, df_processed, header = col_names, fmt="%s", comments='', delimiter=',')
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()