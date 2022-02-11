import argparse
import logging
import os
import fnmatch
import csv
import numpy as np
import pandas as pd
import vaex
import numpy.matlib
import filepattern
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import shutil

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

FILE_EXT = os.environ.get('POLUS_TAB_EXT',None)
FILE_EXT = FILE_EXT if FILE_EXT is not None else '.csv'
  
def elbow(data_array, minimumrange, maximumrange):
    """Determine k value and cluster data using elbow method.
    
    Args:
        data (array): Input data.
        minimumrange (int): Starting number of sequence in range function to determine k-value.
        maximumrange (int): Ending number of sequence in range function to determine k-value.
    
    Returns:
        Labeled data.
    
    """
    sse = []
    label_value = []
    logger.info('Starting Elbow Method...')
    K = range(minimumrange, maximumrange+1)
    for k in K:
        kmeans = KMeans(n_clusters = k, random_state=9).fit(data_array)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(data_array)
        curr_sse = 0
    
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        logger.info('Calculating Euclidean distance...')
        for i in range(len(data_array)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.linalg.norm(data_array[i]-np.array(curr_center)) ** 2
        sse.append(curr_sse)
        labels = kmeans.labels_
        label_value.append(labels)
    
    logger.info('Finding elbow point in curve...')
    #Find the elbow point in the curve
    points = len(sse)
    #Get coordinates of all points
    coord = np.vstack((range(points), sse)).T
    #First point
    f_point = coord[0]
    #Vector between first and last point
    linevec = coord[-1] - f_point
    #Normalize the line vector
    linevecn = linevec / np.sqrt(np.sum(linevec**2))
    #Vector between all point and first point
    vecf = coord - f_point
    #Parallel vector
    prod = np.sum(vecf * np.matlib.repmat(linevecn, points, 1), axis=1)
    vecfpara = np.outer(prod, linevecn)
    #Perpendicular vector
    vecline= vecf - vecfpara
    #Distance from curve to line
    dist = np.sqrt(np.sum(vecline ** 2, axis=1))
    #Maximum distance point
    k_cluster = np.argmax(dist)+minimumrange
    logger.info("k cluster: %s",k_cluster)
    logger.info("label value: %s",label_value)
    logger.info('Setting label_data')
    label_data = label_value[k_cluster]
    return label_data
    
def calinski_davies(data_array, methods, minimumrange, maximumrange):
    """Determine k value and cluster data using Calinski Harabasz Index method or Davies Bouldin based on method selection.
    
    Args:
        data (array): Input data.
        methods (str): Select either Calinski Harabasz or Davies Bouldin method.
        minimumrange (int): Starting number of sequence in range function to determine k-value.
        maximumrange (int):Ending number of sequence in range function to determine k-value.
        
    Returns:
        Labeled data.
    
    """
    K = range(minimumrange, maximumrange+1)
    chdb = []
    label_value = []
    for k in K:
        kmeans = KMeans(n_clusters = k, random_state=9).fit(data_array)
        labels = kmeans.labels_
        label_value.append(labels)
        if methods == 'CalinskiHarabasz':
            ch_db = calinski_harabasz_score(data_array, labels)
        else:
            ch_db = davies_bouldin_score(data_array, labels)
        chdb.append(ch_db)
    if methods == 'CalinskiHarabasz':
        score = max(chdb)
    else:
        score = min(chdb)
    k_cluster = chdb.index(score)
    label_data = label_value[k_cluster]
    return label_data

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='K-means clustering plugin.')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection-Data need to be clustered', required=True)
    parser.add_argument('--minimumrange', dest='minimumrange', type=int,
                        help='Enter minimum k-value:', required=False)
    parser.add_argument('--maximumrange', dest='maximumrange', type=int,
                        help='Enter maximum k-value:', required=False)
    parser.add_argument('--methods', dest='methods', type=str,
                        help='Select Manual or Elbow or Calinski Harabasz or Davies Bouldin method', required=False)
    parser.add_argument('--numofclus', dest='numofclus', type=int,
                        help='Number of clusters:', required=False)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()  
    
    #Path to csvfile directory
    inpdir = args.inpdir
    logger.info('inpdir = {}'.format(inpdir))
    
    #Determine k-value using different methods
    methods = args.methods
    logger.info('methods = {}'.format(methods))
    
    #minimum range value to consider for determining k-value
    minimumrange = args.minimumrange
    logger.info('minimumrange = {}'.format(minimumrange))
    
    #maximum range value to consider for determining k-value
    maximumrange = args.maximumrange
    logger.info('maximumrange = {}'.format(maximumrange))
    
    #k-value for clustering using K-Means
    numofclus = args.numofclus
    logger.info('numofclus = {}'.format(numofclus))
    
    #Path to save output csvfiles
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))
    
        
    #Get list of .csv files in the directory including sub folders for clustering

    # inputcsv = list_file(inpdir)
    
    filePattern ='.*.csv'
    fp = filepattern.FilePattern(inpdir,filePattern)
    if not fp:
        raise ValueError('No .csv files found.')
          
    #Dictionary of methods to determine k-value
    FEAT = {'Elbow': elbow,
            'CalinskiHarabasz': calinski_davies,
            'DaviesBouldin': calinski_davies}
    
    for files in fp:
        file = files[0]
        # Get filepath
        filepath = file.get('file')
        # Get file name
        filename = Path(filepath).stem
        # Copy file into output directory
        print(FILE_EXT)
        outputfile = os.path.join(outdir,(filename + FILE_EXT))
        shutil.copy(filepath, outputfile)
        
        logger.info('Started reading the file ' + filename)
        with open(outputfile, "r", encoding="utf-8") as fr:
            ncols = len(fr.readline().split(","))
        chunk_size = max([2 ** 24 // ncols, 1])
        
        df = vaex.read_csv(outputfile, convert=True, chunk_size=chunk_size)
        # Get list of column names
        cols = df.get_column_names()
        # Separate data by categorical and numerical data types
        numerical = []
        categorical = []
        for col in cols:
            if df[col].dtype == str:
                categorical.append(col)
            else:
                numerical.append(col)
        # Remove label field
        # numerical.remove("label")
        data = df[numerical]
        cat_array = df[categorical]
        
        if methods != 'Manual':
            #Check whether minimum range and maximum range value is entered
            if methods and not (minimumrange or maximumrange):
                raise ValueError('Enter both minimumrange and maximumrange to determine k-value.')
            if minimumrange <=1:
                raise ValueError('Minimumrange should be greater than 1.')
            logger.info('Determining k-value using ' + methods + ' and clustering the data.')
            if FEAT[methods] == calinski_davies:
                label_data = FEAT[methods](data, methods, minimumrange, maximumrange)
            if methods == 'Elbow':
                label_data = FEAT[methods](data, minimumrange, maximumrange)
        else:
            #Check whether numofclus is entered
            if not numofclus:
                raise ValueError('Enter number of clusters')
            kvalue = numofclus
            kmeans = KMeans(n_clusters = kvalue).fit(data)
            label_data = kmeans.labels_
        
        #Cluster data using K-Means clustering
        logger.info('Adding Cluster Data')
        data['Cluster'] = label_data
        
        #Add Categorical Data back to data processed
        logger.info('Adding categorical data')
        for col in categorical:
            data[col] = cat_array[col].values
        
        #Save dataframe to feather file or to csv file
        outputfile = os.path.join(outdir,(filename + FILE_EXT))
        
        if FILE_EXT == '.feather':
            data.export_feather(outputfile)
        else:
            logger.info('Saving csv file')
            # export_csv = np.savetxt('%s.csv'%filename, df_processed, header = cols, fmt="%s", delimiter=',')
            data.export_csv(outputfile, chunk_size=chunk_size)

if __name__ == "__main__":
    main()