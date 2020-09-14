# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:57:57 2020

@author: nagarajanj2
"""
import argparse
import logging
import os
import fnmatch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score 

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
   
def elbow(data,krange):
    """Determine k value using elbow method.
    
    Args:
        data (ndarray): Input csvfile converted to array.
        
    Returns:
        Optimal k-value
    
    """
    data_array = np.array(data)
    sse = []
    K = range(1, krange)
    for k in K:
        kmeans = KMeans(n_clusters = k).fit(data_array)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(data_array)
        curr_sse = 0
        # calculate square of Euclidean distance of each point from its cluster center
        for i in range(len(data_array)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (data_array[i, 0] - curr_center[0]) ** 2 + (data_array[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)
    #Find the elbow point in the curve
    points = len(sse)
    coord = np.vstack((range(points), sse)).T
    np.array([range(points), sse])
    f_point = coord[0]
    linevec = coord[-1] - coord[0]
    linevecn = linevec / np.sqrt(np.sum(linevec**2))
    vecf = coord - f_point
    prod = np.sum(vecf * np.matlib.repmat(linevecn, points, 1), axis=1)
    vecfpara = np.outer(prod, linevecn)
    vecline= vecf - vecfpara
    dist = np.sqrt(np.sum(vecline ** 2, axis=1))
    k_cluster = np.argmax(dist)
    return k_cluster
    
def ch(data, krange):
    """Determine k value using Calinski Harabasz Index method.
    
    Args:
        data (ndarray): Input csvfile converted to array.
        
    Returns:
        Optimal k-value
    
    """
    data_array = np.array(data)
    K = range(2, krange)
    chs = []
    for k in K:
        kmeans = KMeans(n_clusters = k).fit(data_array)
        labels = kmeans.labels_ 
        ch_s = calinski_harabasz_score(data_array, labels)
        chs.append(ch_s)
    score = max(chs)
    k_cluster = chs.index(score) + 1
    return k_cluster

def db(data, krange):
    """Determine k value using Davies Bouldin Index method.
    
    Args:
        data (ndarray): Input csvfile converted to array.
        
    Returns:
        Optimal k-value
    
    """
    data_array = np.array(data)
    K = range(2, krange)
    dbs = []
    for k in K:
        kmeans = KMeans(n_clusters = k).fit(data_array)
        preds = kmeans.fit_predict(data_array)
        db_s = davies_bouldin_score(data_array, preds)
        dbs.append(db_s)
    score = min(dbs)
    k_cluster = dbs.index(score) + 1
    return k_cluster

def kmeans_cluster(data,cluster):
    """Cluster the data using K-Means clustering.
    
    Args:
        data (ndarray): Input csvfile converted to array.
        cluster (int): The number of clusters to be formed using K-Means.
        
    Returns:
        Dataframe containing labeled data.
    
    """
    data_array = np.array(data)
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit_predict(data_array)
    classified_data = kmeans.labels_
    df_processed = data.copy()
    df_processed['Cluster'] = pd.Series(classified_data, index=df_processed.index)
    return df_processed

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='K-means clustering plugin.')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection-Data need to be clustered', required=True)
    parser.add_argument('--numofclus', dest='numofclus', type=int,
                        help='Number of clusters', required=False)
    parser.add_argument('--determinek', dest='determinek', type=str,
                        help='Methods to determine k-value', required=False)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()  
    
    #Path to csvfile directory
    inpdir = args.inpdir
    logger.info('inpdir = {}'.format(inpdir))
    
    #Determine k-value using different methods
    determinek = args.determinek
    logger.info('determinek = {}'.format(determinek))
    
    #k-value for clustering using K-Means
    numofclus = args.numofclus
    logger.info('numofclus = {}'.format(numofclus))
    
    #Path to save output csvfiles
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))
    
    
    #Set default value for range of values to calaculate k-value using the methods
    krange = 20
    
    #Get list of .csv files in the directory including sub folders for clustering
    inputcsv = list_file(inpdir)
    if not inputcsv:
            raise ValueError('No .csv files found.')
            
    #Dictionary of methods to determine k-value
    FEAT = {'elbow': elbow,
            'ch': ch,
            'db': db}
       
    for inpfile in inputcsv:
        #Get the full labeled image path
        split_file = os.path.normpath(inpfile)
        
        #split to get only the filename
        inpfilename = os.path.split(split_file)
        file_name = inpfilename[-1]
        logger.info('Started reading the file ' + file_name)
        
        #Read the csv files using pandas
        data= pd.read_csv(inpfile)
        
        #Check whether any one of the methods is selected to determine k-value
        if determinek and not numofclus:
            logger.info('Determining k-value using' + determinek)
            kvalue = FEAT[determinek](data, krange)
            
        
        #Check value k-value is entered
        if numofclus:
            kvalue = numofclus
        
        #Check whether any one of the method is selected for determining k-value
        if not determinek and not numofclus:
            raise ValueError('Select method to determine k-value or enter k-value.')
        
        logger.info('Clustering the data')
        
        #Cluster data using K-Means clustering
        cluster_data = kmeans_cluster(data,kvalue)
        logger.info('Saving csv file')
        
        os.chdir(outdir)
        #Save dataframe into csv file
        export_csv = cluster_data.to_csv (r'kmeans_clustering_%s.csv'%file_name, index=None, header=True, encoding='utf-8-sig')
        logger.info("Finished all processes!")

if __name__ == "__main__":
    main()