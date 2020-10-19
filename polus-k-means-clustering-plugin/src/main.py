import argparse
import logging
import os
import fnmatch
import csv
import numpy as np
import numpy.matlib
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
    if minimumrange <=0:
        raise ValueError('Minimumrange should be equal to or greater than 1.')
    K = range(minimumrange, maximumrange+1)
    for k in K:
        kmeans = KMeans(n_clusters = k, random_state=9).fit(data_array)
        sse.append(kmeans.inertia_)
        labels = kmeans.labels_
        label_value.append(labels)
 
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
    k_cluster = np.argmax(dist)
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
    if minimumrange <=1:
        raise ValueError('Minimumrange should be greater than 1.')
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
    inputcsv = list_file(inpdir)
    if not inputcsv:
        raise ValueError('No .csv files found.')
            
    #Dictionary of methods to determine k-value
    FEAT = {'Elbow': elbow,
            'CalinskiHarabasz': calinski_davies,
            'DaviesBouldin': calinski_davies}
       
    for inpfile in inputcsv:
        #Get the full path
        split_file = os.path.normpath(inpfile)
        
        #split to get only the filename
        inpfilename = os.path.split(split_file)
        file_name_csv = inpfilename[-1]
        file_name,file_name1 = file_name_csv.split('.', 1)
        logger.info('Started reading the file ' + file_name)
        read_csv = open(inpfile, "rt", encoding="utf8")
        #Read csv file
        reader = csv.reader(read_csv)
        #Get column names
        col_name = next(reader)
        data_list = list(reader)
        data = np.array(data_list, dtype=np.float)
        
        if methods != 'Manual':
            #Check whether minimum range and maximum range value is entered
            if methods and not (minimumrange or maximumrange):
                raise ValueError('Enter both minimumrange and maximumrange to determine k-value.')
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
        classified_data = label_data.reshape(label_data.shape[0],-1)
        classified_data = classified_data.astype(int) + 1
        df_processed = tuple(np.hstack((data, classified_data)))
        col_name.append('Cluster')
        col_name_sep = ","
        col_names = col_name_sep.join(col_name) 
        
        #Save dataframe into csv file
        os.chdir(outdir)
        logger.info('Saving csv file')
        #output_csv = np.vstack ((col_name, df_processed))
        export_csv = np.savetxt('kmeans_clustering_%s.csv'%file_name, df_processed, header = col_names, fmt="%s", delimiter=',')
        logger.info("Finished all processes!")

if __name__ == "__main__":
    main()