from sklearn.preprocessing import StandardScaler
import argparse
import logging
import os
import fnmatch
import csv
import re
import hdbscan
import numpy as np
import pandas as pd
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

def clustering(clu_array,minclustersize):
    """Cluster data using HDBSCAN.
    
    Args:
        clu_array (array): Data that need to be clustered.
        minclustersize (int): Smallest size grouping that should be considered as a cluster.
        
    Returns:
        Labeled data.
        
    """
    hdbclus = hdbscan.HDBSCAN(min_cluster_size = minclustersize).fit(clu_array)
    label_data = hdbclus.labels_
    classified_data = label_data.reshape(label_data.shape[0],-1)
    #Adding one to convert the instances with -1 as 0 for outliers
    classified_data = classified_data.astype(int) + 1
    return classified_data
   
# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='HDBSCAN clustering plugin')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection-Data need to be clustered', required=True)
    parser.add_argument('--pattern', dest='pattern', type=str,
                        help='Regular expression for clustering each group', required=False)
    parser.add_argument('--minclustersize', dest='minclustersize', type=int,
                        help='Minimum cluster size', required=True)
    parser.add_argument('--outlierclusterID', dest='outlierclusterID', type=str,
                        help='Set cluster id as 1', required=False)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    #Path to csvfile directory
    inpdir = args.inpdir
    logger.info('inpdir = {}'.format(inpdir))

    #Regular expression for cluster by channels
    pattern = args.pattern
    logger.info('pattern = {}'.format(pattern))
    
    #Minimum cluster size for clustering using HDBSCAN
    minclustersize = args.minclustersize
    logger.info('minclustersize = {}'.format(minclustersize))

    #Set outlier cluster id as 1
    outlierclusterid = args.outlierclusterID
    logger.info('outlierclusterid = {}'.format(outlierclusterid))
    
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

        if pattern:
            #Check if multiple capture groups
            num_groups = re.compile(pattern).groups
            if num_groups > 1:
                raise ValueError("Cannot pass multiple capture groups")
            mask = data_obj.apply(
                        lambda x: x.str.findall(
                        pattern
                        )
                    ).any(axis=1)
            #Get dataframe with all the files that matches with the pattern
            cluster_data = cluster_data[mask]

            #Get dataframe with all the files that are not matching with the pattern
            cluster_data_nt=data_obj[~mask]
            if not cluster_data_nt.empty:
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', None)
                logger.info(f"Number of rows not matching the pattern in the file {file_name}:{len(cluster_data_nt)}")
                logger.debug(f"List of files not matching the pattern in the file {file_name} \n{cluster_data_nt}\n")
            
            if cluster_data.empty:
                logger.warning(f"Could not find pattern match in the file {file_name}. Skipping...")
                continue
        
            #Create a column group with matching string
            cluster_data['group'] = data_obj.apply(','.join,axis=1).str.extract(pattern, expand=False)
            group_data = cluster_data.groupby('group').apply(lambda x: x.sort_values('group'))

            #Get column names
            col_name = cluster_data.columns.values.tolist()

            #Get unique values in group
            df = group_data.group.unique()
            df_append=pd.DataFrame([])
            logger.info('Clustering the data')
            for i in df:
                sel_group=group_data.loc[group_data['group'] == i]
                sel_group_num = sel_group.select_dtypes(exclude='object')
                sel_group_obj = sel_group.select_dtypes(include='object')
                get_group = sel_group_obj.iloc[:,-1:]
                sel_group_obj = sel_group_obj.drop(labels='group', axis=1)
                
                #Cluster data using HDBSCAN clustering
                classified_data = clustering(sel_group_num,minclustersize)
                if outlierclusterid:
                    classified_data = classified_data + 1
                #Stack all columns
                df_processed = pd.DataFrame(np.hstack((sel_group_obj, sel_group_num, get_group, classified_data)))

                if not df_append.empty:
                    #Get max cluster number of a group and add to next group to have continuous numbering of clusters
                    max_clu = df_append.iloc[:,-1:].max().tolist()
                    add_clu = int(max_clu[0])
                    df_processed[df_processed.iloc[:,-1:] != 0] = add_clu + df_processed.iloc[:,-1:]

                df_append = df_append.append(df_processed)
        else:
            obj_array = np.array(data_obj)
            #Get column names
            col_name = cluster_data.columns.values.tolist()
            num_array = np.array(data_num, dtype=np.float64)

            #Cluster data using HDBSCAN clustering
            logger.info('Clustering the data')
            classified_data = clustering(num_array,minclustersize)
            if outlierclusterid:
                classified_data = classified_data + 1

            #Get clusters as last column
            df_append = tuple(np.hstack((obj_array, data_num, classified_data)))

        #Get clusters as last column
        col_name.append('cluster')
        col_name_sep = ","
        col_names = col_name_sep.join(col_name) 

        #Save dataframe into csv file
        os.chdir(outdir)
        logger.info('Saving csv file')
        export_csv = np.savetxt('%s.csv'%file_name, df_append, header = col_names, fmt="%s", comments='', delimiter=',')
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()