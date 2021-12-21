import argparse
import logging
import os
import fnmatch
import hdbscan
import numpy as np
import pandas as pd
import typing

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def list_files(csv_directory: str) -> typing.List[str]:
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


def clustering(data: np.ndarray, min_cluster_size: int, increment_outlier_id: bool) -> np.ndarray:
    """Cluster data using HDBSCAN.
    
    Args:
        data (array): Data that need to be clustered.
        min_cluster_size (int): Smallest size grouping that should be considered as a cluster.
        increment_outlier_id (bool) : Increment outlier ID to unity.
        
    Returns:
        Cluster labels for each row of data.
    """
    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
    labels = clusters.labels_.flatten().astype(np.uint16) + 1
    labels = labels + 1 if increment_outlier_id else labels

    return labels
    
   
# Setup the argument parsing
def main(inpDir, grouping_pattern, avg_groups, label_col, min_cluster_size, increment_outlier_id, outDir):
    # Get list of .csv files in the directory including sub folders for clustering
    input_csvs = list_files(inpDir)
    if input_csvs is None:
        raise ValueError('No .csv files found.')
            
    for csv in input_csvs:
        # Get the full path and split to get only the filename.
        split_file = os.path.normpath(csv)
        file_name = os.path.split(split_file)[-1]
        file_prefix, _ = file_name.split('.', 1)

        logger.info('Reading the file ' + file_name)

        # Read csv file
        df = pd.read_csv(csv)

        # If user provided a regular expression.
        if grouping_pattern is not None:
            df = df[df[label_col].str.match(grouping_pattern)].copy()
            if df.empty:
                logger.warning(f"Could not find any files matching the pattern {grouping_pattern} in file {csv}. Skipping...")
                continue
        
            #Create a column group with matching string
            df['group'] = df[label_col].str.extract(grouping_pattern, expand=True).apply(','.join, axis=1)

            # Get column(s) containing data.
            df_data = df.select_dtypes(exclude='object').copy()
            df_data['group'] = df['group']
            
            # If we want to average features for each group.
            if avg_groups:
                df_grouped = df_data.groupby('group').apply(lambda x: x.sort_values('group').mean(numeric_only=True))           
            
                # Cluster data using HDBSCAN clustering.
                logger.info('Clustering the data')
                cluster_ids = clustering(df_grouped.values, min_cluster_size, increment_outlier_id)

                df_grouped['cluster'] = cluster_ids
                df = df.merge(df_grouped['cluster'], left_on='group', right_index=True)
            else: # We want separate clustering results for each group.
                dfs = []
                for group, df_ss in df_data.groupby('group'):
                    # Cluster data using HDBSCAN clustering.
                    logger.info(f'Clustering data in group {group}')

                    cluster_ids = clustering(df_ss.values, min_cluster_size, increment_outlier_id)
                    df_ss['cluster'] = cluster_ids
                    dfs.append(df_ss)
                
                df_grouped = pd.concat(dfs)
                df = df.merge(df_grouped['cluster'], left_index=True, right_index=True)
            
        # No grouping. Vanilla clustering. 
        else:
            # Get column(s) containing data.
            df_data = df.select_dtypes(exclude='object').copy()
            
            #Cluster data using HDBSCAN clustering
            logger.info('Clustering the data')
            cluster_ids = clustering(df_data.values, min_cluster_size, increment_outlier_id)
            df['cluster'] = cluster_ids

        df.to_csv(os.path.join(outDir, f'{file_prefix}.csv'), index=None, header=True, encoding='utf-8-sig')
    logger.info("Finished all processes!")

if __name__ == "__main__":
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='HDBSCAN clustering plugin')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input collection-Data need to be clustered', required=True)
    parser.add_argument('--groupingPattern', dest='groupingPattern', type=str,
                        help='Regular expression to group rows. Clustering will be applied across capture groups.', required=False)
    parser.add_argument('--averageGroups', dest='averageGroups', type=str,
                        help='Whether to average data across groups. Requires capture groups.', default='false', required=False)
    parser.add_argument('--labelCol', dest='labelCol', type=str,
                        help='Name of column containing labels. Required only for grouping operations.', required=False)
    parser.add_argument('--minClusterSize', dest='minClusterSize', type=int,
                        help='Minimum cluster size', required=True)
    parser.add_argument('--incrementOutlierId', dest='incrementOutlierId', type=str,
                        help='Increments outlier ID to 1.', default='false', required=False)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments.
    args = parser.parse_args()
    
    # Path to csvfile directory.
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))

    # Regular expression for grouping.
    grouping_pattern = args.groupingPattern
    logger.info('grouping_pattern = {}'.format(grouping_pattern))

    # Whether to average data for each group.
    avg_groups = args.averageGroups.lower() != 'false'
    logger.info('avg_groups = {}'.format(avg_groups))

    # Name of column to use for grouping.
    label_col = args.labelCol
    logger.info('label_col = {}'.format(label_col))

    # Minimum cluster size for clustering using HDBSCAN.
    min_cluster_size = args.minClusterSize
    logger.info('min_cluster_size = {}'.format(min_cluster_size))

    # Set outlier cluster id as 1.
    increment_outlier_id = args.incrementOutlierId.lower() != 'false' 
    logger.info('increment_outlier_id = {}'.format(increment_outlier_id))
    
    # Path to save output csvfiles.
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    main(
        inpDir, 
        grouping_pattern, 
        avg_groups, 
        label_col, 
        min_cluster_size, 
        increment_outlier_id, 
        outDir
    )