# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:14:58 2020

@author: nagarajanj2
"""
import os
import fnmatch
import fcsparser
import argparse
import logging

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def list_file(directory):
    """List all the .fcs files in the directory.
    
    Args:
        directory (str): Path to the directory containing the fcs files.
        
    Returns:
        The path to directory, list of names of the subdirectories in dirpath (if any) and the filenames of .fcs files.
        
    """
    list_of_files = [os.path.join(dirpath, file_name)
                     for dirpath, dirnames, files in os.walk(directory)
                     for file_name in fnmatch.filter(files, '*.fcs')]
    return list_of_files

def fcs_csv(metaDir,outDir):
    """Convert fcs file to csv.
    
    Args:
        metaDir (str): Path to the directory containing the fcs file.
        outDir (str): Path to save the output csv file.
        
    Returns:
        Converted csv file.
        
    """
    fullfile_path = os.path.normpath(metaDir)
    #Split the filename from full file path to save csv file with the same name as fcs file
    filename_path = os.path.split(fullfile_path)
    filename = filename_path[-1]
    file_name,file_name1 = filename.split('.', 1)
    logger.info('Started converting the fcs file ' + file_name)
    meta, data = fcsparser.parse(metaDir, meta_data_only=False, reformat_meta=True)
    logger.info('Saving csv file ' + file_name)
    os.chdir(outDir)
    #Export the file as csv
    export_csv = data.to_csv (r'%s.csv'%file_name, index = None, header=True,encoding='utf-8-sig')
    return export_csv

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Convert fcs file to csv file.')
    parser.add_argument('--metaDir',                       #Path to select files in metadata directory
                        dest='metaDir', 
                        type=str,
                        help='Input fcs file collection', 
                        required=True)
    parser.add_argument('--outDir',                       #Path to save files in output directory
                        dest='outDir', 
                        type=str,
                        help='Output csv collection', 
                        required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    #Path to input file directory
    metaDir = args.metaDir
    logger.info('metaDir = {}'.format(metaDir))
    
    #Path to save output csv files
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
   
    if metaDir:
        #List the files in the directory
        logger.info('Checking for .fcs files in the directory ')
        fcs_filelist = list_file(metaDir)
        #Check whether .fcs files are present in the directory
        if not fcs_filelist:
            raise FileNotFoundError('No .fcs files found in the directory.' )
           
    for each_file in fcs_filelist:
        #Read the fcs file and convert to csv file
        csv_file = fcs_csv(each_file, outDir)
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
        
