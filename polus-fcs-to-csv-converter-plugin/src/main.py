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

def list_file(inpDir):
    """List all the .fcs files in the directory.
    
    Args:
        inpDir (str): Path to the directory containing the fcs files.
        
    Returns:
        The path to directory, list of names of the subdirectories in dirpath (if any) and the filenames of .fcs files.
        
    """
    list_of_files = [os.path.join(dirpath, file_name)
                     for dirpath, dirnames, files in os.walk(inpDir)
                     for file_name in fnmatch.filter(files, '*.fcs')]
    return list_of_files

def fcs_csv(inpDir,outDir):
    """Convert fcs file to csv.
    
    Args:
        inpDir (str): Path to the directory containing the fcs file.
        outDir (str): Path to save the output csv file.
        
    Returns:
        Converted csv file.
        
    """
    meta, data = fcsparser.parse(inpDir, meta_data_only=False, reformat_meta=True)
    fullfile_path = os.path.normpath(inpDir)
    #Split the filename from full file path to save csv file with the same name as fcs file
    filename_path = os.path.split(fullfile_path)
    filename = filename_path[-1]
    os.chdir(outDir)
    #Export the file as csv
    export_csv = data.to_csv (r'%s.csv'%filename, index = None, header=True,encoding='utf-8-sig')
    return export_csv

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Convert fcs file to csv file.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input fcs file collection', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output csv collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    
    #Path to input file directory
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    
    #Path to save output csv files
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    logger.info("Started")
    
    if inpDir:
        #List the files in images directory
        fcs_filelist = list_file(inpDir)
        #Check whether .fcs files are present in the input directory
        if not fcs_filelist:
            logger.warning('No .fcs files found in the images directory.' )

    inpdir_meta = inpDir + '/metadata_files'
    if inpdir_meta:
        #List the files in metadata_files directory
        fcs_metalist = list_file(inpdir_meta)
        #Check whether .fcs files are present in the input directory
        if not fcs_metalist:
            raise ValueError('No .fcs files found in the metadata_files directory.')
            
    fcs_finallist = fcs_filelist + fcs_metalist    
            
    for each_file in fcs_finallist:
        #Get the full fcs file path
        logger.info('Started converting the fcs file ' + each_file)
        #Read the fcs file and convert to csv file
        csv_file = fcs_csv(each_file, outDir)
        logger.info('Finished reading the file ' + each_file)
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
        
