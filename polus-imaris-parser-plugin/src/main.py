import argparse
import pathlib
import extract_ims_data
import link_ims_ids
import merge_ids_to_features
import logging
import time
from pathlib import Path
import os

if __name__=="__main__":

    #: Initialize the logger
    logging.basicConfig(
        format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    #: Setup the argument parsing
    logger.info("Parsing arguments...")
    
    parser = argparse.ArgumentParser(
        prog='main', 
        description='Parses .ims file metadata; organizes features in .csv.')
    
    parser.add_argument('--inpdir', 
        dest='inpdir', type=str, help='Input collection of ims files', 
        required=True)

    parser.add_argument(
        '--metaoutdir', dest='metaoutdir', type=str,
        help='Output metadata collection that will hold overall .xlsx file', 
        required=True)

    parser.add_argument(
        '--outdir', dest='outdir', type=str, help='Output csv collection', 
        required=True)

    #: Parse the arguments
    args = parser.parse_args()

    #: Check for subfolders named images and switch to that subfolder
    inpdir = args.inpdir
    logger.debug("Old input directory: {}".format(inpdir))
    inpdir = pathlib.Path(inpdir)
    parent_meta_path = inpdir / 'metadata_files'
    
    try:
        # If given input directory points to images folder
        if inpdir.name == 'images':
            logger.info("Searching parent subdirectories for metadata_files")
            # Check if a subdirectory of the parent contains /metadata_files
            p = inpdir.parent
            q = p / 'metadata_files'
            if q.exists() == True:
                # Then switch inpdir to that metadata_files directory
                logger.info("Switching to subdirectory metadata_files")
                inpdir = inpdir.with_name("metadata_files")
        
        # If given input directory points to parent of /metadata_files
        elif parent_meta_path.exists() == True:
            logger.info("Switching to subdirectory metadata_files")
            # Switch inpdir to metadata_files directory
            inpdir = inpdir / 'metadata_files'
        
        else:
            logger.error("Directory not found. Please check that the input \
            directory is an image collection with at least one .ims file")
            raise FileNotFoundError("metadata_files directory not found.")
        
        logger.info('Navigated to metadata_files directory...')
        logger.debug("New input directory (inpdir): {}".format(inpdir))

        #: outdir is the csv collection
        outdir = args.outdir
        logger.debug('outdir = {}'.format(outdir))

        #: Overall.xlsx is stored in metadata collection
        metaoutdir = args.metaoutdir
        logger.debug('metaoutdir = {}'.format(metaoutdir))
        logger.debug("Defining paths...")
        #: Define the path
        currentDirectory = inpdir
        outputDirectory = pathlib.Path(outdir)
        metadataDirectory = pathlib.Path(metaoutdir)
        ims_exists = False
        for currentFile in currentDirectory.iterdir():

            if currentFile.suffix == '.ims':
                ims_exists = True
                logger.info("Parsing {}".format(currentFile))
                
                outputDirName = outputDirectory
                metadataDirectory = metadataDirectory
                
                #: ``extract_ims_data`` extracts metadata from **.ims** file
                hdf_to_csv = extract_ims_data.LinkData(currentFile, outputDirName)
                hdf_to_csv.link_data_fun()

                #: ``link_ims_ids`` extract/links track ID to object ID
                link_ims_ids.link_trackid_objectid(currentFile, outputDirName)

                #: ``merge_ids_to_features`` combines linked IDs to features 
                create_final_output = merge_ids_to_features.CreateCsv(
                    currentFile, outputDirName, metadataDirectory)
                
                create_final_output.create_csv_fun()
        if ims_exists == False:
            logger.error('Metadata directory of image collection lacks .ims files')

    except FileNotFoundError as error:
        logger.error(error)