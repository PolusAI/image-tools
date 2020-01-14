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
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    #: Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Parses metadata of Imaris files and outputs features in organized csv format.')
    parser.add_argument('--inpdir', dest='inpdir', type=str,
                        help='Input collection of ims files to be processed by this plugin', required=True)

    parser.add_argument('--metaoutdir', dest='metaoutdir', type=str,
                        help='Output metadata collection that will hold overall .xlsx file', required=True)

    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output csv collection', required=True)

    #: Parse the arguments
    args = parser.parse_args()

    #: Correct the filepath so that script accesses images stored in "metadata_files" directory instead of in "images" directory
    inpdir = args.inpdir
    logger.info("Old input directory: {}".format(inpdir))

    if inpdir.endswith("/images") == True:
        inpdir = inpdir[:-7]
        inpdir = inpdir + "/metadata_files"
    
    logger.info("New input directory: {}".format(inpdir))

    #: inpdir now points to the the collection containing .ims files
    logger.info('inpdir = {}'.format(inpdir))

    #: outdir is the csv collection
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))

    #: Overall.xlsx is stored in metadata collection
    metaoutdir = args.metaoutdir
    logger.info('metaoutdir = {}'.format(metaoutdir))

    #: Define the path
    currentDirectory = pathlib.Path(inpdir)
    outputDirectory = pathlib.Path(outdir)
    metadataDirectory = pathlib.Path(metaoutdir)

    for currentFile in currentDirectory.iterdir():

        if str(currentFile).endswith(".ims") == True:
            
            outputDirName = str(outputDirectory) + "/"
            metadataDirectory = str(metadataDirectory) + "/"
            
            #: Run module ``extract_ims_data`` to extract Imaris metadata from **.ims** filetype. 
            hdf_to_csv = extract_ims_data.LinkData(currentFile, outputDirName)
            hdf_to_csv.link_data_fun()

            #: Next, run module ``link_ims_ids`` to extract and link track ID and object ID information. 
            link_ims_ids.link_trackid_objectid(currentFile, outputDirName)

            #: Third, run module ``merge_ids_to_features`` to combine linked ID information to appropriate feature data
            create_final_output = merge_ids_to_features.CreateCsv(currentFile, outputDirName, metadataDirectory)
            create_final_output.create_csv_fun()