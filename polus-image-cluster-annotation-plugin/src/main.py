from pathlib import Path
from bfio import BioReader,BioWriter
import argparse
import logging
import os
import fnmatch
import csv
import numpy as np
import pandas as pd

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def list_file(directory,ext):
    """List all the files in the directory based on the extension mentioned as input.
    
    Args:
        directory (str): Path to the directory containing the files.
        
    Returns:
        The path to directory, list of names of the subdirectories in dirpath (if any) and the filenames of the files.
        
    """
    list_of_files = [os.path.join(dirpath, file_name)
                     for dirpath, dirnames, files in os.walk(directory)
                     for file_name in fnmatch.filter(files, ext)]
    return list_of_files
   
# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Image clustering annotation plugin.')
    parser.add_argument('--imgdir', dest='imgdir', type=str,
                        help='Input collection- Image data', required=True)
    parser.add_argument('--csvdir', dest='csvdir', type=str,
                        help='Input collection- csv data', required=True)
    parser.add_argument('--borderwidth', dest='borderwidth', type=int, default = 2,
                        help='Border width', required=False)
    parser.add_argument('--outdir', dest='outdir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()  
    
    #Path to image directory
    imgdir = args.imgdir
    logger.info('imgdir = {}'.format(imgdir))
    
    #Path to csvfile directory
    csvdir = args.csvdir
    logger.info('csvdir = {}'.format(csvdir))
    
    #Get the border width
    borderwidth = args.borderwidth
    logger.info('borderwidth = {}'.format(borderwidth))

    #Path to save output image files
    outdir = args.outdir
    logger.info('outdir = {}'.format(outdir))

    #Get list of .ome.tif files in the directory including sub folders
    img_ext='*.ome.tif'
    configfiles = list_file(imgdir,img_ext)
    config = [os.path.basename(path) for path in configfiles]
    #Check whether .ome.tif files are present in the labeled image directory
    if not configfiles:
        raise ValueError('No .ome.tif files found.')

    #Get list of .csv files in the directory including sub folders
    csv_ext='*.csv'
    inputcsv = list_file(csvdir,csv_ext)
    if not inputcsv:
        raise ValueError('No .csv files found.')

    for inpfile in inputcsv:
        #Get the full path
        split_file = os.path.normpath(inpfile)
        #split to get only the filename
        inpfilename = os.path.split(split_file)
        file_name_csv = inpfilename[-1]
        file_path = inpfilename[0]
        file_name,file_name1 = file_name_csv.split('.', 1)
        logger.info('Reading the file ' + file_name)
        #Read csv file
        cluster_data = pd.read_csv(inpfile)
        cluster_data = cluster_data.iloc[:,[0,-1]]
        for index, row in cluster_data.iterrows():
            filename = row[0]
            cluster = row[1]
            #get the image file that matches with the filename in csvfile
            matches = [match for match in config if filename in match]
            if len(matches) == 0:
                logger.warning(f"Could not find image files matching the filename, {filename}. Skipping...")
                continue
            match_getpath =[s for s in configfiles if matches[0] in s]
            #Get the full path
            full_path = os.path.normpath(match_getpath[0])
            #split to get only the filename
            file_path = os.path.split(full_path)[0]
            #Get the image path and output directory path
            imgpath = Path(file_path)
            outpath = Path(outdir)
            #Read and write(after making changes) the .ome.tif files
            with BioReader(imgpath / filename) as br, \
                BioWriter(outpath / filename,metadata=br.metadata) as bw:
                #Make all pixels zero except the borders of specified thickness and assign the cluster_id to border pixels
                mask = np.zeros(br.shape,dtype=np.int16)
                mask[:borderwidth,:]=cluster
                mask[:,:borderwidth]=cluster
                mask[-borderwidth:,:]=cluster
                mask[:,-borderwidth:]=cluster
                bw.dtype = mask.dtype
                bw[:]=mask
        logger.info("Finished all processes!")

if __name__ == "__main__":
    main()