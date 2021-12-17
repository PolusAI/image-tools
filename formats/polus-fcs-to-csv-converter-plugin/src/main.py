from pathlib import Path
import fcsparser
import os
import argparse
import logging


# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def fcs_csv(file,outDir):
    """Convert fcs file to csv.
    
    Args:
        file (str): Path to the directory containing the fcs file.
        outDir (str): Path to save the output csv file.
        
    Returns:
        Converted csv file.
        
    """
    file_name = Path(file).stem
    logger.info('Started converting the fcs file ' + file_name)
    meta, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=True)
    logger.info('Saving csv file ' + file_name)
    #Export the file as csv
    os.chdir(outDir)
    export_csv = data.to_csv (r'%s.csv'%file_name, index = None, header=True,encoding='utf-8-sig')
    return export_csv

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Convert fcs file to csv file.')
    parser.add_argument('--inpDir',                       #Path to select files in input directory
                        dest='inpDir', 
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
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    
    #Path to save output csv files
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    inp_dir = Path(inpDir)
    inpdir_meta = inp_dir.parent.joinpath('metadata_files')
    if not inpdir_meta.is_dir():
        raise FileNotFoundError('metadata_files not found')
    
    #List the files in the directory
    logger.info('Checking for .fcs files in the directory ')
    fcs_filelist = list(Path(inpdir_meta).glob('*.fcs'))
    if not fcs_filelist:
        raise FileNotFoundError('No .fcs files found in the directory.' )
         
    for each_file in fcs_filelist:
        #Read the fcs file and convert to csv file
        csv_file = fcs_csv(each_file, outDir)
    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
        
