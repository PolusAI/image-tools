from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import fcsparser
import filepattern
import os
import argparse
import logging
import vaex

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# Step 1: Access files in input directory
# Step 2: Filter files into array by filepatten provided
# Step 3: Convert filetype to intermediate file type or Vaex DF
# Step 4: Convert Vaex DF to Arrow Feather File
# Step 5: Save feather file in output collection
# Step 6: remove all intermediate files

def csv_to_df(file):
    """Convert csv into datafram or hdf5 file. Copied partially from Nicholas-Schaub polus-csv-to-feather-plugin
            
            Args:
                file (str): Path to input file.
                filePattern (str): extension of file to convert.
                
            Returns:
                Vaex dataframe or hdf5 file
                
    """
    
    logger.info('CSV CONVERSION: Checking size of csv file...')
    # Open csv file and count rows in file
    with open(file,'r', encoding='utf-8') as fr:
        ncols = len(fr.readline().split(','))
        
    chunk_size = max([2**24 // ncols, 1])
    logger.info('CSV CONVERSION: # of columns are: ' + str(ncols))
        
    # Convert large csv files to hdf5 if more than 1,000,000 rows
    if ncols < 1000000:
        logger.info('CSV CONVERSION: converting file into hdf5 format')
        df = vaex.from_csv(file,convert=True,chunk_size=chunk_size)
    else:
        logger.info('CSV CONVERSION: converting file...')
        df = vaex.from_csv(file)
    return df

def binary_to_df(file,filePattern):
    """Convert any binary formats into vaex dataframe (.arrow, .parquet, .hdf5, or .fits).
            
            Args:
                file (str): Path to input file.
                filePattern (str): extension of file to convert.
                
            Returns:
                Vaex dataframe.
                
    """
    binary_patterns = [".fits", ".arrow", ".parquet", ".hdf5"]

    logger.info('BINARY FILE: Scanning directory for binary file pattern... ')
    if filePattern in binary_patterns:
        #convert hdf5 to vaex df
        df = vaex.open(file)
        return df
    else:
        raise FileNotFoundError('No supported binary file extensions were found in the directory. Please check file directory again.' )
    
def fcs_to_feather(file,outDir):
    """Convert fcs file to csv. Copied from polus-fcs-to-csv-converter plugin.
            
        Args:
            file (str): Path to the directory containing the fcs file.
            outDir (str): Path to save the output csv file.
                
        Returns:
            Converted csv file.
                
    """
    file_name = Path(file).stem
    feather_filename = file_name + ".feather"
    logger.info('FCS CONVERSION : Begin parsing data out of .fcs file' + file_name)
    
    #Use fcsparser to parse data into python dataframe
    meta, data = fcsparser.parse(file, meta_data_only=False, reformat_meta=True)
    
    #Export the fcs data to vaex df
    logger.info('FCS CONVERSION: converting data to vaex dataframe...')
    df = vaex.from_pandas(data)
    logger.info('writing file...')
    os.chdir(outDir)
    logger.info('DF to Feather: Writing Vaex Dataframe to Feather File Format for:' + file_name)
    df.export_feather(feather_filename,outDir)
    
    

def df_to_feather(input_file,filePattern,outDir):
    """Convert vaex dataframe to Arrow feather file.
                
        Args:
            inpDir (str): Path to the directory to grab file.
            filepattern (str): File extension.
            outDir (str): Path to the directory to save feather file.
                    
        Returns:
            Feather format file.
                    
    """  
    file_name = Path(input_file).stem
    output_file = file_name + ".feather"
    logger.info('DF to Feather: Scanning input directory files... ')
    if filePattern == ".csv":
        #convert csv to vaex df or hdf5
        df = csv_to_df(input_file)    
    else:
        df = binary_to_df(input_file,filePattern)  
        
    logger.info('writing file...')
    os.chdir(outDir)
    logger.info('DF to Feather: Writing Vaex Dataframe to Feather File Format for:' + file_name)
    df.export_feather(output_file,outDir)

def main(inpDir: Path,
         filePattern: str,
            outDir: Path,
            ) -> None:
        """ Main execution function
        
        """
        logger.info('outDir = {}'.format(outDir))
        logger.info('inpDir = {}'.format(inpDir))
        logger.info('filePattern = {}'.format(filePattern))
        
        if filePattern is None:
            filePattern = '.*'
        
        input_dir = Path(inpDir)
        input_file_list = list(Path(input_dir).glob('*' + filePattern))
        
        print(input_file_list)
        
        for file in input_file_list:
            if filePattern == '.fcs':
                logger.info('Converting fcs file to csv ')
                fcs_to_feather(file, outDir)
            else:
                df_to_feather(file,filePattern,outDir)
        
    
            logger.info("Finished all processes!")

if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='WIPP plugin to converts Tabular Data to Feather file format.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input general data collection to be processed by this plugin', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='File Extension to convert into Feather file format', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    
    
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    
    main(inpDir=inpDir,
         filePattern = filePattern,
         outDir=outDir)