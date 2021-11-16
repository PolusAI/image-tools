from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import fcsparser
import filepattern
import os
import argparse
import logging
import vaex
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
import pyarrow.csv as csv
import shutil

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def feather_to_tabular(file, filePattern, outDir):
    """Converts feather file into tabular file using pyarrow
            
            Args:
                file (str): Path to input file.
                filePattern (str): Filepattern of desired tabular output file
                
            Returns:
                Tabular File
                
    """
    #Get filename for output file
    file_name = Path(file).stem
    pq_file = file_name + ".parquet"
    csv_file = file_name + ".csv"
    # outputF = os.path.join(outDir, pq_file)
    logger.info('Feather CONVERSION: Copy file into outDir for processing...')
    output = file_name + ".feather"
    outputfile = os.path.join(outDir, output)
    shutil.copyfile(file, outputfile)
    
    logger.info('Feather CONVERSION: Converting file into Vaex DF')
    # Result is vaex dataframe
    if Path(outputfile).suffix == '.feather':
        df = pf.read_table(outputfile)
        
        if filePattern == ".csv":
            # Streaming contents of Arrow Table into csv
            logger.info('Feather CONVERSION: converting arrow table into .csv file')
            with csv.CSVWriter(csv_file, df.schema) as writer:
                writer.write_table(df)
        elif filePattern ==".parquet":
            logger.info('Feather CONVERSION: converting arrow table into .parquet file')
            pq.write_table(df, pq_file)
        # If neither, log error
        else:
            logger.error('Feather CONVERSION Error: This filePattern is not supported in this plugin')
    
    
         
def main(inpDir: Path,
         filePattern: str,
            outDir: Path,
            ) -> None:
        """ Main execution function
        
        """
        
        if filePattern is None:
            filePattern = '.*'
        
        input_dir = Path(inpDir)
        # input_file_list = list(Path(input_dir).glob('*' + filePattern))
        
        for file in input_dir.iterdir():
            feather_to_tabular(file, filePattern, outDir)
        
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
    
    