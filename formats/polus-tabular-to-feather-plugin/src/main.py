from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import fcsparser
import filepattern
import os
import argparse
import logging
import vaex
import pyarrow.feather as feather
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)


def csv_to_feather(file):
    """Converts csv into feather file using pyarrow.CSVStreamingReader
            
            Args:
                file (str): Path to input file.
                
            Returns:
                Feather file
                
    """
    file_name = Path(file).stem
    feather_file = file_name + ".feather"
    outputF = os.path.join(outDir, feather_file)
    logger.info('CSV CONVERSION: Checking size of csv file...')
    # Open csv file and count rows in file
    with open(file,'r', encoding='utf-8') as fr:
        ncols = len(fr.readline().split(','))
    
    chunk_size = max([2**24 // ncols, 1])
    logger.info('CSV CONVERSION: # of columns are: ' + str(ncols))
    
    arrow_schema = pa.Schema
    # Preparing convert options
    co = pyarrow.csv.ConvertOptions(column_types=arrow_schema)

    # Preparing read options
    ro = pyarrow.csv.ReadOptions(block_size=chunk_size)
    
    # Streaming contents of CSV into batches
    logger.info('CSV CONVERSION: converting csv into arrow table')
    with pyarrow.csv.open_csv(file,read_options=ro, convert_options=co) as stream_reader:
        for chunk in stream_reader:
            if chunk is None:
                break

            # Emit batches from generator. Arrow schema is inferred unless explicitly specified
            ds = pa.Table.from_batches(batches=[chunk])
    
    # Convert arrow table to feather file
    
    feather.write_feather(ds,outputF)

def binary_to_feather(file,filePattern):
    """Convert any binary formats into feather file via vaex (.arrow, .parquet, .hdf5, or .fits).
            
            Args:
                file (str): Path to input file.
                filePattern (str): extension of file to convert.
                
            Returns:
                feather file.
                
    """
    binary_patterns = [".fits", ".arrow", ".parquet", ".hdf5"]
    file_name = Path(file).stem
    output_file = file_name + ".feather"
    logger.info('BINARY FILE: Scanning directory for binary file pattern... ')
    if filePattern in binary_patterns:
        #convert hdf5 to vaex df
        df = vaex.open(file)
    else:
        raise FileNotFoundError('No supported binary file extensions were found in the directory. Please check file directory again.' )
    
    logger.info('writing file...')
    os.chdir(outDir)
    logger.info('DF to Feather: Writing Vaex Dataframe to Feather File Format for:' + file_name)
    df.export_feather(output_file,outDir)
    
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
        
def main(inpDir: Path,
         filePattern: str,
            outDir: Path,
            ) -> None:
        """ Main execution function
        
        """
        
        if filePattern is None:
            filePattern = '.*'
        
        input_dir = Path(inpDir)
        input_file_list = list(Path(input_dir).glob('*' + filePattern))
        
        for file in input_file_list:
            if filePattern == '.fcs':
                logger.info('Converting fcs file to csv ')
                fcs_to_feather(file, outDir)
            elif filePattern == '.csv':
                csv_to_feather(file)
            else:
                binary_to_feather(file,filePattern)
        
        # remove_files(outDir)
        
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
    
    