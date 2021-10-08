from bfio.bfio import BioReader, BioWriter
from pathlib import Path
import fcsparser
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

def fcs_csv(file,outDir):
        """Convert fcs file to csv. Copied from polus-fcs-to-csv-converter plugin.
            
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
    
def df_toFeather(file,outDir):
        """Convert a csv or fcs file to Arrow feather file. Includes code from polus-fcs-to-csv-converter plugin
                
            Args:
                file (str): Path to the directory containing the csv file.
                outDir (str): Path to the directory to save feather file.
                    
            Returns:
                Feather format file.
                    
            """
        file_name = Path(file).stem
        feather_filename = file_name + '.feather'
        
        logger.info('Started converting the csv file ' + file_name)
        df = vaex.from_csv(file)
        logger.info('writing file...')
        df.export_feather(feather_filename,outDir)

def main(inpDir: Path,
            outDir: Path,
            ) -> None:
        """ Main execution function
        
        All functions in your code must have docstrings attached to them, and the
        docstrings must follow the Google Python Style:
        https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
        """
        
        # pattern = filePattern if filePattern is not None else '.*'
        # fp = filepattern.FilePattern(inpDir,pattern)
        
        
        logger.info('outDir = {}'.format(outDir))
        logger.info('inpDir = {}'.format(inpDir))
        
        # Surround with try/finally for proper error catching
        try:
            
            #List the files in the directory
            logger.info('Checking for .csv or .fcs files in the directory ')
            fcs_filelist = list(Path(inpDir).glob('*.fcs'))
            if not fcs_filelist:
                raise FileNotFoundError('No .fcs files were found in the directory. Please check file directory.' )
        
            for each_file in fcs_filelist:
                #Convert fcs to csv
                logger.info("Checking for fcs files...")
                file_ext = Path(each_file).suffix
                if file_ext == '.fcs':
                    logger.info('Converting fcs file to csv ')
                    each_file = fcs_csv(each_file, outDir)
        finally:    
            logger.info('Checking for .csv files in the directory ')
            filelist = list(Path(outDir).glob('*.csv'))
            if not filelist:
                raise FileNotFoundError('No .csv files were found in the directory. Please check file directory.' )
                
            for each_file in filelist:
                #Read the csv file into feather
                df_toFeather(each_file,outDir)
                
            logger.info("Finished all processes!")

if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='WIPP plugin to converts Tabular Data (.FCS and .CSV) to Feather file format.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=False)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(inpDir))
    
    
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    
    main(inpDir=inpDir,
         outDir=outDir)