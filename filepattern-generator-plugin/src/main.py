import argparse, logging, os
from curses.ascii import EM
from queue import Empty
import filepattern 
from pathlib import Path
from numpy import empty
import pandas as pd
from typing import Optional
import pyarrow.feather
import time



class Filepattern_Generator:
    def __init__(self, 
                inpDir:Path,
                pattern:Optional[str]= None,
                chunkSize:Optional[int]= 30,
                groupBy: Optional[str]= None
                ):
        """
        Parameters
        ----------
        inpDir : Path
            Input image collection
        pattern: : str, optional
            Filepattern regex to parse image files
        chunkSize: : int, optional
            Number of images to generate collective filepattern (default value=30)
        groupBy: : str, optional
            Select a parameter to generate filepatterns in specific order
      
        Returns
        -------
        A collection of generated filepatterns in either (CSV and feather) file format

        """
        self.inpDir=Path(inpDir)
        self.pattern=pattern
        self.chunkSize=chunkSize
        self.groupBy=groupBy
        if self.pattern is None:
            self.fileslist = [f.name for f in self.inpDir.iterdir() if f.with_suffix('.ome.tif')]
        else:
            self.fp = filepattern.FilePattern(self.inpDir, self.pattern,var_order='rxytpc')
            self.fileslist = [file[0] for file in self.fp(group_by=self.groupBy)]

        
    def batch_chunker(self):
        """This function uses List of image files and chunkSize and creates iterator
        Args:
            fileslist: List of Image file names.
            chunkSize (str): Number of images to generate collective filepattern

        Returns:
            Iterator
        """
        batch_iterator = iter(self.fileslist)
        while True:
            chunk = []
            try:
                for _ in range(self.chunkSize):
                    chunk.append(next(batch_iterator))
                yield chunk
            except StopIteration:
                if chunk:
                    yield chunk
                return


    def pattern_generator(self):
        """This function iterates over batches of image files and returns filepatterns for each batch and batch sizes
        Args:
            iterator: calling batch_chunker function
            pattern (str): Filepattern to parse image files

        Returns:
            pandas DataFrame
        
        """
        pf = self.batch_chunker() 
        df = []
        batch_size=[]
        for _, batch in enumerate(pf):
            batch_files = []
            for b in batch:
                batch_files.append(b)
            if self.pattern is None:
                pattern_regex = filepattern.infer_pattern(batch_files)
            else:
                pattern_regex = self.fp.output_name(batch_files)
            df.append(pattern_regex)
            batch_size.append(len(batch))
        prf = pd.DataFrame(list(zip(df, batch_size)), columns=['FilePattern', 'Batch_Size'])
        return prf

def saving_generator_outputs(x:pd.DataFrame, 
                             outDir:Path, 
                             outFormat: Optional[str] = 'csv'):

    """Saving Outputs to CSV/Feather file format
    Args:
        x: pandas DataFrame
        outDir : Path of Ouput Collection     
        outFormat: : Output Format of collective filepatterns. Only Supports (CSV and feather) file format. (default file format is CSV)
                    
    Returns:
        CSV/Feather format file
        """    
    if outFormat == 'feather':
        pyarrow.feather.write_feather(x, os.path.join(outDir, "pattern_generator.feather"))
    else:
        x.to_csv(os.path.join(outDir, "pattern_generator.csv"), index=False)
    return
  
def main(inpDir:Path,
         outDir:Path,
         pattern:str,
         chunkSize:int,
         groupBy:str,
         outFormat:str):
    starttime= time.time()

    logger.info(f'Start parsing Image Filenames') 
    assert inpDir.exists(), logger.info('Input directory does not exist')
    assert [f for f in os.listdir(inpDir) if f.endswith(POLUS_EXT)], logger.error('Image files are not recognized as ome.tif')
    fg = Filepattern_Generator(inpDir,pattern,chunkSize,groupBy)
    prf =fg.pattern_generator()
    logger.info(f'Generated patterns: {prf.head(30)}')
    saving_generator_outputs(prf, outDir, outFormat)
    logger.info(f'Saving the Outputs: pattern_generator{outFormat}') 
    logger.info('Finished all processes')
    endtime = (time.time() - starttime)/60
    logger.info(f'Total time taken to process all images: {endtime}')
    
if __name__=="__main__":

    #Import environment variables
    POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
    POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(POLUS_LOG)


    # ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Filepattern generator Plugin')    
    #   # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                            help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                            help='Output collection', required=True)
    parser.add_argument('--pattern', dest='pattern', type=str,
                            help='Filepattern regex used to parse image files', required=False)
    parser.add_argument('--chunkSize', dest='chunkSize', type=int, default=30,
                            help='Select chunksize for generating Filepattern from collective image set', required=False)
    parser.add_argument('--groupBy', dest='groupBy', type=str,
                            help='Select a parameter to generate Filepatterns in specific order', required=False)
    parser.add_argument('--outFormat', dest='outFormat', type=str, default='csv',
                            help='Output Format of this plugin. It supports only two file-formats: CSV & feather', required=False)
                            

    # # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)

    if (inpDir.joinpath('images').is_dir()):
        inpDir = inpDir.joinpath('images').absolute()
    logger.info('inputDir = {}'.format(inpDir))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    pattern = args.pattern
    logger.info('pattern = {}'.format(pattern))
    chunkSize=int(args.chunkSize)
    logger.info("chunkSize = {}".format(chunkSize))
    groupBy=str(args.groupBy)
    logger.info("groupBy = {}".format(groupBy))
    outFormat= args.outFormat
    logger.info("outFormat = {}".format(outFormat))


    main(inpDir=inpDir,
         outDir=outDir,
         pattern=pattern,
         chunkSize=chunkSize,
         groupBy=groupBy,
         outFormat=outFormat
         )
