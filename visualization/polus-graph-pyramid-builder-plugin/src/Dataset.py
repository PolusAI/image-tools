import os

import pandas as pd
import numpy as np

import itertools

import logging
# Initialize the logger
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("graphpyramid")
logger.setLevel(POLUS_LOG)


class csvDataset():

    """ This class solves for the necessary metadata to create histograms and 2D plots 
        from an input csv file 
    """

    def __init__(self, input_file):

        if type(input_file) == str:
            self.dataframe = pd.read_csv(input_file)
        else:
            self.dataframe = pd.DataFrame.from_dict(input_file)
        first_row_values = self.dataframe.iloc[0]

        if ('F' in first_row_values or 'C' in first_row_values) and len(first_row_values) <= 2:
            logger.debug("Data is Coded - it has F or C values for the first row")
            logger.info("Grabbing all the Data with 'F' header")
            removecols = self.dataframe.iloc[0] != 'F'

            logger.debug(f"Not Plotting Columns: \n{removecols}\n")
            self.dataframe = self.dataframe.drop(self.dataframe.columns[removecols], axis=1)
            self.dataframe = self.dataframe.drop(0)

        else:
            logger.debug("Data is Not Coded - it does not have F or C values for the first row")

        for column in self.dataframe.columns:
            try:
                self.dataframe[column] = self.dataframe[column].astype(np.float64)
            except:
                logger.info(f"dropping {column} - column could not be converted to np.float64")
                self.dataframe = self.dataframe.drop(column, axis=1)

        # 2D plots metadata
        self.column_names = self.dataframe.columns
        self.nfeats    = len(self.column_names)
        self.nexamples = self.dataframe.shape[0]
        self.ngraphs = self.nfeats*self.nfeats
        self.plot_combinations = [combo for combo in itertools.product(self.column_names, repeat=2)] 
        self.plot_combinations_unique = [combo for combo in itertools.combinations(self.column_names, 2)]
        self.stats = {'max' : self.dataframe.max(), 'min' : self.dataframe.min()}

    def __iter__(self):

        for plot_combo in self.plot_combinations:
            yield plot_combo
    
    def bin_data(self, bincount : int):
        """ This function bins the data for heatmaps

        Inputs:
            bincount - the number of bins
        """

        self.bincount = bincount

        self.stats['bin_min']   = self.dataframe.min() # needs to be redefined, because data might be logged
        self.stats['bin_max']   = self.dataframe.max() # needs to be redefined, because data might be logged
        self.stats['binwidth'] = (self.stats['bin_max']-self.stats['bin_min']+(10**-6))/self.bincount

        self.dataframe = ((self.dataframe - self.stats['bin_min'])/self.stats['binwidth']).apply(np.floor).astype(np.uint16)
        self.dataframe [self.dataframe  >= self.bincount] = self.bincount-1 


class LinearData(csvDataset):

    def __init__(self, input_file):
        csvDataset.__init__(self, input_file = input_file)
        self.scale = "linear"
        

class LogData(csvDataset):

    def __init__(self, input_file):
        csvDataset.__init__(self, input_file = input_file)
        self.scale = "log"
        self.C = 1/np.log(10)
        self.dataframe = np.sign(self.dataframe) * np.log10(1 + (abs(self.dataframe/self.C)))