import csv
import gc
import h5py
import logging
import numpy as np
import os
import pandas as pd
import re
import string
import time

class LinkData:
    """Extract data from ``Factor``, ``StatisticsType``, ``StatisticsValue``, and ``Category`` hdf5 groups to produce intermediate files **object_df.csv** and **track_df.csv**
    """
    def __init__(self, ims_filename, dir_name, logger=None):
        """Open .ims file for reading; h5py.File acts like a dictionary

        Args:
            ims_filename (:obj:`str`): Name of the selected Imaris file
            dir_name (:obj:`str`): Output csv collection provided by user
        """

        self.ims_filename = ims_filename
        self.dir_name = dir_name
        self.f = h5py.File(self.ims_filename, 'r')

        #: Set up the logger
        logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%b-%d-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def get_factor(self, opened_h5py_file, channel_name):
        """Open ims file and place contents of ``Scene8/Content/.../Factor`` group into a dataframe. If ``Factor`` does not exist, create an empty factor dataframe.

        Args:
            opened_h5py_file: File object that serves as entry point into Imaris file, which is of the HDF5 filetype (files with .ims extension are HDF5 files)
            channel_name (:obj:`str`): Within the keys of the ``Scene8/Content`` attributes of the Imaris files, there are attributes with names such as ``MegaSurfaces0`` or ``Points0``; channel_name represents each attribute name.

        Returns:
            Pandas dataframe containing required data within the ``Factor`` attribute of the Imaris file's metadata
        """

        factors = opened_h5py_file['Scene8']['Content'][channel_name]['Factor'][()] #: Navigate to the Factor attribute of the hdf5 file
        df_factor = pd.DataFrame.from_records(factors, columns=['ID_List', 'Name', 'Level']) #: Set columns to same columns in original Factor table of ims file
        
        #: Create ID_List value 0 for non-matches
        if df_factor.empty: #: If factor does not exist in the hdf5 file, create a df anyways with the same same headers and data 0, "Overall", "Overall"
            df_factor = pd.DataFrame([[0, 'Overall', 'Overall']], columns = ['ID_List', 'Name', 'Level'])
            df_factor.index = df_factor.index + 1

        elif not df_factor.empty:
            #: Create empty row (loc[-1] means last row) in df_factor
            df_factor.loc[-1] = [0, None, None]
            df_factor.index = df_factor.index + 1
            df_factor = df_factor.sort_index()
        
        '''Rearrange Factor df so that ID List becomes index, 
        items under Name (such as Collection, Channel, Image, 
        Collection, Overall) become new column labels and 
        corresponding "Level" data fills new columns'''

        df_factor = df_factor.pivot(index='ID_List', columns='Name')['Level']
        df_factor.dropna(axis=1, how='all', inplace=True) #: Drop empty column created by pivoting

        #: Not all Factor have b'Image', b'Channel, or b'Collection. If not present, the column gets added here
        if str.encode('Image') not in df_factor:
            df_factor[str.encode('Image')] = None
        if str.encode('Channel') not in df_factor:
            df_factor[str.encode('Channel')] = None
        if str.encode('Collection') not in df_factor:
            df_factor[str.encode('Collection')] = None
        return df_factor
    
    def get_statisticstype(self, opened_h5py_file, channel_name):
        """Combine hdf5 attributes ``StatisticsType`` and ``Category`` and return as df

        Args:
            opened_h5py_file: File object that serves as entry point into Imaris file, which is of the HDF5 filetype (files with .ims extension are HDF5 files)
            channel_name (:obj:`str`): Within the keys of the ``Scene8/Content`` attributes of the Imaris files, there are attributes with names such as ``MegaSurfaces0`` or ``Points0``; channel_name represents each attribute name.

        Returns:
            Pandas dataframe containing required data within the ``StatisticsType`` and ``Category`` attributes of the Imaris file's metadata
        """
        statistics_types = opened_h5py_file['Scene8']['Content'][channel_name]['StatisticsType'][()]
        df_statistics_type = pd.DataFrame.from_records(statistics_types, columns=['ID', 'ID_Category', 'ID_FactorList', 'Name', 'Unit'])
        df_statistics_type.drop_duplicates(['ID', 'ID_Category', 'ID_FactorList', 'Name', 'Unit'], keep='first', inplace=True)
        df_statistics_type.sort_values(by=['Name'], ascending=True, inplace=True)

        #: Save data in Category section of hdf5 file as df (indicates if row of data is Surface, Track, or Overall)
        categories = opened_h5py_file['Scene8']['Content'][channel_name]['Category'][()]
        df_category = pd.DataFrame.from_records(categories, columns=['ID', 'CategoryName', 'Name'])
        df_category.rename(columns={'ID':'catID',
                                    'Name':'redundant_cat_name'}, #: "redundant_cat_name" naming given because 'Name' and 'CategoryName' columns are copies in the hdf5 file
                                    inplace=True)
        
        #: Merge category information with statistics_type data
        df_statistics_type = pd.merge(df_category, df_statistics_type, left_on = 'catID', right_on = 'ID_Category', how = 'outer')
        
        #: Remove columns that are no longer needed, now that correct data is associated (category names to feature)
        df_statistics_type.drop(['catID', 'redundant_cat_name', 'ID_Category'] , axis=1, inplace=True)

        return df_statistics_type

    def convert_byte_to_string_and_format(self, input_df):
        """Formats dataframe containing ``Factor`` and ``StatisticsType`` data by decoding byte data, replacing special characters, and combining data from `Channel`, `Image`, `Unit`, and `Name` columns into a single column.

        Args:
            input_df: Dataframe containing data from ``Factor`` and ``StatisticsType`` attributes of Imaris file.

        Returns:
            Dataframe containing ``Factor`` and ``StatisticsType`` data with UTF-8 formatting, no special characters, no excess columns, and updated `Name` column that includes units of measurement, channel and image info.
        """
        
        #: Convert columns from Factors and StatisticsType merged dataframe from byte to string
        input_df[str.encode('Channel')] = input_df[str.encode("Channel")].str.decode("utf-8")
        input_df[str.encode('Image')] = input_df[str.encode("Image")].str.decode("utf-8")
        input_df[str.encode('Channel')] = '_Channel_' + input_df[str.encode('Channel')] #: Convert from byte to string, and prepend _Channel_
        input_df[str.encode('Image')] = "_" + input_df[str.encode('Image')]
        input_df['Name'] = input_df['Name'].str.decode("utf-8")
        input_df['Unit'] = "_" + input_df['Unit'].str.decode("utf-8")

        #: Convert np.nan to empty string to combine feature names to units (string + string works; string + nan yields error)
        input_df.replace(np.nan, '', regex=True, inplace=True)
        input_df['Name'] = input_df['Name'] + input_df['Unit'] + input_df[str.encode('Image')] + input_df[str.encode('Channel')]

        #: Now that unit, channel, and image information have been appended to feature Name, these columns are no longer required.
        input_df.drop(['Unit', str.encode('Channel'), str.encode('Image'), str.encode('Collection')], axis=1, inplace=True)

        #: Replace special characters in feature names
        input_df.replace({' ': '_', '°': '_deg_'}, inplace=True)
        input_df.replace(r'\/', '_per_', regex=True, inplace=True)
        input_df.replace(r'\^2', '_sqd_', regex=True, inplace=True)
        input_df.replace(r'\^3', '_cubed_', regex=True, inplace=True)
        input_df.replace('[^0-9a-zA-z]+', '_', regex=True, inplace=True)
        input_df.replace('', np.nan, regex=True, inplace=True)

        return input_df

    def merge_statistics_value(self, df, channel_name):
        """Extracts and merges data from ``StatisticsValue`` attribute of Imaris file with input dataframe containing ``Factor`` and ``StatisticsType`` data. Then, isolates and organizes overall, track, and object data by ID. Track IDs get shifted to a new `TrackID` column, and object IDs remain in `ID_Object` column. Overall data remains in `ID_Object` column but gets assigned an `ID_Object` value of -1.

        Args:
            df: Dataframe containing data from ``Factor`` and ``StatisticsType`` attributes of Imaris file.
            channel_name (:obj:`str`): Within the keys of the ``Scene8/Content`` attributes of the Imaris files, there are attributes with names such as ``MegaSurfaces0`` or ``Points0``; channel_name represents each attribute name.

        Returns:
            Dataframe containing ``StatisticsValue``, ``Factor``, and ``StatisticsType`` data, with separation of track, object, and overall data.
        """
        
        #: Get StatisticsValue 
        statistics_value = pd.DataFrame.from_records(self.f['Scene8']['Content'][channel_name]['StatisticsValue'][()], columns=['ID_Time', 'ID_Object', 'ID_StatisticsType', 'Value']) 
        
        #: Merge StatisticsValue with the dataframe that merged Factor to StatisticsType. Using ID as the index, join on the index.
        df = pd.merge(statistics_value, df.set_index('ID'), left_on='ID_StatisticsType', right_index=True)
        
        #: If this is a special file, as in Kiss and Run file, with b'Overall' as a column...
        if str.encode('Overall') in df.columns:
            
        #: Find all instances where df['Overall'] is equal to 'Overall' value using .loc, and assign your desired value, -1, in df['ID_Object'] at those indices: NOTE: feature name will be Overall in certain plugin modified files that did not store Overall info in "Category" sections
            df.loc[df[str.encode('Overall')] == str.encode('Overall'), 'ID_Object'] = -1
        
        #: Add 'TrackID' column for consistency, so even if input files lack track data, an output file for track data is still generated
        df["TrackID"] = None
        
        #: If 'CategoryName' == b'Track, copy the ID_Object from that row into the empty 'TrackID' column
        df.loc[df['CategoryName'] == str.encode('Track'), 'TrackID'] = df['ID_Object']

        #: If 'CategoryName' == b'Track, Remove the TrackID values from ID_Object column for those rows.
        df.loc[df['CategoryName'] == str.encode('Track'), 'ID_Object'] = None

        #: Set all ID_Object to -1 at all instances where CategoryName == Overall.
        df.loc[df['CategoryName'] == str.encode('Overall'), 'ID_Object'] = -1
        return df
    
    def create_object_csv(self, df, channel_name):
        """After moving track data IDs to the `TrackID` column, this isolates non-track data by copying rows where `TrackID` is null into a new dataframe and storing the result in an intermediate file called ``objectdf_channel_name.csv``

        Args:
            df: Dataframe of data merged from ``StatisticsValue``, ``StatisticsType``, and ``Factor`` attributes of Imaris file.
            channel_name (:obj:`str`): Represents each attribute located within the keys of the ``Scene8/Content`` groups of the Imaris file. For example, ``MegaSurfaces0`` or ``Points0``
        """
        
        #: Move rows where TrackID is empty to object_df
        object_df=df.loc[df.TrackID.isnull()].copy()
        object_df.drop(columns=['ID_StatisticsType', 'ID_FactorList'], axis=1, inplace=True)
        
        #: In case multiple rows contain identical ID_Object (-1), Name, Time, and TrackID, automatically select first duplicate, as Imaris does
        object_df.drop_duplicates(subset=['Name', 'ID_Object', 'ID_Time', 'TrackID'], keep='first', inplace=True)
        object_df = object_df.pivot_table(index=['ID_Object', 'ID_Time'], columns='Name', values='Value', fill_value=None)
        object_df.reset_index(inplace=True)
        
        #: Convert to csv
        object_df.to_csv(self.dir_name + "objectdf_" + channel_name + ".csv")

    def create_track_csv(self, df, channel_name):
        """After moving track data IDs to the `TrackID` column, this isolates track data by copying rows where `TrackID` is not null into a new dataframe and storing the result in an intermediate file called ``trackdf_channel_name.csv``

        Args:
            df: Dataframe of data merged from StatisticsValue, StatisticsType, and Factor attributes of Imaris file.
            channel_name (:obj:`str`): Within the keys of the ``Scene8/Content`` attributes of the Imaris files, there are attributes with names such as ``MegaSurfaces0`` or ``Points0``; channel_name represents each attribute name.
        """
        #: Move rows where TrackID is non-empty to track_df dataframe
        track_df=df.loc[df.TrackID.notnull()].copy()
        
        #: Fill track_df with missing empty ID_Object values
        track_df = track_df.drop('ID_Object',1) 
        
        #: TrackID dataframe lacks correct time (all -1), so drop -1 and replace with df2_time_object_id, which has object ids linked to correct time.
        track_df.drop('ID_Time', axis=1, inplace=True)
        track_df.drop(columns=['ID_StatisticsType', 'ID_FactorList'], axis=1, inplace=True)
        track_df = track_df.pivot_table(index='TrackID', columns='Name', values='Value')
        
        #: Convert to csv
        track_df.to_csv(self.dir_name + "trackdf_" + channel_name + ".csv")

    def link_data_fun(self):
        """This is the main function that calls all remaining functions within this class. It extracts data from the Imaris file, cleans and organizes data using Pandas dataframes, and outputs two intermediate csv files for both track and non-track information that gets read in by the next module, ``link_ims_ids.py``.
        """
        
        #: From .ims file, determine names of chunks that house each channel (ex: MegaSurfaces0, Points0, etc); store in channel_names array
        self.logger.debug("Counting channels in Scene8/Content...")
        self.logger.info('Processing file {} stage 1/3...'.format(str(self.ims_filename)))
        
        channel_names_all = list(self.f['Scene8']['Content'].keys())
        channel_names = []

        #: Ignore non-spot, non-surface channels. 'Points' prepends each spot channel. 'MegaSurfaces' prepends each surface channel.
        for channel in channel_names_all:
            if channel.startswith('Points') or channel.startswith('MegaSurfaces'):
                channel_names.append(channel)

        for i in range(0,len(channel_names)):
            #: Loop through each start in Scene8/Content/
            self.logger.debug('\n\nITERATION {}/{} OF FILE {}'.format(i+1, len(channel_names), self.ims_filename))
            current_channel = channel_names[i]
            reading_chan = "Reading " + current_channel + "..."
            self.logger.debug(reading_chan)

            #: Check if attributes StatisticsType and StatisticsValue exist in the .ims file
            contains_stat_type = (self.f['Scene8']['Content'][current_channel]).__contains__('StatisticsType')
            contains_stat_val = (self.f['Scene8']['Content'][current_channel]).__contains__('StatisticsValue')
            
            if contains_stat_type == True and contains_stat_val == True:

                #: Merge Rearranged Factors and StatisticsType
                statisticstype_df = self.get_statisticstype(self.f, current_channel) 
                factor_df = self.get_factor(self.f, current_channel)

                factor_statisticstype = pd.merge(factor_df, statisticstype_df, left_on='ID_List', right_on='ID_FactorList', how='outer')
                factor_statisticstype = self.convert_byte_to_string_and_format(factor_statisticstype)

                #: Merge StatisticsValue with feature names, which are located in the Factor/StatisticsType df; In this step, Track IDs and Object IDs get separated using factor_statisticstype column labeled 'CategoryName'
                statisticsvalue_statisticstype_factor = self.merge_statistics_value(factor_statisticstype, current_channel)
                
                #: Create final csv files for Track data and Non-Track ("object") data
                self.create_object_csv(statisticsvalue_statisticstype_factor, current_channel)
                self.create_track_csv(statisticsvalue_statisticstype_factor, current_channel)   
            
            else:
                pass #: Skips files that lack StatisticsType and StatisticsValue (no data to parse)