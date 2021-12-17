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
    """Extract data from .ims file
    
    This class locates and extracts needed data from the following 
    attributes of the .ims file: ``Factor``, ``StatisticsType``, 
    ``StatisticsValue``, and ``Category``. The .ims file is of the hdf5 
    filetype, and it acts like a dictionary. Data is stored in 
    intermediate files **object_df.csv** and **track_df.csv**
    """
    
    def __init__(self, ims_filename, dir_name, logger=None):
        """Open .ims file for reading.

        Args:
            ims_filename (:obj:`str`): Selected .ims filename
            dir_name (:obj:`str`): Output csv collection
        """

        self.ims_filename = ims_filename
        self.dir_name = dir_name
        self.f = h5py.File(self.ims_filename, 'r')

        #: Set up the logger
        logging.basicConfig(
            format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', 
            datefmt='%b-%d-%y %H:%M:%S')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def get_factor(self, h5py_file, chan_name):
        """Save Factor data to dataframes.
        
        The h5py_file is a file object that serves as entry point into 
        Imaris file. The chan_name variable represents each attribute 
        name, located in the keys of the ``Scene8/Content`` folder in 
        the .ims file. Surface channel types are located in attributes 
        prefixed with 'MegaSurfaces' and Spot channel types are in 
        attributes prefixed with 'Points'. Within each chan_name might 
        be a ``Factor`` attribute. This function returns a Pandas 
        dataframe containing required data within the ``Factor`` 
        attribute of the Imaris metadata. Specifically, this opens the 
        ims file and places the contents of the attribute 
        ``Scene8/Content/.../Factor`` into a dataframe. 
        If ``Factor`` does not exist, an empty dataframe for factor is 
        created. The Factor df is rearranged so that the first column 
        ('ID List' becomes the index and items in the 'Name' column 
        (Collection, Channel, Image, Overall) become new columns.

        Args:
            h5py_file: Opened .ims file object
            chan_name (:obj:`str`): attribute name in ``Scene8/Content``

        Returns:
            Pandas dataframe containing ``Factor`` data
        """
        #: Navigate to the Factor attribute of the hdf5 file
        factors = h5py_file['Scene8']['Content'][chan_name]['Factor'][()]
        
        #: Create df of Factor data
        factor_cols = ['ID_List', 'Name', 'Level']
        df_factor = pd.DataFrame.from_records(factors, columns=factor_cols) 
        
        #: Create empty factor df if Factor data not in channel
        if df_factor.empty: 
            df_factor = pd.DataFrame(
                [[0, 'Overall', 'Overall']], 
                columns = ['ID_List', 'Name', 'Level'])
            df_factor.index = df_factor.index + 1

        #: Create empty final row in df_factor if factor df has data
        elif not df_factor.empty:
            df_factor.loc[-1] = [0, None, None]
            df_factor.index = df_factor.index + 1
            df_factor = df_factor.sort_index()
        
        # Pivot w/ ID as index to make image/channel/collection new cols
        df_factor = df_factor.pivot(index='ID_List', columns='Name')['Level']

        #: Drop empty column created by pivoting
        df_factor.dropna(axis=1, how='all', inplace=True)

        #: Add 'Image', 'Channel', or 'Collection' col if not in Factor
        if str.encode('Image') not in df_factor:
            df_factor[str.encode('Image')] = None
        if str.encode('Channel') not in df_factor:
            df_factor[str.encode('Channel')] = None
        if str.encode('Collection') not in df_factor:
            df_factor[str.encode('Collection')] = None

        return df_factor
    
    def get_statisticstype(self, h5py_file, chan):
        """Combine ``StatisticsType`` and ``Category`` and return as df.

        This function reads the attributes 'Category' and 
        'StatisticsType' of the .ims file. Then, it associates feature 
        information, which is stored in the StatisticsType attribute, 
        with data in the Category attribute, which contains info data 
        types (Surface, Track, or Overall). These data types are useful 
        for later understanding which values should be stored in 
        Overall.xlsx.

        Args:
            h5py_file: Opened .ims file object
            chan (:obj:`str`): attribute name in ``Scene8/Content``
        
        Returns:
            Pandas df with ``StatisticsType`` and ``Category`` data
        """
        stat_types=h5py_file['Scene8']['Content'][chan]['StatisticsType'][()]
        
        df_statistics_type = pd.DataFrame.from_records(
            stat_types, 
            columns=['ID', 'ID_Category', 'ID_FactorList', 'Name', 'Unit'])

        df_statistics_type.drop_duplicates(
            ['ID', 'ID_Category', 'ID_FactorList', 'Name', 'Unit'], 
            keep='first', inplace=True)
        
        df_statistics_type.sort_values(
            by=['Name'], ascending=True, inplace=True)

        #: Store 'Category' in df (indicates Surface, Track, or Overall)
        categories = h5py_file['Scene8']['Content'][chan]['Category'][()]
        df_category = pd.DataFrame.from_records(
            categories, columns=['ID', 'CategoryName', 'Name'])
        
        #: 'Name' and 'CategoryName' columns are copies in the file
        df_category.rename(
            columns={'ID':'catID', 'Name':'redundant_cat_name'}, 
            inplace=True)
        
        #: Associate category names to features (in statistics_type)
        df_statistics_type = pd.merge(
            df_category, df_statistics_type, 
            left_on = 'catID', right_on = 'ID_Category', how = 'outer')
        
        #: Remove unneeded cols
        df_statistics_type.drop(
            ['catID', 'redundant_cat_name', 'ID_Category'] , 
            axis=1, inplace=True)

        return df_statistics_type

    def convert_byte_to_string_and_format(self, f_st):
        """Clean/decode/format ``Factor``/``StatisticsType``/``Cat``.
        
        This function formats a merged dataframe containing ``Factor``, 
        ``StatisticsType``, and ``Category`` data by decoding byte data, 
        replacing special characters, and joining data from `Channel`, 
        `Image`, `Unit`, and `Name` columns into a single column. It 
        takes in input_df as an argument, which is a dataframe 
        containing data from the ``Factor``, ``Category``, and 
        ``StatisticsType`` attributes of the .ims file. It adds 
        substrings that enable the data to be readable when the columns
        are combined in a later step. It returns a dataframe containing 
        ``Factor`` and ``StatisticsType`` data with UTF-8 formatting, 
        no special characters, no excess columns, and an updated `Name` 
        column that includes units of measurement, channel and image 
        information.

        Args:
            f_st: ``Factor`` and ``StatisticsType`` merged dataframe

        Returns:
            Pandas df with cleaned feature and data type information
        """
        
        #: Convert cols from Factors/StatisticsType df to string
        f_st[str.encode('Channel')]=f_st[str.encode('Channel')].str.decode(
            'utf-8')

        f_st[str.encode('Image')]=f_st[str.encode('Image')].str.decode('utf-8')
        
        #: Convert b'Channel' col from byte to string; prepend _Channel_
        f_st[str.encode('Channel')]='_Channel_'+f_st[str.encode('Channel')] 

        #: Add substrings so cols have separators when combined later
        f_st[str.encode('Image')] = "_" + f_st[str.encode('Image')]
        f_st['Name'] = f_st['Name'].str.decode("utf-8")
        f_st['Unit'] = "_" + f_st['Unit'].str.decode("utf-8")

        #: Convert np.nan to empty string to combine feature names/units
        f_st.replace(np.nan, '', regex=True, inplace=True)

        #: Append unit, channel, and image information to 'Name' column
        f_st['Name']=f_st['Name']+f_st['Unit']+f_st[str.encode(
            'Image')]+f_st[str.encode('Channel')]

        #: Remove columns that are no longer required.
        f_st.drop(
            ['Unit', str.encode('Channel'), str.encode('Image'), 
            str.encode('Collection')], axis=1, inplace=True)

        #: Replace special characters in feature names
        f_st.replace({' ': '_', '°': '_deg_'}, inplace=True)
        f_st.replace(r'\/', '_per_', regex=True, inplace=True)
        f_st.replace(r'\^2', '_sqd_', regex=True, inplace=True)
        f_st.replace(r'\^3', '_cubed_', regex=True, inplace=True)
        f_st.replace('[^0-9a-zA-z]+', '_', regex=True, inplace=True)
        f_st.replace('', np.nan, regex=True, inplace=True)

        return f_st

    def merge_stat_value(self, df, channel_name):
        """
        Merge/organize StatisticsValue, StatisticsType/Cat, and Factor.

        This function combines data from the ``StatisticsValue`` 
        attribute of the .ims file with the dataframe containing merged 
        ``Factor``, ``StatisticsType``, and ``Category`` data. Then, it 
        isolates and organizes overall, track, and object datatypes by
        ID. Track IDs get shifted to a new `TrackID` column, and 
        object IDs remain in `ID_Object` column. Overall data remains 
        in `ID_Object` column but gets assigned an `ID_Object` value of 
        -1. Merge StatisticsValue with feature names, which are located 
        in the Factor/StatisticsType df; In this step, Track IDs and 
        Object IDs get separated using factor_statisticstype column 
        labeled 'CategoryName'. It returns a Pandas dataframe containing 
        ``StatisticsValue``, ``Factor``, ``Category``, and 
        ``StatisticsType`` data, with separation of track, object, and 
        overall data.

        Args:
            df: Merged``Factor`` and ``StatisticsType`` dataframe
            channel_name (:obj:`str`): Attribute in ``Scene8/Content``

        Returns:
            Dataframe with separation of track, object, and overall.
        """
        
        #: Get StatisticsValue 
        statistics_value = pd.DataFrame.from_records(
            self.f['Scene8']['Content'][channel_name]['StatisticsValue'][()], 
            columns=['ID_Time', 'ID_Object', 'ID_StatisticsType', 'Value']) 
        
        #: Merge StatisticsValue with remaining data, join on ID (index) 
        df = pd.merge(
            statistics_value, df.set_index('ID'),
            left_on='ID_StatisticsType', right_index=True)
        
        #: Kiss&Run ext. stores in 'Overall' instead of 'Category' col
        if str.encode('Overall') in df.columns:
            
        #: Set ID equal to -1 for 'Overall' rows
            df.loc[df[str.encode(
                'Overall')] == str.encode('Overall'), 'ID_Object'] = -1
        
        #: Make 'TrackID' column, even if track data absent
        df["TrackID"] = None
        
        #: Move IDs from 'ID_Object' to 'TrackID' col if CatName==Track
        df.loc[df['CategoryName'] == str.encode(
            'Track'), 'TrackID'] = df['ID_Object']

        #: If CatName==b'Track, remove TrackID data from ID_Object col
        df.loc[df['CategoryName'] == str.encode('Track'), 'ID_Object'] = None

        #: Set ID_Object to -1 where CategoryName == Overall.
        df.loc[df['CategoryName'] == str.encode('Overall'), 'ID_Object'] = -1

        return df
    
    def create_object_csv(self, df, chan):
        """
        After moving track data IDs to the `TrackID` column, this 
        isolates non-track data by copying rows where `TrackID` is null 
        into a new dataframe and storing the result in an intermediate 
        file called ``objectdf_channel_name.csv``. The arguments are df, 
        which is a dataframe of data merged from ``StatisticsValue``, 
        ``StatisticsType``, and ``Factor`` attributes of Imaris file, 
        and chan, which represents each attribute located within the 
        keys of the ``Scene8/Content`` groups of the Imaris file. 
        For example, ``MegaSurfaces0`` or ``Points0``

        Args:
            df: DF of StatisticsValue, StatisticsType, Category, Factor
            chan (:obj:`str`): attribute name in ``Scene8/Content``
        """
        
        #: Move rows where TrackID is empty to object_df
        object_df=df.loc[df.TrackID.isnull()].copy()
        object_df.drop(
            columns=['ID_StatisticsType', 'ID_FactorList'], 
            axis=1, inplace=True)
        
        #: Select 1st row where ID_Object (-1)/Name/Time/TrackID is same
        object_df.drop_duplicates(
            subset=['Name', 'ID_Object', 'ID_Time', 'TrackID'], 
            keep='first', inplace=True)
        object_df = object_df.pivot_table(
            index=['ID_Object', 'ID_Time'], columns='Name', 
            values='Value', fill_value=None)
        object_df.reset_index(inplace=True)
        
        #: Convert to csv
        temp_filename = "objectdf_" + chan + ".csv"
        temp_path = self.dir_name/temp_filename
        object_df.to_csv(temp_path)

    def create_track_csv(self, df, chan):
        """Stores Track data in trackdf_chan.csv.
        
        After moving track data IDs to the `TrackID` column,
        this function corrects isolates track data by copying rows 
        where `TrackID` is not null into a new dataframe. Then, 
        because TrackID dataframe lacks the correct time (all values are 
        set to -1 seconds), the function drops -1. These times are 
        then replaced using data that has object ids linked to 
        correct time. The result is stored in an 
        intermediate file called ``trackdf_chan.csv``

        Args:
            df: DF of StatisticsValue, StatisticsType, Category, Factor
            chan (:obj:`str`): attribute name in ``Scene8/Content``
        """
        #: Move rows where TrackID is non-empty to track_df dataframe
        track_df=df.loc[df.TrackID.notnull()].copy()
        
        #: Fill track_df with missing empty ID_Object values
        track_df = track_df.drop('ID_Object',1) 
        
        #: Correct Track data time
        track_df.drop('ID_Time', axis=1, inplace=True)

        track_df.drop(
            columns=['ID_StatisticsType', 'ID_FactorList'], 
            axis=1, inplace=True)

        track_df = track_df.pivot_table(
            index='TrackID', columns='Name', values='Value')
        
        #: Convert to csv
        temp_filename = "trackdf_" + chan + ".csv"
        temp_path = self.dir_name/temp_filename
        track_df.to_csv(temp_path)

    def link_data_fun(self):
        """Main function that makes Track and Overall intermediate csv
        
        This is the main function that calls all remaining functions 
        within this class. It extracts data from the Imaris file, 
        cleans and organizes data using Pandas dataframes, and outputs 
        two intermediate csv files for both track and non-track info 
        that gets read in by the next module, ``link_ims_ids.py``. Note
        that channel_names can start with either 'Points' 
        (signifies spot channel types) or 'MegaSurfaces' (signifies 
        surface types).
        """
        
        #: Store attribute names (contain channel info) in channel_names
        self.logger.debug("Counting channels in Scene8/Content...")
        self.logger.info(
            'Processing file {} stage 1/3...'.format(str(self.ims_filename)))
        channel_names_all = list(self.f['Scene8']['Content'].keys())
        channel_names = []

        #: Ignore non-spot, non-surface channels.
        for channel in channel_names_all:
            if channel.startswith('Points') or channel.startswith(
                'MegaSurfaces'):
                channel_names.append(channel)

        for i in range(0,len(channel_names)):
            #: Loop through each attribute in Scene8/Content/
            self.logger.debug(
                '\n\nITERATION {}/{} OF FILE {}'.format(
                    i+1, len(channel_names), self.ims_filename))
            
            current_channel = channel_names[i]
            reading_chan = "Reading " + current_channel + "..."
            self.logger.debug(reading_chan)

            #: Check if attributes StatisticsType/StatisticsValue exist
            contains_stat_type = (
                self.f['Scene8']['Content'][current_channel]).__contains__(
                    'StatisticsType')

            contains_stat_val = (
                self.f['Scene8']['Content'][current_channel]).__contains__(
                    'StatisticsValue')
            
            if contains_stat_type == True and contains_stat_val == True:

                #: Merge Rearranged Factors and StatisticsType
                statisticstype_df = self.get_statisticstype(
                    self.f, current_channel) 
                factor_df = self.get_factor(self.f, current_channel)

                factor_statisticstype = pd.merge(
                    factor_df, statisticstype_df, 
                    left_on='ID_List', right_on='ID_FactorList', how='outer')

                factor_statisticstype = self.convert_byte_to_string_and_format(
                    factor_statisticstype)

                #: Separate Track IDs/Object IDs using CategoryName col
                statisticsvalue_statisticstype_factor=self.merge_stat_value(
                    factor_statisticstype, current_channel)
                
                #: Create csvs for Track data and Non-Track ("object")
                self.create_object_csv(
                    statisticsvalue_statisticstype_factor, current_channel)

                self.create_track_csv(
                    statisticsvalue_statisticstype_factor, current_channel)   
            
            #: Skip attrs w/o StatisticsType/StatisticsValue (no data)
            else:
                pass 