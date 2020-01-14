import csv
import h5py
import logging
import logging.config
import numpy as np
import os
from os import path
import pandas as pd
import pathlib
import re
import xlsxwriter

class CreateCsv:
    """Merge linked track and object IDs to corresponding feature data to produce a csv output file that can be visualized in FlowJo.
    """
    def __init__(self, ims_filename, dir_name, meta_dir_name, logger=None):
        """Open .ims file for reading; h5py.File acts like a dictionary.

        Args:
            ims_filename (:obj:`str`): Name of the selected Imaris file
            dir_name (:obj:`str`): Output csv collection provided by user
            meta_dir_name (:obj:`str`): Output metadata directory provided by user
        """
        
        #: Set up the logger
        logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%b-%d-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.ims_filename = ims_filename
        self.dir_name = dir_name
        self.meta_dir_name = meta_dir_name
        self.f = h5py.File(self.ims_filename, 'r')

    def round_to_six(self, num):
        """Round values to six significant figures

        Args:
            num (:obj:`int`): The number that will be rounded to six significant figures

        Returns:
            A number rounded to six significant figures
        """

        if num != 0:
            if np.isnan(num) != True:
                num = np.round(num, -int(np.floor(np.log10(abs(num)))) + 5)
        elif num == 0:
            pass
        return num

    def get_df_from_csv(self, csv_collection, currentchannel, user_defined_channelname, csv_substring):
        """Read intermediate csv files containing feature data (``extract_ims_data.py`` output) or ID data (``link_ims_ids.py`` output) for each channel, and store in dataframes.

        Args:
            csv_collection (:obj:`str`): Output csv collection provided by user
            currentchannel (:obj:`str`): Within the keys of the ``Scene8/Content`` attributes of the Imaris files, there are attributes with names such as ``MegaSurfaces0`` or ``Points0``; channel_name represents each attribute name.
            user_defined_channelname (:obj:`str`): While currentchannel might have a more general name such as ``Points0``, the user-defined channel name gets extracted from the Imaris file's metadata, converted from byte to string, and stored as **user_defined_channelname**.
            csv_substring (:obj:`str`): The substring of the csv file that gets read in; can be `trackdf_` or `objectdf_` for outputs of first module (``extract_ims_data.py``), or empty string for outputs of second module, ``link_ims_ids.py`` (links `ID_Object` and `TrackID`).

        Returns:
            Pandas dataframe created from intermediate csv files.
        """

        keepsame = {'ID_Time'}
        #: Read Track csv, remove unnamed columns, append suffix containing channel name to all columns except ID_Object
        df = pd.read_csv(csv_collection + csv_substring + currentchannel + ".csv")

        #: Remove "Unnamed" columns:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        #: suffix channel name to all headers (excluding and 'ID_Object, 'ID_Time')
        if len(df.columns) > 0:

            #: Add suffix to column names except some columns (keep same = ID_Time)
            df.columns = ['{}{}'.format(c, '' if c in keepsame else user_defined_channelname) for c in df.columns]
            df.columns = df.columns.str.replace('__', '_')

        #: Remove intermediate csv file
        self.logger.debug("Remove line os.remove from merge_ids_to_features.py to debug. CSV files can indicate where in the process an issue begins.")
        os.remove(csv_collection + csv_substring + currentchannel + ".csv")

        return df
    
    def get_overall(self, overall_df, user_defined_channelname):
        """Reading the output from the ``get_df_from_csv()`` function, extract overall data from dataframe containing non-track data

        Note:
            In the second module, ``extract_ims_data.py``, data tagged `Overall` was assigned an `ID_Object` of -1

        Args:
            overall_df: dataframe of overall data, which is obtained from the **object_df** data where `ID_Object` is less than 0. 
            user_defined_channelname (:obj:`str`): Contains the user-defined channel name that gets extracted from the Imaris file's metadata.

        Returns:
            Dataframe containing overall data in similar format to Imaris-exported overall file
        """

        overall_df.dropna(axis=1, how='all', inplace=True)

        #: All ID_Objects equal to -1 belong to Overall. Replace with np.NaN
        overall_df['ID_Object' + user_defined_channelname] = overall_df['ID_Object' + user_defined_channelname].replace(-1.0, np.NaN, inplace=True)
        
        #: Replace time = -1.0 with np.NaN
        overall_df['ID_Time'].replace(-1.0, np.NaN, inplace=True)
        overall_df.reset_index()
        
        #: Rearrange dataframe to match exact format exported by Imaris file
        overall_df = pd.melt(overall_df, id_vars=['ID_Time', 'ID_Object' + user_defined_channelname], var_name='Variable', value_name='Value')
        overall_df = overall_df[['Variable', 'Value', 'ID_Time', 'ID_Object' + user_defined_channelname]]
        overall_df.rename({'ID_Time': 'Time', 'ID_Object' + user_defined_channelname: 'ID'}, axis='columns', inplace=True)
        overall_df.dropna(subset=['Value'], inplace=True)
        overall_df['Variable'] = overall_df['Variable'].str.replace('_', ' ')
        overall_df=overall_df.dropna(axis=1,how='all')
        return overall_df

    def create_overall_xlsx(self, imaris_filename, meta_directory, all_overall_dict):
        """Create overall xlsx file, with each sheet representing individual channels

        Args:
            imaris_filename (:obj:`str`): Filename of Imaris file
            meta_directory (:obj:`str`): Output metadata directory provided by user
            all_overall_dict: Dictionary of overall dataframes for all channels where key is user-defined channel name and value is overall dataframe
        """

        #: Merge all Overall DFs together and write each channel to an xlsx notebook represented using sheets to represent individual channels

        #: Get basename from imaris filename, to prepend to Overall.xlsx. stem takes the file basename from the path. 
        imaris_basename = imaris_filename.stem

        #: Remove .ims extension
        imaris_basename = imaris_basename[:-4]

        #: Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(meta_directory + imaris_basename + "_" + 'Overall.xlsx', engine='xlsxwriter')
        count = 1
        for chan_name, overall_df_list in all_overall_dict.items():
            for i in range(0, len(overall_df_list)):
                str_i = "_"
                if i >= 1:
                    str_i = "_" + str(i) + "_"
                str_channel_name = re.sub('[^A-Za-z0-9]+', '_', chan_name)

                #: Convert the dataframe to an XlsxWriter Excel object.
                str_channel_name_shortened = ""
                if len(str_channel_name) > 25:
                    str_channel_name_shortened = str_channel_name[:25]
                
                else:
                    str_channel_name_shortened = str_channel_name
                
                #: Round Overall "Values" column to 6 significant digits, as imaris does
                self.logger.debug("Converting data to 6 significant figures...")
                overall_df_list[i]['Value'] = overall_df_list[i]['Value'].apply(self.round_to_six)
                overall_df_list[i].to_excel(writer, sheet_name= str_channel_name_shortened + str_i + str(count), index=False, startrow=2, startcol=0)
                
                #: Get the xlsxwriter workbook and worksheet objects
                worksheet = writer.sheets[str_channel_name_shortened + str_i + str(count)]
                
                #: Add original, unmodified channel name to first row
                worksheet.write(0, 0, chan_name)
                
                #: Set the column width and format.
                worksheet.set_column(0, 0, 50) #: first col, last col, width of col
            
            #: Close the Pandas Excel writer and output the Excel file.
            count = count + 1

        writer.save()

    def create_final_output(self, imaris_filename, non_overall_dfs, csv_collection):
        """Store remaining non-overall data that has `TrackID` (if applicable), `ID_Object` (if applicable), and feature data in a Pandas dataframe

        Args:
            imaris_filename (:obj:`str`): Filename of Imaris file
            non_overall_dfs: Dictionary of non-overall dataframes for all channels where key is user-defined channel name and value is non-overall dataframe
            csv_collection (:obj:`str`): Output csv collection provided by user
        """
        #: Get basename from imaris filename, to prepend to channel.csv
        imaris_basename = imaris_filename.stem
        #: Remove .ims extension
        imaris_basename = imaris_basename[:-4]
        for each_key, each_value in non_overall_dfs.items():

        #: Remove all special characters from channel name (each key) and replace with underscore
            each_key_mod = re.sub('[^0-9a-zA-Z]+', '_', each_key)

            for i in range(0, len(each_value)):
                str_i = ""

                if i == 1:
                    str_i = "_copy"

                if i > 1:
                    str_i = "_copy " + str(i)

                #: If column starts with underscore, as in features added by plugins, remove underscore from the front of the file
                for col in each_value[i].columns:

                    if col[:1] == "_":
                        col_mod = col[1:]
                        each_value[i].rename(columns={col:col_mod}, inplace=True)

                #: Sort header names alphabetically
                header_names = each_value[i].columns
                header_names = header_names.sort_values()
                each_value[i] = each_value[i][header_names]

                for col in each_value[i].columns:
                    
                    #: Round each value excluding ID, TrackID and Time to 6 significant figures, as Imaris does
                    if col != "TrackID_" + each_key_mod and col != "ID_Object_" + each_key_mod and col != "ID_Object_" + each_key_mod and col!= "ID_Time" and "TrackID" not in col:
                        each_value[i][col] = each_value[i][col].apply(self.round_to_six)

                each_value[i].columns = each_value[i].columns.str.replace("___", "_")
                each_value[i].columns = each_value[i].columns.str.replace("__", "_")
                each_value[i].columns = each_value[i].columns.str.replace("ID_Time", "Time")
                each_value[i].columns = each_value[i].columns.str.replace("ID_Object", "ID")

                #: Display np.NaN values as as 'NaN' string instead of blank space, so FlowJo can view
                each_value[i].to_csv(csv_collection + "/" + imaris_basename + "_" + each_key + str_i + ".csv", index=False, na_rep='NaN')

    def create_csv_fun(self):
        """This function combines all intermediate files (``extract_ims_data.py`` and ``link_ims_ids.py`` outputs) to produce csv files that link IDs to features for each channel and an xlsx file containing overall summary statistics.

        Note:
            Inputs:
                ``link_ims_ids.py`` output, ``extract_ims_data.py`` output
                
            Outputs: 
                ``Overall.xlsx`` contains summary data for each channel. 
                
                Remaining feature data is exported within individual csv files for each channel. 
                For example:
                ``Red.csv``
                ``Green.csv``
                ``ColocSurfaces.csv``
        """

        #: Open the file for reading; h5py.File acts like a dictionary
        self.logger.debug("Opening .ims file {}...".format(str(self.ims_filename)))
        self.f = h5py.File(self.ims_filename, 'r')

        #: Determine number of groups (channel_names) in 'Scene8/Content'
        logging.debug("Counting channel names in Scene8/Content...")
        channel_names = list(self.f['Scene8']['Content'].keys())
        
        #: Read the objectdf, trackdf, and track_id_object_df csv files for a single iteration and combines them into a single df
        all_overall_dfs = {}
        non_overall_dfs = {}
        for i in range(0,len(channel_names)):
            
            #: Loop through each attribute in Scene8/Content/
            self.logger.debug("\n\nITERATION {}/{} OF FILE {}".format(i+1, len(channel_names), self.ims_filename))
            current_channel = channel_names[i]
            self.logger.debug("Reading {}...".format(current_channel))

            #: For each channel, read the attribute of that channel, specifically the key 'Name', which contains the channel name info. Then, convert from byte to string
            user_defined_channel_name = self.f['Scene8']['Content'][current_channel].attrs['Name'].tostring(order='C')
            
            #: Convert channel name from class byte to string
            user_defined_channel_name = str(user_defined_channel_name, "utf-8")
            excel_channel_name = user_defined_channel_name
            
            #: Implement regex to remove user-added special characters (_, spaces, /) from channel name.
            regex = re.compile('[^a-zA-Z0-9]+')
            user_defined_channel_name = regex.sub('_', user_defined_channel_name) #: Replaces special characters with _
            user_defined_channel_name = "_" + user_defined_channel_name
            
            #: Skip empty channels
            if user_defined_channel_name == "__":
                pass
            
            #: Read the required input files
            else:
                if (path.exists(self.dir_name + "trackdf_" + current_channel + ".csv") == True) and (path.exists(self.dir_name + "objectdf_" + current_channel + ".csv") == True):
                    
                    #: Load Track Data
                    track_df = self.get_df_from_csv(self.dir_name, current_channel, user_defined_channel_name, "trackdf_")
                    
                    #: Load Object Data
                    object_df = self.get_df_from_csv(self.dir_name, current_channel, user_defined_channel_name, "objectdf_")
                    
                    #: Load Track ID: Object ID data
                    track_id_object_df = self.get_df_from_csv(self.dir_name, current_channel, user_defined_channel_name, "")
                
                    has_track = True
                    has_object = True
                    has_track_id_object = True

                    #: Determine if track_df or object_df is empty. If so, set has_object or has_track to False.
                    if track_df.empty == True:
                        has_track = False
                    
                    if object_df.empty == True:
                        has_object = False

                    if track_id_object_df.empty == True:
                        track_id_object_df = pd.DataFrame({'TrackID' + user_defined_channel_name:np.NaN, 'ID_Object' + user_defined_channel_name:np.NaN}, index=[0])
                        has_track_id_object == True

                    #: Isolate "Overall" data
                    if (has_track_id_object == True and has_object == True) or (has_track_id_object == True and has_object == False):
                        
                        #: Add 1 to all time channels so first time point is 1 sec, not 0 sec
                        object_df['ID_Time'] = object_df['ID_Time'] + 1
                        
                        #: Where Object ID < 0, save as "Overall"
                        overall_df = object_df.loc[object_df['ID_Object' + user_defined_channel_name] < 0].copy()
                        
                        #: Where Object ID > -1, save as "Object"
                        object_df = object_df.loc[object_df['ID_Object' + user_defined_channel_name] >= 0]
                        
                        #: Check if dataframe became empty after moving values from Object DF to Overall file
                        if object_df.empty == True:
                            has_object = False
                        
                        overall_df = self.get_overall(overall_df, user_defined_channel_name)

                        #: Create dictionary of overall dataframes where key is user-defined channel name and value is overall dataframe
                        if excel_channel_name in all_overall_dfs:
                            all_overall_dfs[excel_channel_name].append(overall_df)
                        
                        else:
                            all_overall_dfs[excel_channel_name] = []
                            all_overall_dfs[excel_channel_name].append(overall_df)
                    
                    #: Merge dictionary of IDs and tracks/objects together
                    if has_object == True:
                        
                        #: Wherever Object ID >= 1, save as object data
                        object_df = object_df[object_df['ID_Object' + user_defined_channel_name] >= 0]
                        object_df.dropna(axis=1, how='all', inplace=True)
                    
                    #: Combine ID dictionary, Track, and/or Object data:
                    if has_object == True and has_track == False:
                        track_id_object_df = pd.merge(track_id_object_df, object_df, how='outer', on='ID_Object' + user_defined_channel_name)
                        track_id_object_df.dropna(axis=0, how='all', inplace=True)
                        track_id_object_df.dropna(axis=1, how='all', inplace=True)
                        
                        #: Resolve issue file overwrite for files that share channel name
                        if excel_channel_name in non_overall_dfs:
                            non_overall_dfs[excel_channel_name].append(track_id_object_df)
                        
                        else:
                            non_overall_dfs[excel_channel_name] = []
                            non_overall_dfs[excel_channel_name].append(track_id_object_df)

                    elif has_object == False and has_track == True:
                        track_id_object_df = pd.merge(track_id_object_df, track_df, how='outer', on='TrackID' + user_defined_channel_name)
                        
                        if excel_channel_name in non_overall_dfs:
                            non_overall_dfs[excel_channel_name].append(track_id_object_df)
                        
                        else:
                            non_overall_dfs[excel_channel_name] = []
                            non_overall_dfs[excel_channel_name].append(track_id_object_df)

                    #: Resolve issue file overwrite for files that share channel name
                    elif has_object == True and has_track == True:
                        
                        #: First merge ID dictionary to objects
                        merged_object = pd.merge(object_df, track_id_object_df, how='outer', on='ID_Object' + user_defined_channel_name)
                        
                        #: Second merge above df to tracks
                        features_merged = pd.merge(merged_object, track_df, how='outer', on='TrackID' + user_defined_channel_name)
                        
                        if excel_channel_name in non_overall_dfs:
                            non_overall_dfs[excel_channel_name].append(features_merged)
                        
                        else:
                            non_overall_dfs[excel_channel_name] = []
                            non_overall_dfs[excel_channel_name].append(features_merged)
        
        if all_overall_dfs:
            
            #: Export overall data as xlsx file
            self.create_overall_xlsx(self.ims_filename, self.meta_dir_name, all_overall_dfs)

        #: Create final output
        self.logger.info("Creating final output (stage 3/3)...")
        self.create_final_output(self.ims_filename, non_overall_dfs, self.dir_name)

        self.logger.info("{} complete!".format(str(self.ims_filename)))