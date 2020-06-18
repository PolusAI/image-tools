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
    """Class combines csv linking IDs to csv containing feature info.

    Merge linked track and object IDs to corresponding feature data to 
    produce a csv output file that can be visualized in FlowJo.
    """
    def __init__(self, ims_filename, dir_name, meta_dir_name, logger=None):
        """Open .ims file for reading; h5py.File acts like a dictionary.

        Args:
            ims_filename (:obj:`str`): Name of the selected Imaris file
            dir_name (:obj:`str`): Output csv collection
            meta_dir_name (:obj:`str`): Output metadata directory
        """
        
        #: Set up the logger
        logging.basicConfig(
            format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', 
            datefmt='%b-%d-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.ims_filename = ims_filename
        self.dir_name = dir_name
        self.meta_dir_name = meta_dir_name
        self.f = h5py.File(self.ims_filename, 'r')

    def round_to_six(self, num):
        """Round values to six significant figures

        Args:
            num (:obj:`int`): Num to be rounded to 6 significant figures

        Returns:
            A number rounded to six significant figures
        """

        if num != 0:
            if np.isnan(num) != True:
                num = np.round(num, -int(np.floor(np.log10(abs(num)))) + 5)
        elif num == 0:
            pass
        return num

    def get_df_from_csv(self, dirname, chan, chan_name, csv_substring):
        
        """Read intermediate csv files containing feature data 
        (``extract_ims_data.py`` output) or ID data (``link_ims_ids.py`` 
        output) for each channel, and store in dataframes. ``chan`` 
        represents attribute names within the ``Scene8/Content`` keys
        of the .ims files with names ``MegaSurfaces0`` or ``Points0``.
        Each attribute contains data belonging to particular channels.
        The argument chan_name differs from chan because while chan 
        might have a more general name such as ``Points0``, chan_name is 
        extracted from the Imaris file's metadata, converted from byte to 
        string, and stored as **chan_name**. csv_substring is the 
        substring of the csv file that gets read in; can be `trackdf_`/
        `objectdf_` for outputs of the first module 
        (``extract_ims_data.py``), or empty string for outputs of the 
        second module, ``link_ims_ids.py`` (links `ID_Object` and 
        `TrackID`).

        Args:
            dirname (:obj:`str`): Output csv collection provided by user
            chan (:obj:`str`): attribute name in ``Scene8/Content``
            chan_name (:obj:`str`): Channel name entered in Imaris
            csv_substring (:obj:`str`): Temp .csv prefix (ex: trackdf_)

        Returns:
            Pandas dataframe created from intermediate csv files.
        """

        keepsame = {'ID_Time'}
        #: Read csv
        temp_string = csv_substring + chan + ".csv"
        temp_path = dirname/temp_string
        df = pd.read_csv(temp_path)

        #: Remove "Unnamed" columns:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        #: suffix chan name to all headers except 'ID_Object, 'ID_Time'
        if len(df.columns) > 0:

            #: Suffix col names except some columns (keep same=ID_Time)
            df.columns = ['{}{}'.format(
                c, '' if c in keepsame else chan_name) for c in df.columns]
            df.columns = df.columns.str.replace('__', '_')

        #: Remove intermediate csv file
        self.logger.debug("Remove line file_to_remove.unlink() to debug.")
        self.logger.debug(
            "CSV files can indicate where in the process an issue begins.")
        file_to_remove = temp_path
        file_to_remove.unlink()

        return df
    
    def get_overall(self, overall_df, chan_name):
        """Extract overall data from object data

        This reads the output from the ``get_df_from_csv()`` function 
        and extracts overall data from the df containing non-track data.
        Note that in the second module, ``extract_ims_data.py``, 
        data tagged `Overall` was assigned an `ID_Object` of -1

        Args:
            overall_df: Df obtained `ID_Object` of object df < 0. 
            chan_name (:obj:`str`): Channel name entered in Imaris

        Returns:
            DF containing overall data; formatted like Imaris version
        """

        overall_df.dropna(axis=1, how='all', inplace=True)

        #: All ID_Objects == -1 belong to Overall. Replace with np.NaN
        overall_df['ID_Object' + chan_name] = \
            overall_df['ID_Object'+chan_name].replace(
                -1.0, np.NaN, inplace=True)
        
        #: Replace time = -1.0 with np.NaN
        overall_df['ID_Time'].replace(-1.0, np.NaN, inplace=True)
        overall_df.reset_index()
        
        #: Rearrange df to match exact format exported by Imaris file
        overall_df = pd.melt(
            overall_df, id_vars=['ID_Time', 'ID_Object' + chan_name], 
            var_name='Variable', value_name='Value')

        overall_df = overall_df[
            ['Variable','Value','ID_Time','ID_Object'+chan_name]]

        overall_df.rename(
            {'ID_Time': 'Time', 'ID_Object' + chan_name: 'ID'}, 
            axis='columns', inplace=True)
        overall_df.dropna(subset=['Value'], inplace=True)
        overall_df['Variable'] = overall_df['Variable'].str.replace('_', ' ')
        overall_df=overall_df.dropna(axis=1,how='all')
        return overall_df

    def create_overall_xlsx(self,imaris_filename,meta_dirname,all_overall_dict):
        """Create overall xlsx. Each sheet represents one channel.

        This function merges all Overall DFs together and write each 
        channel to an xlsx notebook that uses sheets to represent 
        individual channels


        Args:
            imaris_filename (:obj:`str`): Filename of Imaris file
            meta_dirname (:obj:`str`): Output metadata directory
            all_overall_dict: Dict key=Imaris channel, value=overall df
        """


        #: Get basename from imaris filename, to prepend to Overall.xlsx
        imaris_basename = imaris_filename.stem

        #: Remove .ims extension
        imaris_basename = imaris_basename[:-4]

        #: Create a Pandas Excel writer using XlsxWriter as the engine
        temp_string = imaris_basename + "_" + 'Overall.xlsx'
        temp_path = meta_dirname/temp_string
        writer = pd.ExcelWriter(temp_path, engine='xlsxwriter')
        count = 1
        for chan_name, overall_df_list in all_overall_dict.items():
            for i in range(0, len(overall_df_list)):
                str_i = "_"
                if i >= 1:
                    str_i = "_" + str(i) + "_"
                str_channel_name = re.sub('[^A-Za-z0-9]+', '_', chan_name)

                #: Convert the dataframe to an XlsxWriter Excel object
                str_channel_name_shortened = ""
                if len(str_channel_name) > 25:
                    str_channel_name_shortened = str_channel_name[:25]
                
                else:
                    str_channel_name_shortened = str_channel_name
                
                #: Round Overall "Values" column to 6 significant digits
                self.logger.debug("Converting data to 6 significant figures...")
                overall_df_list[i]['Value'] = overall_df_list[i]\
                    ['Value'].apply(self.round_to_six)
                
                overall_df_list[i].to_excel(
                    writer, 
                    sheet_name=str_channel_name_shortened + str_i + str(count), 
                    index=False, startrow=2, startcol=0)
                
                #: Get the xlsxwriter workbook and worksheet objects
                worksheet = writer.sheets[
                    str_channel_name_shortened + str_i + str(count)]
                
                #: Add original, unmodified channel name to first row
                worksheet.write(0, 0, chan_name)
                
                #: Set the column width and format.
                worksheet.set_column(0, 0, 50) #: 1st, last col, width
            
            #: Close the Pandas Excel writer and output the Excel file.
            count = count + 1

        writer.save()

    def create_final_output(self, imaris_filename, non_overall_dfs, dirname):
        """Stores non-overall data in dataframes

        Store remaining non-overall data with `TrackID` (if applicable), 
        `ID_Object` (if applicable), and feature data in a Pandas 
        dataframe.

        Args:
            imaris_filename (:obj:`str`): Filename of Imaris file
            non_overall_dfs: dict key=channel name, value=non-overall df
            dirname (:obj:`str`): Output csv collection provided by user
        """
        #: Get basename from imaris filename, to prepend to channel.csv
        imaris_basename = imaris_filename.stem
        #: Remove .ims extension
        imaris_basename = imaris_basename[:-4]
        for chan_name, non_ov in non_overall_dfs.items():

        #: Replace special characters from channel name (key) with _
            chan_mod = re.sub('[^0-9a-zA-Z]+', '_', chan_name)

            for i in range(0, len(non_ov)):
                str_i = ""

                if i == 1:
                    str_i = "_copy"

                if i > 1:
                    str_i = "_copy " + str(i)

                #: Remove _ from the front of file (due to some plugins)
                for col in non_ov[i].columns:

                    if col[:1] == "_":
                        col_mod = col[1:]
                        non_ov[i].rename(columns={col:col_mod}, inplace=True)

                #: Sort header names alphabetically
                header_names = non_ov[i].columns
                header_names = header_names.sort_values()
                non_ov[i] = non_ov[i][header_names]

                for c in non_ov[i].columns:
                    
                    #: Round all but ID, TrackID, Time to 6 sigfigs
                    if c != "TrackID_"+chan_mod and c != "ID_Object_"+chan_mod:
                        if c!="ID_Time" and "TrackID" not in c:
                            non_ov[i][c]=non_ov[i][c].apply(self.round_to_six)

                non_ov[i].columns = non_ov[i].columns.str.replace("___", "_")
                non_ov[i].columns = non_ov[i].columns.str.replace("__", "_")
                non_ov[i].columns = non_ov[i].columns.str.replace(
                    "ID_Time", "Time")
                non_ov[i].columns = non_ov[i].columns.str.replace(
                    "ID_Object", "ID")

                #: Display np.NaN values as as 'NaN' so FlowJo can view
                temp_string = imaris_basename + "_" + chan_name + str_i + ".csv"
                temp_path = dirname/temp_string
                non_ov[i].to_csv(temp_path, index=False, na_rep='NaN')

    def create_csv_fun(self):
        """Main function; combines intermediate files to produce output.
        
        This function combines all intermediate files 
        (``extract_ims_data.py`` and ``link_ims_ids.py`` outputs) 
        to produce csv files that link IDs to features for each channel 
        and an xlsx file containing overall summary statistics. 
        It takes in as inputs the csv files created from 
        ``link_ims_ids.py`` and  ``extract_ims_data.py``. It outputs an 
        ``Overall.xlsx`` file containing summary data for each channel. 
        The remaining feature data is exported within individual csv 
        files for each channel. For example: ``Red.csv``, ``Green.csv``, 
        and ``ColocSurfaces.csv``
        """

        #: Open the file for reading; h5py.File acts like a dictionary
        self.logger.debug(
            "Opening .ims file {}...".format(str(self.ims_filename)))
        self.f = h5py.File(self.ims_filename, 'r')

        #: Determine # of groups (channel_names) in 'Scene8/Content'
        logging.debug("Counting channel names in Scene8/Content...")
        channel_names = list(self.f['Scene8']['Content'].keys())

        # Ignore irrelevant channel types
        channel_names = [
            chan for chan in channel_names if chan.startswith(
                "Points") or chan.startswith("MegaSurfaces")]
        
        #: Combine objectdf, trackdf, track_id_object_df csv into 1 df
        all_overall_dfs = {}
        non_overall_dfs = {}
        for i in range(0,len(channel_names)):
            
            #: Loop through each attribute in Scene8/Content/
            self.logger.debug(
                "\n\nITERATION {}/{} OF FILE {}".format(
                    i+1, len(channel_names), self.ims_filename))
            current_channel = channel_names[i]
            self.logger.debug("Reading {}...".format(current_channel))

            #: Read 'Name' attribute of each channel to get channel name
            chan_name=self.f['Scene8']['Content'][current_channel].attrs['Name']
            chan_name = chan_name.tostring(order='C')
            
            #: Convert channel name from class byte to string
            chan_name = str(chan_name, "utf-8")
            excel_channel = chan_name
            
            #: Remove special characters from channel name using regex 
            regex = re.compile('[^a-zA-Z0-9]+')
            #: Replaces special characters with _
            chan_name = regex.sub('_', chan_name)
            chan_name = "_" + chan_name
            
            #: Skip empty channels
            if chan_name == "__":
                pass
            
            #: Read the required input files
            else:
                temp_string1 = "trackdf_" + current_channel + ".csv"
                path1 = self.dir_name / temp_string1
                temp_string2 = "objectdf_" + current_channel + ".csv"
                path2 = self.dir_name / temp_string2
                if path.exists(path1)==True and path.exists(path2)==True:
                    
                    #: Load Track Data
                    track_df = self.get_df_from_csv(
                        self.dir_name, current_channel, chan_name, "trackdf_")
                    
                    #: Load Object Data
                    object_df = self.get_df_from_csv(
                        self.dir_name, current_channel, chan_name, "objectdf_")
                    
                    #: Load Track ID: Object ID data
                    track_id_object_df = self.get_df_from_csv(
                        self.dir_name, current_channel, chan_name, "")
                
                    has_track = True
                    has_object = True
                    has_track_id_object = True

                    #: Determine if track_df or object_df is empty. 
                    if track_df.empty == True:
                        #: If so, set has_object or has_track to False.
                        has_track = False
                    
                    if object_df.empty == True:
                        has_object = False

                    if track_id_object_df.empty == True:
                        track_id_object_df = pd.DataFrame(
                            {'TrackID' + chan_name:np.NaN, 'ID_Object' + \
                                chan_name:np.NaN}, index=[0])
                        has_track_id_object == True

                    #: Isolate "Overall" data
                    if (has_track_id_object == True and has_object == True) or \
                        (has_track_id_object == True and has_object == False):
                        
                        #: Add 1 to all time chans (sets t=0 to t=1)
                        object_df['ID_Time'] = object_df['ID_Time'] + 1
                        
                        #: Where Object ID < 0, save as "Overall"
                        overall_df = object_df.loc[object_df[
                            'ID_Object' + chan_name] < 0].copy()
                        
                        #: Where Object ID > -1, save as "Object"
                        object_df = object_df.loc[
                            object_df['ID_Object' + chan_name] >= 0]
                        
                        #: Flag empty dfs after moving object to overall
                        if object_df.empty == True:
                            has_object = False
                        
                        overall_df = self.get_overall(overall_df, chan_name)

                        #: Make dict key=.ims channel, val=overall df
                        if excel_channel in all_overall_dfs:
                            all_overall_dfs[excel_channel].append(overall_df)
                        
                        else:
                            all_overall_dfs[excel_channel] = []
                            all_overall_dfs[excel_channel].append(overall_df)
                    
                    #: Merge dict of IDs and tracks/objects together
                    if has_object == True:
                        
                        #: Wherever Object ID >= 1, save as object data
                        object_df=object_df[object_df['ID_Object'+chan_name]>=0]
                        object_df.dropna(axis=1, how='all', inplace=True)
                    
                    #: Combine ID dictionary, Track, and/or Object data
                    if has_object == True and has_track == False:
                        track_id_object_df = pd.merge(
                            track_id_object_df, object_df, 
                            how='outer', on='ID_Object' + chan_name)

                        track_id_object_df.dropna(
                            axis=0, how='all', inplace=True)

                        track_id_object_df.dropna(
                            axis=1, how='all', inplace=True)
                        
                        #: Resolve overwrite for files sharing chan name
                        if excel_channel in non_overall_dfs:
                            non_overall_dfs[excel_channel].append(
                                track_id_object_df)
                        
                        else:
                            non_overall_dfs[excel_channel] = []
                            non_overall_dfs[excel_channel].append(
                                track_id_object_df)

                    elif has_object == False and has_track == True:
                        track_id_object_df = pd.merge(
                            track_id_object_df, track_df, how='outer', 
                            on='TrackID' + chan_name)
                        
                        if excel_channel in non_overall_dfs:
                            non_overall_dfs[excel_channel].append(
                                track_id_object_df)
                        
                        else:
                            non_overall_dfs[excel_channel] = []
                            non_overall_dfs[excel_channel].append(
                                track_id_object_df)

                    #: Fix issue overwrite for files sharing chan name
                    elif has_object == True and has_track == True:
                        
                        #: First merge ID dictionary to objects
                        merged_object = pd.merge(
                            object_df, track_id_object_df, how='outer', 
                            on='ID_Object' + chan_name)
                        
                        #: Second merge above df to tracks
                        features_merged = pd.merge(
                            merged_object, track_df, how='outer', 
                            on='TrackID' + chan_name)
                        
                        if excel_channel in non_overall_dfs:
                            non_overall_dfs[excel_channel].append(
                                features_merged)
                        
                        else:
                            non_overall_dfs[excel_channel] = []
                            non_overall_dfs[excel_channel].append(
                                features_merged)
        
        if all_overall_dfs:
            
            #: Export overall data as xlsx file
            self.create_overall_xlsx(
                self.ims_filename, self.meta_dir_name, all_overall_dfs)

        #: Create final output
        self.logger.info("Creating final output (stage 3/3)...")
        self.create_final_output(
            self.ims_filename, non_overall_dfs, self.dir_name)

        self.logger.info("{} complete!".format(str(self.ims_filename)))