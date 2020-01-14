import csv
import h5py
import logging
import numpy as np
import os
import pandas as pd

def link_trackid_objectid(ims_filename, dir_name):
    """This function extracts and links track and object IDs from the imaris file using metadata within the ``Track0`` and ``TrackObject0`` attributes

    Args:
        ims_filename (:obj:`str`): Name of the selected Imaris file
        dir_name (:obj:`str`): Output csv collection provided by user

    Returns:
        Temporary csv file for each channel, named after dir_name and channel_name. The file displays a table of linked `TrackID` data with corresponding `ID_Object` data for each channel.
    """

    #: Open file
    f = h5py.File(ims_filename, 'r')

    '''There are 6 outer groups in the file: 
    ['DataSet', 'DataSetInfo', 'DataSetTimes', 'Scene', 'Scene8', 'Thumbnail']. 
    Examine the 'Scene8' data set as a Dataset object.'''

    channel_names = list(f['Scene8']['Content'].keys())
    id_dict = {}

    for ims_channel in channel_names:
        #: Create Track0 df
        contains_trackobject0 = (f['Scene8']['Content'][ims_channel]).__contains__('TrackObject0')
        contains_track0 = (f['Scene8']['Content'][ims_channel]).__contains__('Track0')

        if contains_trackobject0 == True and contains_track0 == True:
            #: Get Track0 attribute from .ims file
            track0_df = pd.DataFrame(f['Scene8']['Content'][ims_channel]['Track0'][()])
            
            #: Drop unneeded columns
            track0_df = track0_df.drop(columns=['IndexTrackEdgeBegin', 'IndexTrackEdgeEnd']) 

            #: Get TrackObject0 attribute from .ims file
            trackobject0 = pd.DataFrame(f['Scene8']['Content'][ims_channel]['TrackObject0'][()])

            #: Add ID column data from Track0 using indexes of TrackObject0

            '''Link TrackID to ID_Object in such a way 
            that if (IndexBegin:IndexEnd) is (0:3), then 
            TrackID gets inserted into trackobject0_df at 
            indexes 0 through 2 (IndexEnd - 1). The 
            starting index and ending index are determined 
            using the last two columns of track0_df.'''

            track_object_ids = (pd.merge_asof(trackobject0.reset_index(), track0_df, left_on='index', right_on='IndexTrackObjectBegin').reindex(['ID_Object', 'ID'], axis=1))
            
            #: Column containing TrackIDs is labeled "ID." Change to TrackID
            track_object_ids.rename(columns = {'ID':'TrackID'}, inplace = True)

            #: Store channel name as key and df with linked Object IDs and TrackIDs as value id_dict
            id_dict[ims_channel] = track_object_ids
        
        else:
            #: Create output file for each channel, even if there is no data for TrackID and ObjectIDs, for consistency
            empty_track_object_ids_df = pd.DataFrame(columns = ['ID_Object', 'TrackID'])
            id_dict[ims_channel] = empty_track_object_ids_df

    #: Set up the logger
    logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%b-%d-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.debug("Creating intermediate csv for link_ims_ids module...")
    logger.info("Linking IDs (stage 2/3)...")
    
    #: Create ID_Object, Track ID .csv file using data above
    for channel_name in id_dict:
        id_dict[channel_name].to_csv(dir_name + channel_name + ".csv", index=False)
    
    logger.debug("Done with level 2")