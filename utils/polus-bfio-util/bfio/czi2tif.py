"""
The primary function in this file is write_ome_tiffs. It will extract
individual fields of view from a czi file into tiff files.

Example:
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)
write_ome_tiffs("Path/To/File.czi","Path/To/Output/Directory")
javabridge.kill_vm()

Required packages:
javabridge (also requires jdk >= 8)
python-bioformats
numpy
czifile

Note: Prior to conversion, the javabridge session must be started.
"""

import czifile
from bfio.bfio import BioWriter
import numpy as np
import re
from pathlib import Path
import os
            
def _get_image_dim(s,dim):
    ind = s.axes.find(dim)
    if ind<0:
        return 1
    else:
        return s.shape[ind]
    
def _get_physical_dimensions(metadata):
    # The czifile package seems to import the physical sizes incorrectly, so
    # this function searches the metadata to specially handle the physical size
    # information.
    #
    # NOTE: Units are always in um
    if type(metadata) is not str:
        TypeError("Metadata must be a string.")
    phys_dim = {'X': None,
                'Y': None,
                'Z': None}
    regex = r'<Distance Id="([XYZ])">[\s]*<Value>([\b-?[1-9](?:\.\d+)?[Ee][-+]?\d+\b)'
    matches = re.findall(regex,metadata)
    for m in matches:
        if m[0] in phys_dim.keys():
            phys_dim[m[0]] = float(m[1])*1000000
    return phys_dim
    
def _get_channel_names(metadata):
    if type(metadata) is not dict:
        TypeError("Metadata must be a dictionary.")
        
    return [c['Name'] for c in metadata['DisplaySetting']['Channels']['Channel']]
    
def _get_image_name(base_name,row,col,Z=None,C=None,T=None,padding=3):
    name = base_name
    name += "_y" + str(row).zfill(padding)
    name += "_x" + str(col).zfill(padding)
    if Z is not None:
        name += "_z" + str(Z).zfill(padding)
    if C is not None:
        name += "_c" + str(C).zfill(padding)
    if T is not None:
        name += "_t" + str(T).zfill(padding)
    name += ".ome.tif"
    return name
    
def write_ome_tiffs(file_path,out_path):
    if Path(file_path).suffix != '.czi':
        TypeError("Path must be to a czi file.")
        
    base_name = Path(Path(file_path).name).stem
    
    czi = czifile.CziFile(file_path,detectmosaic=False)
    subblocks = [s for s in czi.filtered_subblock_directory if s.mosaic_index is not None]
    
    metadata_str = czi.metadata(True)
    metadata = czi.metadata(False)['ImageDocument']['Metadata']
    
    chan_name = _get_channel_names(metadata)
    pix_size = _get_physical_dimensions(metadata_str)
    
    ind = {'X': [],
           'Y': [],
           'Z': [],
           'C': [],
           'T': [],
           'Row': [],
           'Col': []}
    
    for s in subblocks:
        scene = [dim.start for dim in s.dimension_entries if dim.dimension=='S']
        if scene is not None and scene[0] != 0:
            continue
        
        for dim in s.dimension_entries:
            if dim.dimension=='X':
                ind['X'].append(dim.start)
            elif dim.dimension=='Y':
                ind['Y'].append(dim.start)
            elif dim.dimension=='Z':
                ind['Z'].append(dim.start)
            elif dim.dimension=='C':
                ind['C'].append(dim.start)
            elif dim.dimension=='T':
                ind['T'].append(dim.start)
                
    row_conv = {y:row for (y,row) in zip(np.unique(np.sort(ind['Y'])),range(0,len(np.unique(ind['Y']))))}
    col_conv = {x:col for (x,col) in zip(np.unique(np.sort(ind['X'])),range(0,len(np.unique(ind['X']))))}
    
    ind['Row'] = [row_conv[y] for y in ind['Y']]
    ind['Col'] = [col_conv[x] for x in ind['X']]
    
    for s,i in zip(subblocks,range(0,len(subblocks))):
        dims = [_get_image_dim(s,'Y'),
                _get_image_dim(s,'X'),
                _get_image_dim(s,'Z'),
                _get_image_dim(s,'C'),
                _get_image_dim(s,'T')]
        data = s.data_segment().data().reshape(dims)
        
        Z = None if len(ind['Z'])==0 else ind['Z'][i]
        C = None if len(ind['C'])==0 else ind['C'][i]
        T = None if len(ind['T'])==0 else ind['T'][i]
        
        out_file_path = os.path.join(out_path,_get_image_name(base_name,
                                                              row=ind['Row'][i],
                                                              col=ind['Col'][i],
                                                              Z=Z,
                                                              C=C,
                                                              T=T))
        
        bw = BioWriter(out_file_path,data)
        bw.channel_names([chan_name[C]])
        bw.physical_size_x(pix_size['X'],'µm')
        bw.physical_size_y(pix_size['Y'],'µm')
        if pix_size['Z'] is not None:
            bw.physical_size_y(pix_size['Z'],'µm')
        
        bw.write_image(data)
        bw.close_image()