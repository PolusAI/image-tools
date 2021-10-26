# This file originally appeared in bfio 1.X, but was removed in 2.X
# It has been updated to work with bfio 2.X
# https://github.com/PolusAI/polus-plugins/blob/40495bfeaff31ec636bbc73e7335179dbb3a3eb7/utils/polus-bfio-util/bfio/czi2tif.py

import czifile
from bfio.bfio import BioWriter, BioReader, OmeXml
import numpy as np
import re, os, logging, argparse
from pathlib import Path
from preadator import ProcessManager

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# Quiet the bfio logger since the metadata extracted from the czi is
# incompatible with the ome tiled tiff format
logging.getLogger('bfio').setLevel(logging.ERROR)
            
def _get_image_dim(s,dim):
    ind = s.axes.find(dim)
    if ind<0:
        return 1
    else:
        return s.shape[ind]
    
def _get_image_name(base_name,row,col,Z=None,C=None,T=None,padding=3):
    """ This function generates an image name from image coordinates """
    
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

def write_thread(out_file_path: Path,
                 data: np.ndarray,
                 metadata: OmeXml,
                 chan_name: str):
    """ Thread for saving images

    This function is intended to be run inside a threadpool to save an image.

    Args:
        out_file_path (Path): Path to an output file
        data (np.ndarray): FOV to save
        metadata (OmeXml): Metadata for the image
        chan_name (str): Name of the channel
    """
        
    ProcessManager.log(f'Writing: {out_file_path.name}')
    
    with BioWriter(out_file_path,metadata=metadata) as bw:
        
        bw.X = data.shape[1]
        bw.Y = data.shape[0]
        bw.Z = 1
        bw.C = 1
        bw.cnames = [chan_name]
        
        bw[:] = data
    
def extract_fovs(file_path: Path,
                 out_path: Path):
    """ Extract individual FOVs from a czi file

    When CZI files are loaded by BioFormats, it will generally try to mosaic
    images together by stage position if the image was captured with the
    intention of mosaicing images together. At the time this function was
    written, there was no clear way of extracting individual FOVs so this
    algorithm was created.
    
    Every field of view in each z-slice, channel, and timepoint contained in a
    CZI file is saved as an individual image.

    Args:
        file_path (Path): Path to CZI file
        out_path (Path): Path to output directory
    """
    
    with ProcessManager.process(file_path.name):
        
        logger.info('Starting extraction from ' + str(file_path) + '...')

        if Path(file_path).suffix != '.czi':
            TypeError("Path must be to a czi file.")
            
        base_name = Path(file_path.name).stem
        
        # Load files without mosaicing
        czi = czifile.CziFile(file_path,detectmosaic=False)
        subblocks = [s for s in czi.filtered_subblock_directory if s.mosaic_index is not None]
        
        ind = {'X': [],
               'Y': [],
               'Z': [],
               'C': [],
               'T': [],
               'Row': [],
               'Col': []}
        
        # Get the indices of each FOV
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
        
        with BioReader(file_path) as br:
            
            metadata = br.metadata
            chan_names = br.cnames
        
        for s,i in zip(subblocks,range(0,len(subblocks))):
            
            Z = None if len(ind['Z'])==0 else ind['Z'][i]
            C = None if len(ind['C'])==0 else ind['C'][i]
            T = None if len(ind['T'])==0 else ind['T'][i]
        
            out_file_path = out_path.joinpath(_get_image_name(base_name,
                                                              row=ind['Row'][i],
                                                              col=ind['Col'][i],
                                                              Z=Z,
                                                              C=C,
                                                              T=T))
            
            dims = [_get_image_dim(s,'Y'),
                    _get_image_dim(s,'X'),
                    _get_image_dim(s,'Z'),
                    _get_image_dim(s,'C'),
                    _get_image_dim(s,'T')]
            
            data = s.data_segment().data().reshape(dims)
            
            write_thread(out_file_path,
                         data,
                         metadata,
                         chan_names[C])

def main(input_dir: Path,
         output_dir: Path
         ) -> None:

    logger.info('Extracting tiffs and saving as ome.tif...')
    files = [f for f in Path(input_dir).iterdir() if f.suffix=='.czi']
    if not files:
        logger.error('No CZI files found.')
        raise ValueError('No CZI files found.')
        
    ProcessManager.init_processes()
    
    for file in files:
        ProcessManager.submit_process(extract_fovs,file,output_dir)
        
    ProcessManager.join_processes()

if __name__ == "__main__":
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)


    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    
    main(input_dir,
         output_dir)