from bfio.bfio import BioReader
import numpy as np
import json
import copy
import gzip
import os
from pathlib import Path
import sys

# Data types used by Neuroglancer
NEUROGLANCER_DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")

# Conversion factors to nm
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 1024

# Modified and condensed from FileAccessor class in neuroglancer-scripts
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.py
class SlideWriter():
    """Access a Neuroglancer pre-computed pyramid on the local file system.
    :param str base_dir: path to the directory containing the pyramid
    """

    can_write = True

    def __init__(self, base_dir):
        self.base_path = Path(base_dir)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"
        self.gzip = True

    def store_chunk(self, buf, key, chunk_coords):
        chunk_path = self._chunk_path(key, chunk_coords)
        mode = "wb"
        try:
            os.makedirs(str(chunk_path.parent), exist_ok=True)
            with open(
                    str(chunk_path.with_name(chunk_path.name)),
                    mode) as f:
                f.write(buf)
        except OSError as exc:
            raise FileNotFoundError(
                "Error storing chunk {0} in {1}: {2}" .format(
                    self._chunk_path(key, chunk_coords),
                    self.base_path, exc))

    def _chunk_path(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        xmin, xmax, ymin, ymax, zmin, zmax = chunk_coords
        chunk_filename = pattern.format(
            xmin, xmax, ymin, ymax, zmin, zmax, key=key)
        return self.base_path / chunk_filename

# Modified and condensed from multiple functions and classes
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/chunk_encoding.py
class ChunkEncoder:
    """Encode/decode chunks from NumPy arrays to byte buffers.
    :param str data_type: data type supported by Neuroglancer
    :param int num_channels: number of image channels
    """

    lossy = False
    """True if this encoder is lossy."""

    mime_type = "application/octet-stream"
    """MIME type of the encoded chunk."""

    def __init__(self, info):
        
        try:
            data_type = info["data_type"]
            num_channels = info["num_channels"]
            encoding = info["scales"][0]['encoding']
        except KeyError as exc:
            raise KeyError("The info dict is missing an essential key {0}"
                                .format(exc)) from exc
        if not isinstance(num_channels, int) or not num_channels > 0:
            raise KeyError("Invalid value {0} for num_channels (must be "
                                "a positive integer)".format(num_channels))
        if data_type not in NEUROGLANCER_DATA_TYPES:
            raise KeyError("Invalid data_type {0} (should be one of {1})"
                                .format(data_type, NEUROGLANCER_DATA_TYPES))
        
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")

    def encode(self, chunk):
        """Encode a chunk from a NumPy array into bytes.
        :param numpy.ndarray chunk: array with four dimensions (C, Z, Y, X)
        :returns: encoded chunk
        :rtype: bytes
        """
        chunk = np.asarray(chunk).astype(self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = chunk.tobytes()
        return buf

def bfio_metadata_to_slide_info(bfio_reader,outPath):
    
    # Create an output path object for the info file
    op = Path(outPath).joinpath("info")
    
    # Get metadata info from the bfio reader
    sizes = [bfio_reader.num_x(),bfio_reader.num_y(),bfio_reader.num_z()]
    phys_x = bfio_reader.physical_size_x()
    phys_y = bfio_reader.physical_size_y()
    resolution = [phys_x[0] * UNITS[phys_x[1]]]
    resolution.append(phys_y[0] * UNITS[phys_y[1]])
    resolution.append((phys_y[0] * UNITS[phys_y[1]] + phys_x[0] * UNITS[phys_x[1]])/2) # Just used as a placeholder
    dtype = bfio_reader.read_metadata().image().Pixels.get_PixelType()
    
    # create a scales template, use the full resolution
    scales = {
        "chunk_sizes":[[CHUNK_SIZE,CHUNK_SIZE,1]],
        "encoding":"raw",
        "key":"0",
        "resolution":resolution,
        "size":sizes,
        "voxel_offset":[0,0,0]
    }
    
    # initialize the json dictionary
    info = {
        "data_type": dtype,
        "num_channels":1,
        "scales": [scales],       # Will build scales belows
        "type": "image"
    }
    
    num_scales = int(np.floor(np.min(np.log2(info['scales'][-1]['size'][0:2]))-np.log2(CHUNK_SIZE)))
    num_scales = max(0,num_scales)
    
    # for i in range(1,num_scales):
    #     previous_scale = info['scales'][-1]
    #     current_scale = copy.deepcopy(previous_scale)
    #     current_scale['key'] = str(i)
    #     current_scale['size'] = [np.round(previous_scale['size'][0]/2),np.round(previous_scale['size'][1]/2),1]
    #     current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
    #     info['scales'].append(current_scale)
        
    with open(op,'w') as writer:
        writer.write(json.dumps(info))
    
    return info

def main():
    import logging
    import argparse
    import javabridge as jutil
    import bioformats
    from pathlib import Path
    import pprint
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)


    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    logger.info('input_dir = {}'.format(input_dir))
    logger.info('output_dir = {}'.format(output_dir))
    
    logger.info('Initializing the javabridge...')
    jutil.start_vm(class_path=bioformats.JARS)
    
    # Path to bioformats supported image
    logger.info('Getting the images...')
    image_path = Path(input_dir)
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    
    # Create the BioReader object
    logger.info('Getting the BioReader...')
    bf = BioReader(str(images[0].absolute()))
    logger.info('Done!')
    
    # Create the output path and info file
    out_path = output_dir
    file_info = bfio_metadata_to_slide_info(bf,out_path)
    logger.info("info:")
    logger.info(pprint.pformat(file_info))
    
    # Create the classes needed to generate a precomputed slice
    encoder = ChunkEncoder(file_info)
    file_writer = SlideWriter(out_path)
    
    # Open the image in tiles, encode, and save
    for Y in range(0, bf.num_y(),CHUNK_SIZE):
        if Y+CHUNK_SIZE > bf.num_y():
            Y_max = bf.num_y()
        else:
            Y_max = Y+CHUNK_SIZE
        logger.info("Processing Y range: {}-{}".format(Y,Y_max))
        for X in range(0,bf.num_x(),CHUNK_SIZE):
            if X+CHUNK_SIZE > bf.num_x():
                X_max = bf.num_x()
            else:
                X_max = X+CHUNK_SIZE
            logger.info("Processing X range: {}-{}".format(X,X_max))
            # Read the chunk
            image = bf.read_image(X=(X,X_max),Y=(Y,Y_max))
            image = image.reshape(image.shape[:-1])
            image = np.moveaxis(image, (0, 1, 2, 3), (2, 3, 1, 0))
            
            # Encode the chunk
            image_encoded = encoder.encode(image)
            
            # Write the chunk
            file_writer.store_chunk(image_encoded,"0",(X,X_max,Y,Y_max,0,1))
            
    logger.info("Finished precomputing. Closing the javabridge and exiting...")
    jutil.kill_vm()

if __name__ == "__main__":
    main()