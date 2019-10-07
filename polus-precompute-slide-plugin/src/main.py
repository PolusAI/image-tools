from bfio.bfio import BioReader
import numpy as np
import json
import copy
import os
from pathlib import Path
import logging

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

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def _avg2(image):
    image = image.astype('uint64')
    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2
    avg_img = np.zeros(np.ceil([d/2 for d in image.shape]).astype('int'))
    avg_img[0:int(y_max/2),0:int(x_max/2)]= (image[0:y_max-1:2,0:x_max-1:2] + \
                                             image[1:y_max:2,0:x_max-1:2] + \
                                             image[0:y_max-1:2,1:x_max:2] + \
                                             image[1:y_max:2,1:x_max:2]) / 4
    if y_max != image.shape[0]:
        avg_img[-1,:int(x_max/2)] = (image[-1,0:x_max-1:2] + \
                                     image[-1,1:x_max:2]) / 2
    if x_max != image.shape[1]:
        avg_img[:int(y_max/2),-1] = (image[0:y_max-1:2,-1] + \
                                     image[1:y_max:2,-1]) / 2
    if y_max != image.shape[0] and x_max != image.shape[1]:
        avg_img[-1,-1] = image[-1,-1]

    return avg_img

def _get_lower_res(S,bfio_reader,slide_writer,encoder,logger,X=None,Y=None):
    # Get the scale info
    scale_info = None
    for res in encoder.info['scales']:
        if int(res['key'])==S:
            scale_info = res
            break
    if scale_info==None:
        ValueError("No scale information for resolution {}.".format(S))
        
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    
    # Initialize the output
    image = np.zeros((Y[1]-Y[0],X[1]-X[0]),dtype=bfio_reader.read_metadata().image().Pixels.get_PixelType())
    
    # If requesting from the lowest scale, then just read the image
    if S==0:
        image = bfio_reader.read_image(X=X,Y=Y).squeeze()
    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)
        
        for y in range(0,len(subgrid_dims[1])-1):
            y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
            y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
            for x in range(0,len(subgrid_dims[0])-1):
                x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                sub_image = _get_lower_res(X=subgrid_dims[0][x:x+2],
                                           Y=subgrid_dims[1][y:y+2],
                                           S=S-1,
                                           bfio_reader=bfio_reader,
                                           slide_writer=slide_writer,
                                           encoder=encoder,
                                           logger=logger)
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1]] = _avg2(sub_image)

    # Rearrange the image for Neuroglancerx
    image_shifted = np.moveaxis(image.reshape(image.shape[0],image.shape[1],1,1),
                                (0, 1, 2, 3), (2, 3, 1, 0))
    # Encode the chunk
    image_encoded = encoder.encode(image_shifted)
    # Write the chunk
    logger.info("Saving (S,x-X,y-Y): ({},{}-{},{}-{})".format(S,X[0],X[1],Y[0],Y[1]))
    slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],0,1))
    
    return image

# Modified and condensed from FileAccessor class in neuroglancer-scripts
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.pyx_max != image.shape[1]
class SlideWriter():
    """Write a Neuroglancer pre-computed pyramid on the local file system.
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
    """Encode/decode chunks from NumPy aout_pathrrays to byte buffers.
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
        
        self.info = info
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
    
    # create a scales template, use the full resolution8
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
    
    for i in range(1,num_scales+1):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),1]
        current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
        info['scales'].append(current_scale)
        
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
    
    for image in images:
        out_dir = Path(output_dir).joinpath(image.name)
        out_dir.mkdir()
        out_dir = str(out_dir.absolute())
        
        # Create the BioReader object
        logger.info('Getting the BioReader for file: {}'.format(str(image.absolute())))
        bf = BioReader(str(image.absolute()))
        
        # Create the output path and info file
        file_info = bfio_metadata_to_slide_info(bf,out_dir)
        logger.info("data_type: {}".format(file_info['data_type']))
        logger.info("num_channels: {}".format(file_info['num_channels']))
        logger.info("number of scales: {}".format(len(file_info['scales'])))
        logger.info("type: {}".format(file_info['type']))
        
        # Create the classes needed to generate a precomputed slice
        logger.info("Creating encoder...")
        encoder = ChunkEncoder(file_info)
        logger.info("Creating file_writer...")
        file_writer = SlideWriter(out_dir)

        for y in range(0,file_info['scales'][-1]['size'][1],CHUNK_SIZE):
            if y+CHUNK_SIZE > file_info['scales'][-1]['size'][1]:
                y_max = file_info['scales'][-1]['size'][1]
            else:
                y_max = y+CHUNK_SIZE
            for x in range(0,file_info['scales'][-1]['size'][0],CHUNK_SIZE):
                if x+CHUNK_SIZE > file_info['scales'][-1]['size'][0]:
                    x_max = file_info['scales'][-1]['size'][0]
                else:
                    x_max = x+CHUNK_SIZE
            
                image = _get_lower_res(int(file_info['scales'][-1]['key']),bf,file_writer,encoder,logger,X=[x,x_max],Y=[y,y_max])
            
                # Rearrange the image for Neuroglancer
                image_shifted = np.moveaxis(image.reshape(image.shape[0],image.shape[1],1,1),
                                            (0, 1, 2, 3), (2, 3, 1, 0))
                # Encode the chunk
                image_encoded = encoder.encode(image_shifted)
                # Write the chunk
                logger.info("Saving (S,x-X,y-Y): ({},{}-{},{}-{})".format(file_info['scales'][-1]['key'],x,x_max,y,y_max))
                file_writer.store_chunk(image_encoded,file_info['scales'][-1]['key'],(x,x_max,y,y_max,0,1))
    
    logger.info("Finished precomputing. Closing the javabridge and exiting...")
    jutil.kill_vm()

if __name__ == "__main__":
    main()