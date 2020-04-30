from bfio.bfio import BioReader
import numpy as np
import json, copy, os
from pathlib import Path
import imageio
import filepattern
import os

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 1024

def recursivefiles(files, stackvars, stackvals, stackby, stackheight, pattern):
    channels = files
    if len(stackvars) > 1:
        return recursivefiles(files[stackvals[0]], stackvars[1:], stackvals[1:], stackby, stackheight, pattern)
    else:
        channels = [files[stackvals[0]][i]['file'] for i in range(0, stackheight)]
        channelvals = [filepattern.parse_filename(os.path.basename(files[stackvals[0]][i]['file']), pattern=pattern)[stackby] 
                        for i in range(0, stackheight)]
        channelvals.sort()
        channels.sort()
        channels = [Path(i) for i in channels]
        return channels, channelvals

def _avg2(image):
    """ Average pixels together with optical field 2x2 and stride 2
    
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
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

def _get_higher_res(S, zlevel, bfio_reader,slide_writer,encoder,X=None,Y=None):
    """ Recursive function for pyramid building
    
    This is a recursive function that builds an image pyramid by indicating
    an original region of an image at a given scale. This function then
    builds a pyramid up from the highest resolution components of the pyramid
    (the original images) to the given position resolution.
    
    As an example, imagine the following possible pyramid:
    
    Scale S=0                     1234
                                 /    \
    Scale S=1                  12      34
                              /  \    /  \
    Scale S=2                1    2  3    4
    
    At scale 2 (the highest resolution) there are 4 original images. At scale 1,
    images are averaged and concatenated into one image (i.e. image 12). Calling
    this function using S=0 will attempt to generate 1234 by calling this
    function again to get image 12, which will then call this function again to
    get image 1 and then image 2. Note that this function actually builds images
    in quadrants (top left and right, bottom left and right) rather than two
    sections as displayed above.
    
    Due to the nature of how this function works, it is possible to build a
    pyramid in parallel, since building the subpyramid under image 12 can be run
    independently of the building of subpyramid under 34.
    
    Inputs:
        S - Top level scale from which the pyramid will be built
        bfio_reader - BioReader object used to read the tiled tiff
        slide_writer - SlideWriter object used to write pyramid tiles
        encoder - ChunkEncoder object used to encode numpy data to byte stream
        X - Range of X values [min,max] to get at the indicated scale
        Y - Range of Y values [min,max] to get at the indicated scale
    Outputs:
        image - The image corresponding to the X,Y values at scale S
    """
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
    if str(S)==encoder.info['scales'][0]['key']:
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
                sub_image = _get_higher_res(X=subgrid_dims[0][x:x+2],
                                            Y=subgrid_dims[1][y:y+2],
                                            S=S+1,
                                            zlevel=zlevel,
                                            bfio_reader=bfio_reader,
                                            slide_writer=slide_writer,
                                            encoder=encoder)
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1]] = _avg2(sub_image)

    # Encode the chunk
    image_encoded = encoder.encode(image)
    # Write the chunk
    slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],zlevel,zlevel + 1))
    
    return image

# Modified and condensed from FileAccessor class in neuroglancer-scripts
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.py
class PyramidWriter():
    """ Pyramid file writing base class
    This class should not be called directly. It should be inherited by a pyramid
    writing class type.
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    can_write = True
    chunk_pattern = None

    def __init__(self, base_dir):
        self.base_path = Path(base_dir)

    def store_chunk(self, buf, key, chunk_coords):
        """ Store a pyramid chunk
        
        Inputs:
            buf - byte stream to save to disk
            key - pyramid scale, folder to save chunk to
            chunk_coords - X,Y,Z coordinates of data in buf
        """
        try:
            self._write_chunk(key,chunk_coords,buf)
        except OSError as exc:
            raise FileNotFoundError(
                "Error storing chunk {0} in {1}: {2}" .format(
                    self._chunk_path(key, chunk_coords),
                    self.base_path, exc))

    def _chunk_path(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        chunk_coords = self._chunk_coords(chunk_coords)
        chunk_filename = pattern.format(*chunk_coords, key=key)
        return self.base_path / chunk_filename

    def _chunk_coords(self,chunk_coords):
        return chunk_coords

    def _write_chunk(self,key,chunk_path,buf):
        NotImplementedError("_write_chunk was never implemented.")

class NeuroglancerWriter(PyramidWriter):
    """ Method to write a Neuroglancer pre-computed pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

    def _write_chunk(self,key,chunk_coords,buf):
        chunk_path = self._chunk_path(key,chunk_coords)
        os.makedirs(str(chunk_path.parent), exist_ok=True)
        with open(str(chunk_path.with_name(chunk_path.name)),'wb') as f:
            f.write(buf)

class DeepZoomWriter(PyramidWriter):
    """ Method to write a DeepZoom pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.chunk_pattern = "{key}/{0}_{1}.png"

    def _chunk_coords(self,chunk_coords):
        chunk_coords = [chunk_coords[0]//CHUNK_SIZE,chunk_coords[2]//CHUNK_SIZE]
        return chunk_coords

    def _write_chunk(self,key,chunk_coords,buf):
        chunk_path = self._chunk_path(key,chunk_coords)
        os.makedirs(str(chunk_path.parent), exist_ok=True)
        imageio.imwrite(str(chunk_path.with_name(chunk_path.name)),buf,format='PNG-FI',compression=1)

# Modified and condensed from multiple functions and classes
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/chunk_encoding.py
class NeuroglancerChunkEncoder:
    """ Encode chunks from Numpy array to byte buffer.
    
    Inputs:
        info - info dictionary
    """

    # Data types used by Neuroglancer
    DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")

    def __init__(self, info):
        
        try:
            data_type = info["data_type"]
            num_channels = info["num_channels"]
        except KeyError as exc:
            raise KeyError("The info dict is missing an essential key {0}"
                                .format(exc)) from exc
        if not isinstance(num_channels, int) or not num_channels > 0:
            raise KeyError("Invalid value {0} for num_channels (must be "
                                "a positive integer)".format(num_channels))
        if data_type not in self.DATA_TYPES:
            raise KeyError("Invalid data_type {0} (should be one of {1})"
                                .format(data_type, self.DATA_TYPES))
        
        self.info = info
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")

    def encode(self, chunk):
        """ Encode a chunk from a Numpy array into bytes.
        Inputs:
            chunk - array with four dimensions (C, Z, Y, X)
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Rearrange the image for Neuroglancer
        chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],1,1),
                            (0, 1, 2, 3), (2, 3, 1, 0))
        chunk = np.asarray(chunk).astype(self.dtype, casting="safe")
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = chunk.tobytes()
        return buf

class DeepZoomChunkEncoder(NeuroglancerChunkEncoder):
    """ Properly formats numpy array for DeepZoom pyramid.
    
    Inputs:
        info - info dictionary
    """

    # Data types used by Neuroglancer
    DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")

    def __init__(self, info):
        super().__init__(info)

    def encode(self, chunk):
        """ Squeeze the input array.
        Inputs:
            chunk - array with four dimensions (C, Z, Y, X)
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Check to make sure the data is formatted properly
        assert chunk.ndim == 2
        return chunk

def bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight):
    """ Generate a Neuroglancer info file from Bioformats metadata
    
    Neuroglancer requires an info file in the root of the pyramid directory.
    All information necessary for this info file is contained in Bioformats
    metadata, so this function takes the metadata and generates the info
    file.
    
    Inputs:
        bfio_reader - A BioReader object
        outPath - Path to directory where pyramid will be generated
    Outputs:
        info - A dictionary containing the information in the info file
    """
    
    # Get metadata info from the bfio reader
    # sizes = [bfio_reader.num_x(),bfio_reader.num_y(),bfio_reader.num_z()]
    sizes = [bfio_reader.num_x(),bfio_reader.num_y(),stackheight]
    phys_x = bfio_reader.physical_size_x()
    if None in phys_x:
        phys_x = (1000,'nm')
    phys_y = bfio_reader.physical_size_y()
    if None in phys_y:
        phys_y = (1000,'nm')
    resolution = [phys_x[0] * UNITS[phys_x[1]]]
    resolution.append(phys_y[0] * UNITS[phys_y[1]])
    resolution.append((phys_y[0] * UNITS[phys_y[1]] + phys_x[0] * UNITS[phys_x[1]])/2) # Just used as a placeholder
    dtype = bfio_reader.read_metadata().image().Pixels.get_PixelType()
    
    num_scales = int(np.log2(max(sizes))) + 1
    
    # create a scales template, use the full resolution8
    scales = {
        "chunk_sizes":[[CHUNK_SIZE,CHUNK_SIZE,1]],
        "encoding":"raw",
        "key": str(num_scales),
        "resolution":resolution,
        "size":sizes,
        "voxel_offset":[0,0,0]
    }
    
    # initialize the json dictionary
    info = {
        "data_type": dtype,
        "num_channels":1,
        "scales": [scales],       # Will build scales below
        "type": "image"
    }
    
    for i in range(1,num_scales+1):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(num_scales - i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),stackheight]
        current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
        info['scales'].append(current_scale)
    
    return info

def neuroglancer_info_file(bfio_reader,outPath, stackheight):
    # Create an output path object for the info file
    op = Path(outPath).joinpath("info")
    
    # Get pyramid info
    info = bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight)

    # Write the neuroglancer info file
    with open(op,'w') as writer:
        writer.write(json.dumps(info))
        
    return info

def dzi_file(bfio_reader,outPath,imageNum):
    # Create an output path object for the info file
    op = Path(outPath).parent.joinpath("{}.dzi".format(imageNum))
    
    # DZI file template
    DZI = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="{}" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'
    
    # Get pyramid info
    info = bfio_metadata_to_slide_info(bfio_reader,outPath)

    # write the dzi file
    with open(op,'w') as writer:
        writer.write(DZI.format(CHUNK_SIZE,info['scales'][0]['size'][0],info['scales'][0]['size'][1]))
        
    return info
