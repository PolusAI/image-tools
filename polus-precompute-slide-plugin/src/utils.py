import copy, os, json, filepattern, imageio, pathlib, typing, abc, zarr
import bfio
import numpy as np
from numcodecs import Blosc
from concurrent.futures import ThreadPoolExecutor
from preadator import ProcessManager
from bfio.OmeXml import OMEXML

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 1024

def _mode2(image: np.ndarray) -> np.ndarray:
    """ Find mode of pixels in optical field 2x2 and stride 2
    
    This method approximates the mode by finding the largest number that occurs
    at least twice in a 2x2 grid of pixels, then sets that value to the output
    pixel.
    
    Args:
        image - numpy array with only two dimensions (m,n)
    Returns:
        mode_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """

    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2

    # Initialize the mode output image (Half the size)
    mode_img = np.zeros(np.ceil([d/2 for d in image.shape]).astype(int),dtype=image.dtype)
    
    # Default the output to the upper left pixel value
    mode_img[0:y_max//2,0:x_max//2] = image[0:-1:2, 0:-1:2]
    
    # Handle images with odd-valued image dimensions
    if y_max != image.shape[0]:
        mode_img[-1,:x_max//2] = image[-1,0:x_max-1:2]
    if x_max != image.shape[1]:
        mode_img[:y_max//2,-1] = image[0:y_max-1:2,-1]
    if y_max != image.shape[0] and x_max != image.shape[1]:
        mode_img[-1,-1] = image[-1,-1]
        
    # Garnering the four different pixels that we would find the modes of
    # Finding the mode of: 
    # vals00[1], vals01[1], vals10[1], vals11[1] 
    # vals00[2], vals01[2], vals10[2], vals11[2]
    # etc 
    vals00 = image[0:-1:2, 0:-1:2]
    vals01 = image[0:-1:2,   1::2]
    vals10 = image[  1::2, 0:-1:2]
    vals11 = image[  1::2,   1::2]

    # Finding where pixels adjacent to the top left pixel are not identical
    index = (vals00 != vals01) | (vals00 != vals10)

    # Initialize indexes where the two pixels are not the same
    valueslist = [vals00[index], vals01[index], vals10[index], vals11[index]]
    
    # Do a deeper mode search for non-matching pixels
    temp_mode = mode_img[:y_max//2,:x_max//2]
    for i in range(3):
        rvals = valueslist[i]
        for j in range(i+1,4):
            cvals = valueslist[j]
            ind = np.logical_and(cvals==rvals,rvals>temp_mode[index])
            temp_mode[index][ind] = rvals[ind]
        
    mode_img[:y_max//2,:x_max//2] = temp_mode

    return mode_img

def _avg2(image: np.ndarray) -> np.ndarray:
    """ Average pixels together with optical field 2x2 and stride 2
    
    Args:
        image - numpy array with only two dimensions (m,n)
    Returns:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    
    # Since we are adding pixel values, we need to update the pixel type 
    # This helps to avoid integer overflow
    if image.dtype == np.uint8:
        dtype = np.uint16
    elif image.dtype == np.uint16:
        dtype = np.uint32
    elif image.dtype == np.uint32:
        dtype = np.uint64
    elif image.dtype == np.int8:
        dtype = np.int16
    elif image.dtype == np.int16:
        dtype = np.int32
    elif image.dtype == np.int32:
        dtype = np.int64
    else:
        dtype = image.dtype
        
    odtype = image.dtype
    image = image.astype(dtype)
    
    y_max = image.shape[0] - image.shape[0] % 2
    x_max = image.shape[1] - image.shape[1] % 2
    
    # Calculate the mean
    avg_img = np.zeros(np.ceil([d/2 for d in image.shape]).astype(int),dtype=dtype)
    avg_img[0:y_max//2,0:x_max//2] = (image[0:y_max-1:2, 0:x_max-1:2] + \
                                      image[1:  y_max:2, 0:x_max-1:2] + \
                                      image[0:y_max-1:2,   1:x_max:2] + \
                                      image[1:  y_max:2,   1:x_max:2]) // 4
    
    # Fill in the final row if the image height is odd-valued
    if y_max != image.shape[0]:
        avg_img[-1,:x_max//2] = (image[-1,0:x_max-1:2] + \
                                 image[-1,1:x_max:2]) // 2
    # Fill in the final column if the image width is odd-valued
    if x_max != image.shape[1]:
        avg_img[:y_max//2,-1] = (image[0:y_max-1:2,-1] + \
                                 image[1:y_max:2,-1]) // 2
    # Fill in the lower right pixel if both image width and height are odd
    if y_max != image.shape[0] and x_max != image.shape[1]:
        avg_img[-1,-1] = image[-1,-1]
        
    return avg_img.astype(odtype)

# Modified and condensed from FileAccessor class in neuroglancer-scripts
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/file_accessor.py
class PyramidWriter():
    """ Pyramid file writing base class
    This class should not be called directly. It should be inherited by a pyramid
    writing class type.
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    chunk_pattern = None

    def __init__(self,
                 base_dir: typing.Union[pathlib.Path,str],
                 image_path: typing.Union[pathlib.Path,str],
                 image_depth: int = 0,
                 output_depth: int = 0,
                 max_output_depth: int = None,
                 image_type: str = "image"):
        
        if isinstance(image_path,str):
            image_path = pathlib.Path(image_path)
        self.image_path = image_path
        if isinstance(base_dir,str):
            base_path = pathlib.Path(base_path)
        self.base_path = base_dir
        self.image_depth = image_depth
        self.output_depth = output_depth
        self.max_output_depth = max_output_depth
        self.image_type = image_type
        
        if image_type == 'image':
            self.scale = _avg2
        elif image_type == 'segmentation':
            self.scale = _mode2
        else:
            raise ValueError('image_type must be one of ["image","segmentation"]')
            
        self.info = bfio_metadata_to_slide_info(self.image_path,
                                                self.base_path,
                                                self.max_output_depth,
                                                self.image_type)
        
        self.dtype = self.info['data_type']
        
        self.encoder = self._encoder()
    
    @abc.abstractmethod
    def _encoder(self):
        pass
    
    @abc.abstractmethod
    def _write_chunk(self,key,chunk_path,buf):
        pass
    
    @abc.abstractmethod
    def write_info(self):
        pass
    
    @abc.abstractmethod
    def write_segment_info(self):
        pass
    
    def write_slide(self):
        
        with ProcessManager.process(f'{self.base_path} - {self.output_depth}'):
            
            ProcessManager.submit_thread(self._write_slide)
            
            ProcessManager.join_threads()
    
    def scale_info(self,S):
        
        if S == -1:
            return self.info['scales'][0]
        
        scale_info = None
        
        for res in self.info['scales']:
            if int(res['key'])==S:
                scale_info = res
                break
            
        if scale_info==None:
            ValueError("No scale information for resolution {}.".format(S))
            
        return scale_info

    def store_chunk(self, image, key, chunk_coords):
        """ Store a pyramid chunk
        
        Inputs:
            image: byte stream to save to disk
            key: pyramid scale, folder to save chunk to
            chunk_coords: X,Y,Z coordinates of data in buf
        """
        
        buf = self.encoder.encode(image)
        
        self._write_chunk(key,chunk_coords,buf)

    def _chunk_path(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        chunk_coords = self._chunk_coords(chunk_coords)
        chunk_filename = pattern.format(*chunk_coords, key=key)
        return self.base_path / chunk_filename

    def _chunk_coords(self,chunk_coords):
        if len(chunk_coords) == 4:
            chunk_coords = chunk_coords + (self.output_depth,self.output_depth+1)
        elif len(chunk_coords) != 6:
            raise ValueError('chunk_coords must be a 4-tuple or a 6-tuple.')
        return chunk_coords

def _get_higher_res(S: int,
                    slide_writer: PyramidWriter,
                    X: typing.Tuple[int,int] = None,
                    Y: typing.Tuple[int,int] = None,
                    Z: typing.Tuple[int,int] = (0,1)):
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
    
    Args:
        S: Top level scale from which the pyramid will be built
        file_path: Path to image
        slide_writer: object used to encode and write pyramid tiles
        X: Range of X values [min,max] to get at the indicated scale
        Y: Range of Y values [min,max] to get at the indicated scale
    Returns:
        image: The image corresponding to the X,Y values at scale S
    """
    
    # Get the scale info
    scale_info = slide_writer.scale_info(S)
    
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    
    if str(S)==slide_writer.scale_info(-1)['key']:
        with ProcessManager.thread():
        
            with bfio.BioReader(slide_writer.image_path,max_workers=1) as br:
            
                image = br[Y[0]:Y[1],X[0]:X[1],Z[0]:Z[1],...].squeeze()

            # Write the chunk
            slide_writer.store_chunk(image,str(S),(X[0],X[1],Y[0],Y[1]))
        
        return image

    else:
        # Initialize the output
        image = np.zeros((Y[1]-Y[0],X[1]-X[0]),dtype=slide_writer.dtype)
        
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)

        def load_and_scale(*args,**kwargs):
            sub_image = _get_higher_res(**kwargs)

            with ProcessManager.thread():
                image = args[0]
                x_ind = args[1]
                y_ind = args[2]
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1]] = kwargs['slide_writer'].scale(sub_image)
        
        with ThreadPoolExecutor(1) as executor:
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                    x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                    executor.submit(load_and_scale,
                                    image,x_ind,y_ind,           # args
                                    X=subgrid_dims[0][x:x+2],    # kwargs
                                    Y=subgrid_dims[1][y:y+2],
                                    Z=Z,
                                    S=S+1,
                                    slide_writer=slide_writer)
    
    # Write the chunk
    slide_writer.store_chunk(image,str(S),(X[0],X[1],Y[0],Y[1]))
    return image

class NeuroglancerWriter(PyramidWriter):
    """ Method to write a Neuroglancer pre-computed pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"
        
        min_level = min([int(self.scale_info(-1)['key']),10])
        self.info = bfio_metadata_to_slide_info(self.image_path,
                                                self.base_path,
                                                self.max_output_depth,
                                                self.image_type,
                                                min_level)
        
        if self.image_type == 'segmentation':
            self.labels = set()
            
    def store_chunk(self, image, key, chunk_coords):
        
        # Add in a label aggregator to the store_chunk operation
        # Only aggregate labels at the highest resolution
        if self.image_type == 'segmentation':
            if key == self.scale_info(-1)['key']:
                self.labels = self.labels.union(set(np.unique(image)))
            elif key == self.info['scales'][-1]['key']:
                root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
                if str(self.output_depth) not in root.array_keys():
                    labels = root.empty(str(self.output_depth),
                                        shape=(len(self.labels),),
                                        dtype=np.uint64)
                else:
                    labels = root[str(self.output_depth)]
                labels[:] = np.asarray(list(self.labels),np.uint64).squeeze()
            
        super().store_chunk(image, key, chunk_coords)

    def _write_chunk(self,key,chunk_coords,buf):
        chunk_path = self._chunk_path(key,chunk_coords)
        os.makedirs(str(chunk_path.parent), exist_ok=True)
        with open(str(chunk_path.with_name(chunk_path.name)),'wb') as f:
            f.write(buf)
            
    def _encoder(self):
        
        return NeuroglancerChunkEncoder(self.info)
    
    def _write_slide(self):
        
        pathlib.Path(self.base_path).mkdir(exist_ok=True)
    
        # Don't create a full pyramid to help reduce bounding box size
        start_level = int(self.info['scales'][-1]['key'])
        image = _get_higher_res(start_level,self,
                                Z=(self.image_depth,self.image_depth+1))

    def write_info(self):
        """ This creates the info file specifying the metadata for the precomputed format """

        # Create an output path object for the info file
        op = pathlib.Path(self.base_path)
        op.mkdir(exist_ok=True,parents=True)
        op = op.joinpath("info")

        # Write the neuroglancer info file
        with open(op,'w') as writer:
            json.dump(self.info,writer,indent=2)
            
        if self.image_type == 'segmentation':
            self._write_segment_info()
            
    def _write_segment_info(self):
        """ This function creates the info file needed to segment the image """
        if self.image_type != 'segmentation':
            raise TypeError('The NeuroglancerWriter object must have image_type = "segmentation" to use write_segment_info.')
        
        op = pathlib.Path(self.base_path).joinpath("infodir")
        op.mkdir(exist_ok=True)
        op = op.joinpath("info")
        
        # Get the labels
        root = zarr.open(str(self.base_path.joinpath("labels.zarr")))
        labels = set()
        for d in root.array_keys():
            labels = labels.union(set(root[d][:].squeeze().tolist()))

        inlineinfo = {
            "ids":[str(item) for item in labels],
            "properties":[
                {
                "id":"label",
                "type":"label",
                "values":[str(item) for item in labels]
                },
                {
                "id":"description",
                "type":"label",
                "values": [str(item) for item in labels]
                }
            ]
        }

        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": inlineinfo
        }

        # writing all the information into the file
        with open(op,'w') as writer:
            json.dump(info,writer,indent=2)
            
class ZarrWriter(PyramidWriter):
    """ Method to write a Zarr pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        out_name = self.base_path.name.replace(''.join(self.base_path.suffixes),'')
        self.base_path = self.base_path.with_name(out_name)
        self.base_path.mkdir(exist_ok=True)
        self.root = zarr.open(str(self.base_path.joinpath("data.zarr").resolve()),
                              mode='a')
        if "0" in self.root.group_keys():
            self.root = self.root["0"]
        else:
            self.root = self.root.create_group("0")
        
        
        self.writers = {}
        max_scale = int(self.scale_info(-1)['key'])
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        for S in range(10,len(self.info['scales'])):
            scale_info = self.scale_info(S)
            key = str(max_scale - int(scale_info['key']))
            if key not in self.root.array_keys():
                self.writers[key] = self.root.zeros(key,
                                                    shape=(1,self.max_output_depth,1) + tuple(scale_info['size'][0:2]),
                                                    chunks=(1,1,1,CHUNK_SIZE,CHUNK_SIZE),
                                                    dtype=self.dtype,
                                                    compressor=compressor)
            else:
                self.root[key].resize((1,self.max_output_depth,1) + tuple(scale_info['size'][0:2]))
                self.writers[key] = self.root[key]

    def _write_chunk(self,key,chunk_coords,buf):
        key = str(int(self.scale_info(-1)['key']) - int(key))
        chunk_coords = self._chunk_coords(chunk_coords)
        self.writers[key][0:1,
                          chunk_coords[4]:chunk_coords[5],
                          0:1,
                          chunk_coords[2]:chunk_coords[3],
                          chunk_coords[0]:chunk_coords[1]] = buf
            
    def _encoder(self):
        
        return ZarrChunkEncoder(self.info)
    
    def _write_slide(self):
    
        _get_higher_res(10,self,Z=(self.image_depth,self.image_depth+1))
            
    def write_info(self):
        """ This creates the multiscales metadata for zarr pyramids """
        # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
        multiscales = [{
            "version": "0.1",
            "name": self.base_path.name,
            "datasets": [],
            "metadata": {
                "method": "mean"
            }
        }]
        
        pad = len(self.scale_info(-1)['key'])
        max_scale = int(self.scale_info(-1)['key'])
        for S in reversed(range(10,len(self.info['scales']))):
            scale_info = self.scale_info(S)
            key = str(max_scale - int(scale_info['key']))
            multiscales[0]["datasets"].append({"path": key})
        self.root.attrs["multiscales"] = multiscales
        
        with bfio.BioReader(self.image_path,max_workers=1) as bfio_reader:
            
            metadata = OMEXML(str(bfio_reader.metadata))
            metadata.image(0).Pixels.SizeC = self.max_output_depth
            metadata.image(0).Pixels.channel_count = self.max_output_depth
            
            for c in range(self.max_output_depth):
                metadata.image().Pixels.Channel(c).Name = f'Channel {c}'
            
            with open(self.base_path.joinpath("METADATA.ome.xml"),'x') as fw:
                
                fw.write(str(metadata).replace("<ome:","<").replace("</ome:","</"))

class DeepZoomWriter(PyramidWriter):
    """ Method to write a DeepZoom pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_pattern = "{key}/{0}_{1}.png"
        self.base_path = self.base_path.joinpath(str(self.output_depth) + '_files')

    def _chunk_coords(self,chunk_coords):
        chunk_coords = [chunk_coords[0]//CHUNK_SIZE,chunk_coords[2]//CHUNK_SIZE]
        return chunk_coords

    def _write_chunk(self,key,chunk_coords,buf):
        chunk_path = self._chunk_path(key,chunk_coords)
        os.makedirs(str(chunk_path.parent), exist_ok=True)
        imageio.imwrite(str(chunk_path.with_name(chunk_path.name)),buf,format='PNG-FI',compression=1)
    
    def write_info(self):
        # Create an output path object for the info file
        op = pathlib.Path(self.base_path).parent.joinpath("{}.dzi".format(self.output_depth))
        
        # DZI file template
        DZI = '<?xml version="1.0" encoding="utf-8"?><Image TileSize="{}" Overlap="0" Format="png" xmlns="http://schemas.microsoft.com/deepzoom/2008"><Size Width="{}" Height="{}"/></Image>'
        
        # write the dzi file
        with open(op,'w') as writer:
            writer.write(DZI.format(CHUNK_SIZE,self.info['scales'][0]['size'][0],self.info['scales'][0]['size'][1]))
    
    def _write_slide(self):
        
        pathlib.Path(self.base_path).mkdir(exist_ok=False)
        
        _get_higher_res(0,self,Z=(self.image_depth,self.image_depth+1))

    def _encoder(self):
        
        return DeepZoomChunkEncoder(self.info)

    def write_segment_info(self):
        raise NotImplementedError('DeepZoom does not have a segmentation format.')

# Modified and condensed from multiple functions and classes
# https://github.com/HumanBrainProject/neuroglancer-scripts/blob/master/src/neuroglancer_scripts/chunk_encoding.py
class ChunkEncoder:
    # Data types used by Neuroglancer
    DATA_TYPES = ("uint8",  "int8",
                  "uint16", "int16",
                  "uint32", "int32",
                  "uint64", "int64",
                  "float32")
    
    def __init__(self, info):
        
        try:
            data_type = info["data_type"]
            num_channels = info["num_channels"]
        except KeyError as exc:
            raise KeyError("The info dict is missing an essential key {0}".format(exc)) from exc
        
        if not isinstance(num_channels, int) or not num_channels > 0:
            raise KeyError("Invalid value {0} for num_channels (must be a positive integer)".format(num_channels))
        
        if data_type not in ChunkEncoder.DATA_TYPES:
            raise KeyError("Invalid data_type {0} (should be one of {1})".format(data_type, ChunkEncoder.DATA_TYPES))
        
        self.info = info
        self.num_channels = num_channels
        self.dtype = np.dtype(data_type).newbyteorder("<")
        
    @abc.abstractmethod
    def encode(self,chunk):
        pass

class NeuroglancerChunkEncoder(ChunkEncoder):

    def encode(self, chunk):
        """ Encode a chunk from a Numpy array into bytes.
        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream)
        """
        
        # Rearrange the image for Neuroglancer
        chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],1,1),
                            (0, 1, 2, 3), (2, 3, 1, 0))
        chunk = np.asarray(chunk).astype(self.dtype)
        assert chunk.ndim == 4
        assert chunk.shape[0] == self.num_channels
        buf = chunk.tobytes()
        return buf
    
class ZarrChunkEncoder(ChunkEncoder):

    def encode(self, chunk):
        """ Encode a chunk from a Numpy array into bytes.
        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream)
        """
        
        # Rearrange the image for Neuroglancer
        chunk = chunk.reshape(chunk.shape[0],chunk.shape[1],1,1,1).transpose(4,2,3,0,1)
        chunk = np.asarray(chunk).astype(self.dtype)
        return chunk

class DeepZoomChunkEncoder(ChunkEncoder):

    def encode(self, chunk):
        """ Encode a chunk for DeepZoom
        
        Nothing special to do for encoding except checking the number of
        dimentions.
        
        Inputs:
            chunk - array with 2 dimensions
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Check to make sure the data is formatted properly
        assert chunk.ndim == 2
        return chunk

def bfio_metadata_to_slide_info(image_path,outPath,stackheight,imagetype,min_scale=0):
    """ Generate a Neuroglancer info file from Bioformats metadata
    
    Neuroglancer requires an info file in the root of the pyramid directory.
    All information necessary for this info file is contained in Bioformats
    metadata, so this function takes the metadata and generates the info file.
    
    Inputs:
        bfio_reader - A BioReader object
        outPath - Path to directory where pyramid will be generated
    Outputs:
        info - A dictionary containing the information in the info file
    """
    with bfio.BioReader(image_path,max_workers=1) as bfio_reader:
        # Get metadata info from the bfio reader
        sizes = [bfio_reader.X,bfio_reader.Y,stackheight]
        
        phys_x = bfio_reader.ps_x
        if None in phys_x:
            phys_x = (1000,'nm')
        
        phys_y = bfio_reader.ps_y
        if None in phys_y:
            phys_y = (1000,'nm')
            
        phys_z = bfio_reader.ps_z
        if None in phys_z:
            phys_z = ((phys_x[0] + phys_y[0]) / 2,phys_x[1])
            
        resolution = [phys_x[0] * UNITS[phys_x[1]]]
        resolution.append(phys_y[0] * UNITS[phys_y[1]])
        resolution.append(phys_z[0] * UNITS[phys_z[1]]) # Just used as a placeholder
        dtype = str(np.dtype(bfio_reader.dtype))
    
    num_scales = int(np.ceil(np.log2(max(sizes))))
    
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
        "num_channels": 1,
        "scales": [scales],
        "type": imagetype,
    }
    
    if imagetype == "segmentation":
        info["segment_properties"] = "infodir"

    for i in reversed(range(min_scale,num_scales)):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),stackheight]
        current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
        info['scales'].append(current_scale)
    
    return info
