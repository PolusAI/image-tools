"""
This file originally appeared in the polus precomputed slide plugin.
It has been modified to create color DeepZoom pyramids.

Original Code:
https://github.com/LabShare/polus-plugins/tree/master/polus-precompute-slide-plugin
"""

from bfio.bfio import BioReader
import numpy as np
import copy, os
from pathlib import Path
import imageio, re, filepattern
from concurrent.futures import ThreadPoolExecutor

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
STITCH_LINE = "file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n"

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 1024

def get_number(s):
    """ Check that s is number
    
    In this plugin, heatmaps are created only for columns that contain numbers. This
    function checks to make sure an input value is able to be converted into a number.
    
    This function originally appeared in the image asembler plugin:
    https://github.com/Nicholas-Schaub/polus-plugins/blob/imageassembler/polus-image-assembler-plugin/src/main.py
    
    Inputs:
        s - An input string or number
    Outputs:
        value - Either float(s) or False if s cannot be cast to float
    """
    try:
        return int(s)
    except ValueError:
        return s

class BioAssembler():
    
    def __init__(self,file_path,stitch_path,max_workers=None):
        self._file_path = file_path
        self._stitch_path = stitch_path
        self._file_dict = self._parse_stitch(stitch_path,file_path)
        self._max_workers = max_workers
        self.X = [0,0]
        self.Y = [0,0]
        self.Z = [0,0]
        self._X_offset = 0
        self._Y_offset = 0
        self._Z_offset = 0
        self._image = None
        
    def physical_size_x(self):
        return [None,None]
    
    def physical_size_y(self):
        return [None,None]
    
    def physical_size_z(self):
        return [None,None]
        
    def num_x(self):
        return self._file_dict['width']
    
    def num_y(self):
        return self._file_dict['height']
    
    def num_z(self):
        return 1
        
    def buffer_image(self,image_path,Xi,Yi,Xt,Yt,color=False):
        """buffer_image Load and image and store in buffer

        This method loads an image and stores it in the appropriate
        position based on the stitching vector coordinates within
        a large tile of the output image. It is intended to be
        used as a thread to increase the reading component to
        assembling the image.
        
        Args:
            image_path ([str]): Path to image to load
            Xi ([list]): Xmin and Xmax of pixels to load from the image
            Yi ([list]): Ymin and Ymax of pixels to load from the image
            Xt ([list]): X position within the buffer to store the image
            Yt ([list]): Y position within the buffer to store the image
        """
        
        # Load the image
        br = BioReader(image_path,max_workers=2)
        image = br.read_image(X=Xi,Y=Yi) # only get the first z,c,t layer
            
        # Put the image in the buffer
        if color != None:
            image_temp = (255*(image[...,0,0].astype(np.float32) - self.bounds[0])/(self.bounds[1] - self.bounds[0]))
            image_temp[image_temp>255] = 255
            image_temp[image_temp<0] = 0
            image_temp = image_temp.astype(np.uint8)
            self._image[Yt[0]:Yt[1],Xt[0]:Xt[1],...] = 0
            self._image[Yt[0]:Yt[1],Xt[0]:Xt[1],self.color] = image_temp
        else:
            self._image[Yt[0]:Yt[1],Xt[0]:Xt[1],...] = image[:,:,:,0,0]
        
    def make_tile(self,x_min,x_max,y_min,y_max,color=None):
        """make_tile Create a supertile

        This method identifies images that have stitching vector positions
        within the bounds of the supertile defined by the x and y input
        arguments. It then spawns threads to load images and store in the
        supertile buffer. Finally it returns the assembled supertile to
        allow the main thread to generate the write thread.

        Args:
            x_min ([int]): Minimum x bound of the tile
            x_max ([int]): Maximum x bound of the tile
            y_min ([int]): Minimum y bound of the tile
            y_max ([int]): Maximum y bound of the tile
            stitchPath ([str]): Path to the stitching vector

        Returns:
            [type]: [description]
        """
        
        self._X_offset = x_min
        self._Y_offset = y_min

        # Get the data type
        br = BioReader(str(Path(self._file_path).joinpath(self._file_dict['filePos'][0]['file'])))
        dtype = br._pix['type']

        # initialize the image
        if color!=None:
            self._image = np.full((y_max-y_min,x_max-x_min,4),color,dtype=dtype)
        else:
            self._image = np.zeros((y_max-y_min,x_max-x_min,1),dtype=dtype)

        # get images in bounds of current super tile
        with ThreadPoolExecutor(max([self._max_workers,2])) as executor:
            for f in self._file_dict['filePos']:
                if (f['posX'] >= x_min and f['posX'] <= x_max) or (f['posX']+f['width'] >= x_min and f['posX']+f['width'] <= x_max):
                    if (f['posY'] >= y_min and f['posY'] <= y_max) or (f['posY']+f['height'] >= y_min and f['posY']+f['height'] <= y_max):
                
                        # get bounds of image within the tile
                        Xt = [max(0,f['posX']-x_min)]
                        Xt.append(min(x_max-x_min,f['posX']+f['width']-x_min))
                        Yt = [max(0,f['posY']-y_min)]
                        Yt.append(min(y_max-y_min,f['posY']+f['height']-y_min))

                        # get bounds of image within the image
                        Xi = [max(0,x_min - f['posX'])]
                        Xi.append(min(f['width'],x_max - f['posX']))
                        Yi = [max(0,y_min - f['posY'])]
                        Yi.append(min(f['height'],y_max - f['posY']))
                        
                        # self.buffer_image(str(Path(self._file_path).joinpath(f['file'])),Xi,Yi,Xt,Yt,color)
                        executor.submit(self.buffer_image,str(Path(self._file_path).joinpath(f['file'])),Xi,Yi,Xt,Yt,color)
    
    def _parse_stitch(self,stitchPath,imagePath):
        """ Load and parse image stitching vectors
        
        This function creates a list of file dictionaries that include the filename and
        pixel position and dimensions within a stitched image. It also determines the
        size of the final stitched image and the suggested name of the output image based
        on differences in file names in the stitching vector.
        
        This method originally appeared in the image assembler plugin:
        https://github.com/Nicholas-Schaub/polus-plugins/blob/imageassembler/polus-image-assembler-plugin/src/main.py

        Inputs:
            stitchPath - A path to stitching vectors
            imagePath - A path to tiled tiff images
            timepointName - Use the vector timeslice as the image name
        Outputs:
            out_dict - Dictionary with keys (width, height, name, filePos)
        """

        # Initialize the output
        out_dict = {'width': int(0),
                    'height': int(0),
                    'filePos': []}

        # Set the regular expression used to parse each line of the stitching vector
        line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"

        # Get a list of all images in imagePath
        images = [p.name for p in Path(imagePath).iterdir()]

        # Open each stitching vector
        fpath = str(Path(stitchPath).absolute())
        name_pos = {}
        with open(fpath,'r') as fr:

            # Read the first line to get the filename for comparison to all other filenames
            line = fr.readline()
            stitch_groups = re.match(line_regex,line)
            stitch_groups = {key:val for key,val in zip(STITCH_VARS,stitch_groups.groups())}
            name = stitch_groups['file']
            name_ind = [i for i in range(len(name))]
            fr.seek(0) # reset to the first line

            # Read each line in the stitching vector
            for line in fr:
                # Read and parse values from the current line
                stitch_groups = re.match(line_regex,line)
                stitch_groups = {key:get_number(val) for key,val in zip(STITCH_VARS,stitch_groups.groups())}
                
                # If an image in the vector doesn't match an image in the collection, then skip it
                if stitch_groups['file'] not in images:
                    continue

                # Get the image size
                stitch_groups['width'], stitch_groups['height'] = BioReader.image_size(str(Path(imagePath).joinpath(stitch_groups['file']).absolute()))
                if out_dict['width'] < stitch_groups['width']+stitch_groups['posX']:
                    out_dict['width'] = stitch_groups['width']+stitch_groups['posX']
                if out_dict['height'] < stitch_groups['height']+stitch_groups['posY']:
                    out_dict['height'] = stitch_groups['height']+stitch_groups['posY']

                # Set the stitching vector values in the file dictionary
                out_dict['filePos'].append(stitch_groups)

        return out_dict
        
    def read_image(self,X,Y,Z,color=None):
        if X[0] >= self.X[0] and X[1] <= self.X[1]:
            if Y[0] >= self.Y[0] and Y[1] <= self.Y[1]:
                if Z[0] >= self.Z[0] and Z[1] <= self.Z[1]:
                    return self._image[Y[0]-self._Y_offset:Y[1]-self._Y_offset,
                                       X[0]-self._X_offset:X[1]-self._X_offset,...]
                else:
                    raise ValueError('Z must be [0,1]')
        
        x_min = 2**13 * (X[0]//2**13)
        x_max = min([x_min+2**13,self._file_dict['width']])
        y_min = 2**13 * (Y[0]//2**13)
        y_max = min([y_min+2**13,self._file_dict['height']])
        
        self._X_offset = x_min
        self._Y_offset = y_min
        
        self.make_tile(x_min,x_max,y_min,y_max,color)
        
        return self._image[Y[0]-self._Y_offset:Y[1]-self._Y_offset,
                           X[0]-self._X_offset:X[1]-self._X_offset,...]

def _avg2(image):
    """ Average pixels together with optical field 2x2 and stride 2
    
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    
    # The data fed into this is the same as the native file format.
    # We need to make sure the type will not cause overflow - NJS
    if image.dtype == np.uint8:
        dtype = np.uint16
    elif image.dtype == np.uint16:
        dtype = np.uint32
    elif image.dtype == np.uint32:
        dtype = np.uint64
    else:
        dtype = image.dtype
    
    odtype = image.dtype
    image = image.astype(dtype)
    imgshape = image.shape
    ypos = imgshape[0]
    xpos = imgshape[1]
    
    y_max = ypos - ypos % 2 # if odd then subtracting 1
    x_max = xpos - xpos % 2

    avg_imgshape = np.ceil([d/2 for d in imgshape]).astype(int)
    avg_imgshape[2] = 4 # Only deal with color images in color pyramid builder plugin
    avg_img = np.zeros(avg_imgshape,dtype=dtype)
    avg_img[0:int(y_max/2),0:int(x_max/2),:]= (\
                                                image[0:y_max-1:2,0:x_max-1:2,:] + \
                                                image[1:y_max:2  ,0:x_max-1:2,:] + \
                                                image[0:y_max-1:2,1:x_max:2  ,:] + \
                                                image[1:y_max:2  ,1:x_max:2  ,:])/4

    return avg_img.astype(odtype)

def _get_higher_res(S,bfio_reader,slide_writer,encoder,alpha,color=None,stitch=False,X=None,Y=None):
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
        bfio_reader - List of BioReader objects used to read the tiled tiffs
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
    
    # Channels designates color indices defining the following colors:
    # red, green, blue, yellow, cyan, magenta, gray
    # When creating the image, if the 3rd value in the bfio_reader list is
    # defined, then the image is defined by channels[2], or blue.
    channels = [[0,3],
                [1,3],
                [2,3],
                [0,1,3],
                [0,2,3],
                [1,2,3],
                [0,1,2,3]]
    
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    Z = [0,1]
        
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]

    # Initialize the output
    image = np.zeros((Y[1]-Y[0],X[1]-X[0],4),dtype=np.uint8)
    if not alpha:
        image[:,:,3] = 255
    
    # If requesting from the lowest scale, then just read the images
    if str(S)==encoder.info['scales'][0]['key']:
        for ind,br in enumerate(bfio_reader):
            if br == None:
                continue
            if isinstance(br,BioAssembler):
                br.color = channels[ind]
                image_color_temp = br.read_image(X,Y,Z,color).astype(np.uint8)
            else:
                image_temp = (255*(br.read_image(X=X,Y=Y,Z=Z)[...,0,0].astype(np.float32) - br.bounds[0])/(br.bounds[1] - br.bounds[0]))
                image_temp[image_temp>255] = 255
                image_temp[image_temp<0] = 0
                image_temp = image_temp.astype(np.uint8)
                image_color_temp = copy.deepcopy(image)
                image_color_temp[:,:,channels[ind]] = image_temp
                del image_temp
            image = np.maximum(image,image_color_temp)

    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]],[0,1]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)
                
        def load_and_scale(*args,**kwargs):
            sub_image = _get_higher_res(**kwargs)
            image = args[0]
            x_ind = args[1]
            y_ind = args[2]
            image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],0:4] = _avg2(sub_image)
        
        if (S % 2 == 0 or str(S+1)==encoder.info['scales'][0]['key']) and not stitch:
            with ThreadPoolExecutor() as executor:
                for y in range(0,len(subgrid_dims[1])-1):
                    y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                    y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                    for x in range(0,len(subgrid_dims[0])-1):
                        x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                        x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                        executor.submit(load_and_scale,
                                        image,x_ind,y_ind, # args
                                        alpha=alpha,       # kwargs
                                        X=subgrid_dims[0][x:x+2],
                                        Y=subgrid_dims[1][y:y+2],
                                        S=S+1,
                                        bfio_reader=bfio_reader,
                                        slide_writer=slide_writer,
                                        encoder=encoder)
        else:
            for y in range(0,len(subgrid_dims[1])-1):
                y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                for x in range(0,len(subgrid_dims[0])-1):
                    x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                    x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                    load_and_scale(image,x_ind,y_ind, # args
                                   alpha=alpha,       # kwargs
                                   X=subgrid_dims[0][x:x+2],
                                   Y=subgrid_dims[1][y:y+2],
                                   S=S+1,
                                   bfio_reader=bfio_reader,
                                   slide_writer=slide_writer,
                                   encoder=encoder,
                                   color=color,
                                   stitch=stitch)

    # Encode the chunk
    image_encoded = encoder.encode(image)
    
    # Write the chunk
    slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],0,1))
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

class DeepZoomChunkEncoder(NeuroglancerChunkEncoder):

    # Data types used by Neuroglancer
    DATA_TYPES = ("uint8", "uint16", "uint32", "uint64", "float32")

    def __init__(self, info):
        """ Properly formats numpy array for DeepZoom pyramid.
    
        Inputs:
            info - info dictionary
        """
        
        super().__init__(info)

    def encode(self, chunk):
        """ Squeeze the input array.
        Inputs:
            chunk - array with four dimensions (C, Z, Y, X)
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Check to make sure the data is formatted properly
        assert chunk.ndim == 3
        return chunk

def bfio_metadata_to_slide_info(bfio_reader,outPath):
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
    sizes = [bfio_reader.num_x(),bfio_reader.num_y(),bfio_reader.num_z()]
    phys_x = bfio_reader.physical_size_x()
    if None in phys_x:
        phys_x = (1000,'nm')
    phys_y = bfio_reader.physical_size_y()
    if None in phys_y:
        phys_y = (1000,'nm')
    resolution = [phys_x[0] * UNITS[phys_x[1]]]
    resolution.append(phys_y[0] * UNITS[phys_y[1]])
    resolution.append((phys_y[0] * UNITS[phys_y[1]] + phys_x[0] * UNITS[phys_x[1]])/2) # Just used as a placeholder
    dtype = bfio_reader.read_image(X=[0,1024],Y=[0,1024],Z=[0,1]).dtype
    
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
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),1]
        current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
        info['scales'].append(current_scale)
    
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
