from bfio.bfio import BioReader
import numpy as np
import json, copy, os
from pathlib import Path
import imageio
import filepattern
import os
import math
import time
import pandas as pd
from collections import Counter
# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 1024

def findRepeating(arr, n): 
  
    missingElement = 0
  
    # indexing based 
    for i in range(0, n): 
  
        element = arr[abs(arr[i])] 
  
        if(element < 0): 
            missingElement = arr[i] 
            break
          
        arr[abs(arr[i])] = -arr[abs(arr[i])] 
      
    return abs(missingElement) 

def modecalc(w, x, y, z):

    uniques = set([w, x, y, z])
    num_uniques = len(uniques)

    # if num_uniques == 1:
    #     return x
    # if num_uniques == 4:
    #     return max(w, x, y, z)

    if w == x:
        if y == z:
            return max(x,y)
        else:
            return x
    else:
        if y == z:
            return y
        else:
            a = min(x,w)
            b = max(x,w)
            x = min(y,z)
            y = max(y,z)
            if b == y:
                return b
            elif b == x:
                return x
            elif x == a:
                return a
            else:
                return max(b, y)

    # unique = set([w, x, y, z])
    # lenunique = len(unique)

    # if lenunique == 1:
    #     return x
    # elif lenunique == 4:
    #     return max(unique)
    # elif lenunique == 2:


    #     return findRepeating([w, x, y, z], n=4)
    #     # if w == x:
    #     #     return x
    #     # else:
    #     #     if y == z:
    #     #         return y
    #     #     else:
    #     #         A = min(x,w)
    #     #         B = max(x,w)
    #     #         X = min(y,z)
    #     #         Y = max(y,z)
    #     #         if B == Y:
    #     #             return B
    #     #         elif B == X:
    #     #             return X
    #     #         elif X == A:
    #     #             return A
    #     #         else:
    #     #             return max(B, Y)

def segmentinfo(encoder,idlabels,out_dir):

    op = Path(out_dir).joinpath("infodir")
    # op = Path(encoder.info["segment_properties"])
    op.mkdir()
    op = op.joinpath("info")

    inlineinfo = {
        "ids":[str(item) for item in idlabels],
        "properties":[
            {
            "id":"label",
            "type":"label",
            "values":[str(item) for item in idlabels]
            },
            {
            "id":"description",
            "type":"label",
            "values": [str(item) for item in idlabels]
            }
        ]
    }

    info = {
        "@type": "neuroglancer_segment_properties",
        "inline": inlineinfo
    }

    with open(op,'w') as writer:
        writer.write(json.dumps(info))
    writer.close()

    return op



def squeeze_generic(a, axes_to_keep):
    " Reduces the number of dimensions of an array to the number specified"
    out_s = [s for i,s in enumerate(a.shape) if i in axes_to_keep or s!=1]
    return a.reshape(out_s)

def _mode2(image, dtype):
    """ Average pixels together with optical field 2x2 and stride 2
    
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """

    """ Find mode of pixels in optical field 2x2 and stride 2
    This method works by finding the largest number that occurs at least twice
    in a 2x2 grid of pixels, then sets that value to the output pixel.
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        mode_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    def forloop(mode_img, idxfalse, vals):

        for i in range(3):
            rvals = vals[i]
            for j in range(i+1,4):
                cvals = vals[j]
                ind = np.logical_and(cvals==rvals,rvals>mode_img[idxfalse])
                mode_img[idxfalse][ind] = rvals[ind]
        return mode_img

    imgshape = image.shape
    ypos, xpos, zpos = imgshape

    y_max = ypos - ypos % 2
    x_max = xpos - xpos % 2
    z_max = zpos - zpos % 2

    mode_imgshape = np.ceil([d/2 for d in imgshape]).astype('int')
    mode_img = np.zeros(mode_imgshape).astype(dtype)

    vals00 = image[0:y_max-1:2, 0:x_max-1:2,:]
    vals01 = image[0:y_max-1:2, 1:x_max:2,:]
    vals10 = image[1:y_max:2,   0:x_max-1:2,:]
    vals11 = image[1:y_max:2,   1:x_max:2,:]

    index = (vals00 == vals01) & (vals10 == vals11)
    maxarray = np.maximum(vals00, vals10)
    indexfalse = index==False
    indextrue = index==True
    valueslist = [vals00[indexfalse], vals01[indexfalse], vals10[indexfalse], vals11[indexfalse]]

    if ypos != y_max and xpos == x_max:
        shortmode_img = mode_img[:-1,:,:]
        shortmode_img[indextrue] = maxarray[indextrue]
        shortmode_img = forloop(shortmode_img, indexfalse, valueslist)
        mode_img[:-1,:,:] = shortmode_img
    elif xpos != x_max and ypos == y_max:
        shortmode_img = mode_img[:,:-1,:]
        shortmode_img[indextrue] = maxarray[indextrue]
        shortmode_img = forloop(shortmode_img, indexfalse, valueslist)
        mode_img[:,:-1,:] = shortmode_img
    elif xpos != x_max and ypos != y_max:
        shortmode_img = mode_img[:-1,:-1,:]
        shortmode_img[indextrue] = maxarray[indextrue]
        shortmode_img = forloop(shortmode_img, indexfalse, valueslist)
        mode_img[:-1,:-1,:] = shortmode_img
    else:
        mode_img[indextrue] = maxarray[indextrue]
        mode_img = forloop(mode_img, indexfalse, valueslist)

    return mode_img


def _modetwo(image, dtype):
    """ Average pixels together with optical field 2x2 and stride 2
    
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """

    """ Find mode of pixels in optical field 2x2 and stride 2
    This method works by finding the largest number that occurs at least twice
    in a 2x2 grid of pixels, then sets that value to the output pixel.
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        mode_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    imgshape = image.shape

    ypos = imgshape[0]
    xpos = imgshape[1]
    zpos = imgshape[2]
    y_max = ypos - ypos % 2
    x_max = xpos - xpos % 2
    z_max = zpos - zpos % 2

    mode_imgshape = np.ceil([d/2 for d in imgshape]).astype('int')
    mode_img = np.zeros(mode_imgshape)
    for i in range(3):
        x1 = i//2
        y1 = i%2
        rvals = image[0:y_max+(ypos-y_max)+y1:2,0:x_max+(xpos-x_max)+x1:2,:] # reference values
        for j in range(i+1,4):
            x2 = j//2
            y2 = j%2
            cvals = image[0:y_max+(ypos-y_max)+y2:2,0:x_max+(xpos-x_max)+x2:2,:] # compare values
            ind = np.logical_and(cvals==rvals,rvals>mode_img)
            mode_img[ind] = rvals[ind]


    return mode_img
    
    # print(" ")


    # yblack, xblack= np.where((image[:, :, :] > [0,0,0]).all(2))

    # yblack = yblack - (yblack%2)
    # xblack = xblack = (xblack%2)
    # zblack = np.zeros(len(xblack)).astype('int')
    
    # sets = np.array([yblack, xblack, zblack])
    # sets = sets.transpose()

    # try:
    #     unique = np.unique(sets, axis=0)
    # except:
    #     return mode_img

    # unique = unique.transpose()

    # yblack = unique[0]
    # xblack = unique[1]
    # zblack = unique[2]
    
    # halfy = (yblack/2).astype('int')
    # halfx = (xblack/2).astype('int')

    # rvals = np.zeros(np.ceil(mode_imgshape).astype('int'))
    # cvals = np.zeros(np.ceil(mode_imgshape).astype('int'))
    # ind = np.zeros(np.ceil(mode_imgshape).astype('int'))

    # for i in range(3):
    #     x1 = i//2
    #     y1 = i%2
    #     rvals[halfy, halfx, zblack]= image[yblack+y1, xblack+x1, zblack]
    #     # rvals = image[0:y_max+(ypos-y_max)+y1:2,0:x_max+(xpos-x_max)+x1:2,:] # reference values
    #     # rvals = image[yblack[i] - (yblack[i]%2)+y1,xblack[i] - (xblack[i]%2)+x1,:]  for i in range(num_blacks)]
    #     for j in range(i+1,4):
    #         x2 = j//2
    #         y2 = j%2
    #         cvals[halfy, halfx, zblack] = image[yblack+y2, xblack+x2, zblack]
    #         # cvals = image[0:y_max+(ypos-y_max)+y2:2,0:x_max+(xpos-x_max)+x2:2,:] # compare values
    #         # cvals = [image[yblack[i] - (yblack[i]%2)+y2,xblack[i] - (xblack[i]%2)+x2,:]  for i in range(num_blacks)]
    #         # calculate local mode
    #         ind = np.logical_and(cvals==rvals,rvals>mode_img)
    #         mode_img[ind] = rvals[ind]

    # print(" ")
    # return mode_img
    
    # image = image.astype('uint16')
    # imgshape = image.shape
    # ypos = imgshape[0]
    # xpos = imgshape[1]
    # zpos = imgshape[2]
    # z_max = zpos - zpos % 2    # if even then subtracting 0. 
    # y_max = ypos - ypos % 2 # if odd then subtracting 1
    # x_max = xpos - xpos % 2

    # mode_imgshape = np.ceil([d/2 for d in imgshape]).astype('int')
    # mode_img = np.zeros(mode_imgshape)

    # one = image[0:y_max-1:2,0:x_max-1:2,:]
    # two = image[1:y_max:2  ,0:x_max-1:2,:]
    # three = image[0:y_max-1:2,1:x_max:2  ,:]
    # four = image[1:y_max:2  ,1:x_max:2  ,:]

    # ogshape = one.shape

    # modevector = np.vectorize(modecalc)
    # modes = modevector(one, two, three, four)
    
    # mode_img[0:int(y_max/2),0:int(x_max/2),:]= modes

def _avg2(image):
    """ Average pixels together with optical field 2x2 and stride 2
    
    Inputs:
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
    """
    
    image = image.astype('uint16')
    imgshape = image.shape
    ypos = imgshape[0]
    xpos = imgshape[1]
    zpos = imgshape[2]
    z_max = zpos - zpos % 2    # if even then subtracting 0. 
    y_max = ypos - ypos % 2 # if odd then subtracting 1
    x_max = xpos - xpos % 2

    avg_imgshape = [d/2 for d in imgshape]
    avg_img = np.zeros(np.ceil(avg_imgshape).astype('int')).astype('uint16')
    avg_img[0:int(y_max/2),0:int(x_max/2),:]= (\
                                                image[0:y_max-1:2,0:x_max-1:2,:] + \
                                                image[1:y_max:2  ,0:x_max-1:2,:] + \
                                                image[0:y_max-1:2,1:x_max:2  ,:] + \
                                                image[1:y_max:2  ,1:x_max:2  ,:])/4


    return avg_img

def _get_higher_res(S, zlevel, bfio_reader,slide_writer,encoder,imageType, X=None,Y=None,slices=None):
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
    Z = [0,1]
        
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]

    datatype = bfio_reader.read_metadata().image().Pixels.get_PixelType()
    # Initialize the output
    image = np.zeros((Y[1]-Y[0],X[1]-X[0],1),dtype=datatype)
    
    # If requesting from the lowest scale, then just read the image
    if str(S)==encoder.info['scales'][0]['key']:
        if slices == None:
            image = bfio_reader.read_image(X=X,Y=Y,Z=Z)
            image = squeeze_generic(image, axes_to_keep=(0, 1, 2))
        else:
            image = bfio_reader.read_image(X=X,Y=Y,Z=slices)
            image = squeeze_generic(image, axes_to_keep=(0, 1, 2))
    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]],[0,1]]
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
                                            imageType=imageType,
                                            bfio_reader=bfio_reader,
                                            slide_writer=slide_writer,
                                            encoder=encoder,
                                            slices=slices)
                start_time = time.time()
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],0:1] = _avg2(sub_image)
                end_avgtime = time.time()
                image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],0:1] = _mode2(sub_image, datatype)
                end_modetime = time.time()
                # print("THE TIME IT TAKES TO DO AVERAGING", end_avgtime-start_time)
                # print("THE TIME IT TAKES TO DO MODE FUNCTION", end_modetime-end_avgtime)
                # print("THE MODE FUNCTION IS ", (end_modetime-end_avgtime)/(end_avgtime-start_time), "SLOWER")
                # if ((end_modetime-end_avgtime)/(end_avgtime-start_time)) > 5:
                #     print("too slow", ((end_modetime-end_avgtime)/(end_avgtime-start_time)))


    # Encode the chunk
    image_encoded = encoder.encode(image)
    # Write the chunk
    slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],zlevel,zlevel + 1))
    print(" ")
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

def bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight, imagetype):
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
    # phys_x = bfio_reader.physical_size_x()
    # if None in phys_x:
    phys_x = (325,'nm')
    # phys_y = bfio_reader.physical_size_y()
    # if None in phys_y:
    phys_y = (325,'nm')
    # phys_z = bfio_reader.physical_size_z()
    # if None in phys_z:
    phys_z = (325,'nm')
    resolution = [phys_x[0] * UNITS[phys_x[1]]]
    resolution.append(phys_y[0] * UNITS[phys_y[1]])
    resolution.append(phys_z[0] * UNITS[phys_z[1]]) # Just used as a placeholder
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
    if imagetype == "segmentation":
        info = {
            "data_type": dtype,
            "num_channels":1,
            "scales": [scales],       # Will build scales below
            "type": imagetype,
            "segment_properties": "infodir"
        }
    else:
        info = {
            "data_type": dtype,
            "num_channels":1,
            "scales": [scales],       # Will build scales below
            "type": imagetype,
        }

    for i in range(1,num_scales+1):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(num_scales - i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),stackheight]
        current_scale['resolution'] = [2*previous_scale['resolution'][0],2*previous_scale['resolution'][1],previous_scale['resolution'][2]]
        info['scales'].append(current_scale)
    
    return info

def neuroglancer_info_file(bfio_reader,outPath, stackheight, imagetype):
    # Create an output path object for the info file
    op = Path(outPath).joinpath("info")
    
    # Get pyramid info
    info = bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight, imagetype)

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
