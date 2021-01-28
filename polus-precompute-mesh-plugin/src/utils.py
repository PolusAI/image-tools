from bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import numpy as np
import json, copy, os
import simplejson as json
from pathlib import Path
import imageio
import filepattern
# import os
import logging
# import math
from concurrent.futures import ThreadPoolExecutor
import shutil
import threading
from concurrent import futures
import struct,json
from numpy.linalg import matrix_rank
import collections
from skimage import measure
import traceback
from json import JSONEncoder
import trimesh
import scalable_multires
import ast

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utilities")
logger.setLevel(logging.INFO)

# Conversion factors to nm, these are based off of supported Bioformats length units
UNITS = {'m':  10**9,
         'cm': 10**7,
         'mm': 10**6,
         'µm': 10**3,
         'nm': 1,
         'Å':  10**-1}

# Chunk Scale
CHUNK_SIZE = 256

def find_ids(image):
    """ This function unravels the image into a 1D numpy array and looks for any unique labels """
    mesh = image.ravel()
    ids = [int(i) for i in np.unique(mesh[:])]
    return ids

def progressive_meshes(ide, chunkfiles, temp_dir, out_dir, bit_depth):
    logger.info('Starting Progressive Meshes for ID {}'.format(ide))

    # Identify all the files belonging to the same segment 
    idenfiles = [f for f in chunkfiles if f.split('_')[0] == str(ide)]
    len_files = len(idenfiles)
    logger.info('ID {} is scattered amoung {} chunk(s)'.format(str(ide), len_files))
    
    # Identify which tile each chunk belongs to
    stripped_files = [i.strip('.ply').split('_')[1:] for i in idenfiles]

    # Identify the tile closest to the origin
    starts = []
    for fil in range(len_files):
        start = [ast.literal_eval(trans)[0] for trans in stripped_files[fil]]
        starts.append(start)
    start_mesh = min(starts)
    mesh_index = starts.index(start_mesh)
    mesh_fileobj = idenfiles.pop(mesh_index)

    # Load the mesh that is closest to the origin
    mesh1_path = str(Path(temp_dir).joinpath(mesh_fileobj))
    mesh1 = trimesh.load_mesh(file_obj=mesh1_path, file_type='ply')

    # Translate the mesh to its respective spot.  If starting_mesh = [0,0,0], then there is no translation
    translate_start = ([1, 0, 0, start_mesh[1]],
                    [0, 1, 0, start_mesh[0]],
                    [0, 0, 1, start_mesh[2]],
                    [0, 0, 0, 1])
    mesh1.apply_transform(translate_start)
    logger.info('(ID: {}) Loaded chunk 1/{}: {} ---- {} bytes'.format(ide, len_files, mesh_fileobj, os.path.getsize(mesh1_path)))

    
    if len_files == 1: # If segment belongs to only one tile, then create progressive mesh of that segment
        scalable_multires.generate_multires_mesh(mesh=mesh1,
                                                directory=str(out_dir),
                                                segment_id=ide,
                                                quantization_bits=bit_depth)
    else: # Else concatenate other pieces of the segment

        # Need to recalculate because the first mesh has been removed from list and is already loaded
        stripped_files_middle = [idy.strip('.ply').split('_')[1:] for idy in idenfiles]

        for i in range(len_files-1): # For each of the remaining chunks of the segment:
            mesh2_path = str(Path(temp_dir).joinpath(idenfiles[i]))
            mesh2 = trimesh.load_mesh(file_obj=mesh2_path, file_type='ply') # Load the chunk
            logger.info('(ID: {}) Loaded chunk {}/{}: {} ---- {} bytes'.format(ide, i+2, len_files, idenfiles[i], os.path.getsize(mesh2_path)))

            # Translate the chunk to its respective spot.  
            transformationmatrix = [ast.literal_eval(trans) for trans in stripped_files_middle[i]]
            offset = [transformationmatrix[i][0]/CHUNK_SIZE for i in range(3)] # With every chunk away from the origin, there is an offset of 1z
            middle_mesh = [trans[0] for trans in transformationmatrix] 
            translate_middle = ([1, 0, 0, middle_mesh[1] - offset[1]],
                                [0, 1, 0, middle_mesh[0] - offset[0]],
                                [0, 0, 1, middle_mesh[2] - offset[2]],
                                [0, 0, 0, 1])
            mesh2.apply_transform(translate_middle) # Concatenate mesh2 to mesh1
            mesh1 = trimesh.util.concatenate(mesh1, mesh2) # mesh1 is now bigger, since its been concatenated to another chunk of the segmented mesh.
        
        # Once we concatenate all the pieces of the segmented mesh, then we can continue to create progressive meshes and convert to Draco file format. 
        scalable_multires.generate_multires_mesh(mesh=mesh1,
                                                directory=str(out_dir),
                                                segment_id=ide,
                                                quantization_bits=bit_depth)

def meshdata(volume, ids, outDir_mesh, X, Y, Z):
    """ This function generates a temporary directory of polygon meshes that are chunked.
        The polygon meshes are chunked for scalability """
    for iden in ids:
        if iden == 0:
            continue
        logger.info('Processing label ID {} in section ({}, {}, {})'.format(iden, X, Y, Z))
        dtype = volume.dtype

        # use the marching cube algorithm to create most detailed mesh
        vertices,faces,_,_ = measure.marching_cubes((volume==iden).astype(str(np.iinfo(dtype).dtype)), level=0, step_size=1)

        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        dimensions = root_mesh.bounds
        scalable_multires.generate_trimesh_chunks(mesh=root_mesh,
                                            directory=str(outDir_mesh),
                                            segment_id=iden,
                                            chunks=(X, Y, Z))
def meshdir_files(opmeshdir, encoder):
    """ This function creates an additional info file that is required by Neuroglancer to view meshes. """
    transform_vars = encoder.info['scales'][0]['resolution']
    
    opmesh_mesh = opmeshdir.joinpath("info")
    infomesh = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": 16, #should be 10 or 16
        "transform": [0, transform_vars[0], 0, 0, 
                      transform_vars[1], 0, 0, 0, 
                      0, 0, transform_vars[2], 0], 
        "lod_scale_multiplier": "1"
    }

    with open(opmesh_mesh, 'w') as writemesh:
        json.dump(infomesh, writemesh)
    writemesh.close()

def infodir_files(encoder,idlabels,out_dir):
    """ This function creates an additional info file that is required by Neuroglancer to view segmentations. """

    opinfodir = Path(out_dir).joinpath("infodir")
    opinfodir.mkdir()
    opinfo_info = opinfodir.joinpath("info")
    
    # subsection of the info file
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

    with open(opinfo_info,'w') as writer:
        writer.write(json.dumps(info))
    writer.close()


def _mode2(image, dtype):
    """ Find mode of pixels in optical field 2x2 and stride 2
    This method works by finding the largest number that occurs at least twice
    in a 2x2x2 grid of pixels, then sets that value to the output pixel.
    Inputs:
        image - numpy array with three dimensions (m,n,p)
    Outputs:
        mode_img - numpy array with only two dimensions (round(m/2),round(n/2),round(p/2))
    """
    def forloop(mode_img, idxfalse, vals):

        for i in range(7):
            rvals = vals[i]
            for j in range(i+1,8):
                cvals = vals[j]
                ind = np.logical_and(cvals==rvals,rvals>mode_img[idxfalse])
                mode_img[idxfalse][ind] = rvals[ind]
        return mode_img

    imgshape = image.shape
    ypos, xpos, zpos = imgshape

    y_edge = ypos % 2
    x_edge = xpos % 2
    z_edge = zpos % 2

    # Initialize the mode output image (Half the size)
    mode_imgshape = np.ceil([d/2 for d in imgshape]).astype('int')
    mode_img = np.zeros(mode_imgshape).astype('uint16')

    # Garnering the eight different pixels that we would find the modes of
    # Finding the mode of: 
    # vals000[1], vals010[1], vals100[1], vals110[1], vals001[1], vals011[1], vals101[1], vals111[1]
    # vals000[2], vals010[2], vals100[2], vals110[2], vals001[2], vals011[2], vals101[2], vals111[2]
    # etc 
    vals000 = image[0:-1:2, 0:-1:2,0:-1:2]
    vals010 = image[0:-1:2, 1::2,0:-1:2]
    vals100 = image[1::2,   0:-1:2,0:-1:2]
    vals110 = image[1::2,   1::2,0:-1:2]
    vals001 = image[0:-1:2, 0:-1:2,1::2]
    vals011 = image[0:-1:2, 1::2,1::2]
    vals101 = image[1::2,   0:-1:2,1::2]
    vals111 = image[1::2,   1::2,1::2]

    # Finding all quadrants where at least two of the pixels are the same
    index = ((vals000 == vals010) & (vals000 == vals100)) & (vals000 == vals110) 
    indexfalse = index==False
    indextrue = index==True

    # Going to loop through the indexes where the two pixels are not the same
    valueslist = [vals000[indexfalse], vals010[indexfalse], vals100[indexfalse], vals110[indexfalse], vals001[indexfalse], vals011[indexfalse], vals101[indexfalse], vals111[indexfalse]]
    edges = (y_edge,x_edge,z_edge)

    mode_edges = {
        (0,0,0): mode_img[:, :, :],
        (0,1,0): mode_img[:,:-1,:],
        (1,0,0): mode_img[:-1,:,:],
        (1,1,0): mode_img[:-1,:-1,:],
        (0,0,1): mode_img[:,:,:-1],
        (0,1,1): mode_img[:,:-1,:-1],
        (1,0,1): mode_img[:-1,:,:-1],
        (1,1,1): mode_img[:-1, :-1, :-1]
    }
    # Edge cases, if there are an odd number of pixels in a row or column, then we ignore the last row or column
    # Those columns will be black

    if edges == (0,0,0):
        mode_img[indextrue] = vals000[indextrue]
        mode_img = forloop(mode_img, indexfalse, valueslist)
        return mode_img
    else:
        shortmode_img = mode_edges[edges]
        shortmode_img[indextrue] = vals000[indextrue]
        shortmode_img = forloop(shortmode_img, indexfalse, valueslist)
        mode_edges[edges] = shortmode_img
        return mode_edges[edges]


def _get_higher_res(S, bfio_reader,slide_writer,encoder,ids, meshes, imagetype, outDir_mesh, X=None,Y=None,Z=None):
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

    voxel_size = np.float32(encoder.info['scales'][0]['resolution'])

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
    if Z == None:
        Z = [0,scale_info['size'][2]] #[0, stackheight]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    if Z[1] > scale_info['size'][2]:
        Z[1] = scale_info['size'][2]

    # Initialize the output
    datatype = bfio_reader.dtype
    image = np.zeros((Y[1]-Y[0],X[1]-X[0],Z[1]-Z[0]),dtype=datatype)

    # If requesting from the lowest scale, then just read the image
    if str(S)==encoder.info['scales'][0]['key']:
        image = bfio_reader[Y[0]:Y[1],X[0]:X[1],Z[0]:Z[1],0,0]
        if imagetype == "segmentation":
            id_in_chunk = find_ids(image)

            if meshes:
                try:
                    #creates polygon meshes
                    meshdata(image.squeeze(), id_in_chunk, outDir_mesh, X, Y, Z)
                except Exception as e:
                    traceback.print_exc()

            # Want only unique ids
            if len(ids) == 0: 
                ids.extend(id_in_chunk)
            else: 
                ids.extend(set(id_in_chunk) - set(ids))
                ids.sort() 

        # Encode the chunk
        image_encoded = encoder.encode(image, bfio_reader.z)

        # Write the chunk
        slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],Z[0],Z[1]))
        logger.info('Scale ({}): {}-{}_{}-{}_{}-{}'.format(S, str(X[0]), str(X[1]), str(Y[0]),str(Y[1]), str(Z[0]), str(Z[1])))
        return image

    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]],[2*Z[0],2*Z[1]]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)

        def load_and_scale(*args,**kwargs):
            jutil.attach()
            sub_image = _get_higher_res(**kwargs)
            jutil.detach()
            image = args[0]
            x_ind = args[1]
            y_ind = args[2]
            z_ind = args[3]
            image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],z_ind[0]:z_ind[1]] = _mode2(sub_image, datatype)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for x in range(0,len(subgrid_dims[1])-1):
                x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                for y in range(0,len(subgrid_dims[0])-1):
                    y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                    y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                    for z in range(0, len(subgrid_dims[2])-1):
                        z_ind = [subgrid_dims[2][z] - subgrid_dims[2][0],subgrid_dims[2][z+1] - subgrid_dims[2][0]]
                        z_ind = [np.ceil(zi/2).astype('int') for zi in z_ind]
                        
                        executor.submit(load_and_scale, 
                                            image, x_ind, y_ind, z_ind, 
                                            X=subgrid_dims[0][x:x+2],
                                            Y=subgrid_dims[1][y:y+2],
                                            Z=subgrid_dims[2][z:z+2],
                                            S=S+1,
                                            bfio_reader=bfio_reader,
                                            slide_writer=slide_writer,
                                            encoder=encoder,
                                            ids=ids,
                                            meshes=meshes,
                                            imagetype=imagetype,
                                            outDir_mesh=outDir_mesh)

        # Encode the chunk
        image_encoded = encoder.encode(image, image.shape[2])
        slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],Z[0],Z[1]))
        logger.info('Scale ({}): {}-{}_{}-{}_{}-{}'.format(S, str(X[0]), str(X[1]), str(Y[0]),str(Y[1]), str(Z[0]), str(Z[1])))
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
        f.close()

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

    def encode(self, chunk, stackheight):
        """ Encode a chunk from a Numpy array into bytes.
        Inputs:
            chunk - array with four dimensions (C, Z, Y, X)
        Outputs:
            buf - encoded chunk (byte stream)
        """
        # Rearrange the image for Neuroglancer
        chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],chunk.shape[2],1),
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

def bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight,imagetype):
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
    sizes = [bfio_reader.x,bfio_reader.y,stackheight]
    phys_x = bfio_reader.physical_size_x
    if None in phys_x:
        phys_x = (325,'nm')
    phys_y = bfio_reader.physical_size_y
    if None in phys_y:
        phys_y = (325,'nm')
    phys_z = bfio_reader.physical_size_z
    if None in phys_z:
        phys_z = ((phys_y[0] * UNITS[phys_y[1]] + phys_x[0] * UNITS[phys_x[1]])/2, 'nm')
    # phys_x = (325,'nm')
    # phys_y = (325,'nm')
    # phys_z = (325,'nm')
    resolution = [phys_x[0] * UNITS[phys_x[1]]]
    resolution.append(phys_y[0] * UNITS[phys_y[1]])
    resolution.append(phys_z[0] * UNITS[phys_z[1]])
    dtype = bfio_reader.dtype
    
    num_scales = int(np.log2(max(sizes))) + 1
    
    # create a scales template, use the full resolution8
    scales = {
        "chunk_sizes":[[CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE]],
        "encoding": "raw",
        "key": str(num_scales),
        "resolution":resolution,
        "size":sizes,
        "voxel_offset":[0,0,0]
    }
    info = {}
    
    if imagetype == "segmentation":
        # initialize the json dictionary
        info = {
            "@type": "neuroglancer_multiscale_volume",
            "data_type": str(np.iinfo(dtype).dtype),
            "num_channels":1,
            "scales": [scales],       # Will build scales below
            "type": imagetype,
            "segment_properties": "infodir",
            "mesh": "meshdir"
        }
    else:
        info = {
            "@type": "neuroglancer_multiscale_volume",
            "data_type": str(np.iinfo(dtype).dtype),
            "num_channels":1,
            "scales": [scales],       # Will build scales below
            "type": imagetype
        }
    
    for i in range(1,num_scales+1):
        previous_scale = info['scales'][-1]
        current_scale = copy.deepcopy(previous_scale)
        current_scale['key'] = str(num_scales - i)
        current_scale['size'] = [int(np.ceil(previous_scale['size'][0]/2)),int(np.ceil(previous_scale['size'][1]/2)),int(np.ceil(previous_scale['size'][2]/2))]
        for i in range(0,3):
            if current_scale['size'][i] == previous_scale['size'][i]:
                current_scale['resolution'][i] = previous_scale['resolution'][i]
            else:
                current_scale['resolution'][i] = 2*previous_scale['resolution'][i]
        info['scales'].append(current_scale)
    
    return info

def neuroglancer_info_file(bfio_reader,outPath, stackheight, imagetype, meshes):
    """ This function generates the necessary info files for the data to be viewed in Neuroglancer"""

    # Create an output path object for the info file
    op = Path(outPath).joinpath("info")
    # Get pyramid info
    info = bfio_metadata_to_slide_info(bfio_reader,outPath,stackheight,imagetype)
    
    # Write the neuroglancer info file
    with open(op,'w') as writer:
        json.dump(info, writer)
    writer.close()
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
    writer.close()
        
    return info