from typing import Tuple
import numpy as np
import json, copy, os
import math

import logging, traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import tempfile
from numpy.lib.arraysetops import unique

import trimesh
from skimage import measure

from neurogen import mesh as ngmesh
from neurogen import info as nginfo
from neurogen import volume as ngvol


from itertools import repeat
from itertools import product

import bfio
from bfio import BioReader, BioWriter

import traceback

from PIL import Image

CHUNK_SIZE = 256
chunk_size = [CHUNK_SIZE,CHUNK_SIZE,CHUNK_SIZE]

MESH_CHUNK_SIZE = 512
mesh_chunk_size = [MESH_CHUNK_SIZE, MESH_CHUNK_SIZE, MESH_CHUNK_SIZE]


bit_depth = 10

get_dim1dim2 = lambda dimension1, dimension_size, rng_size: \
                    (int(dimension1), int(min(dimension1+rng_size, dimension_size)))

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def get_resolution(phys_y : tuple,
                   phys_x : tuple,
                   phys_z : tuple):
    
    """ This function generates a resolution in nanometers (nm)
    
    Args:
        phys_y : Actual y dimension of input
        phys_x : Actual x dimension of input
        phys_z : Actual z dimension of input
    
    Returns: 
        resolution : A list of integer values of resolution in nanometers in [Y, X, Z] order
                     If Y and X resolutions are none, then default to 325 nm
                     If Z resolution is none, then defaults to the average of Y and X
    """
    # Conversion factors to nm, these are based off of supported Bioformats length units
    UNITS = {'m':  10**9,
            'cm': 10**7,
            'mm': 10**6,
            'µm': 10**3,
            'nm': 1,
            'Å':  10**-1}

    if None in phys_y:
        phys_y = 325
    else:
        phys_y = phys_y[0] * UNITS[phys_y[1]]
    if None in phys_x:
        phys_x = 325
    else:
        phys_x = phys_x[0] * UNITS[phys_x[1]]
    if None in phys_z:
        phys_z = (phys_x + phys_y)/2
    else:
        phys_z = phys_z[0] * UNITS[phys_z[1]]
    
    return [phys_y, phys_x, phys_z]


def save_resolution(output_directory: str,
                    xyz_volume: tuple):
    """This function encodes a chunked volume.
    
    Args:
        output_directory : where the encoded image gets saved to
        xyz_volume : (xyz, volume)
            xyz : contains the dimensions of the chunk
            volume : the volume that gets encoded
    """
    try:
        xyz, volume = xyz_volume
        x1_chunk, x2_chunk = xyz[0]
        y1_chunk, y2_chunk = xyz[1]
        z1_chunk, z2_chunk = xyz[2] 
        volume = np.reshape(volume, volume.shape[:3])
        logger.debug("({0:0>4}, {0:0>4}), ".format(x1_chunk, x2_chunk) + \
                    "({0:0>4}, {0:0>4}), ".format(y1_chunk, y2_chunk) + \
                    "({0:0>4}, {0:0>4})".format(z1_chunk, z2_chunk))
        volume_encoded = ngvol.encode_volume(volume)
        ngvol.write_image(image=volume_encoded, volume_directory=output_directory, 
                y=(y1_chunk, y2_chunk),
                x=(x1_chunk, x2_chunk),  
                z=(z1_chunk, z2_chunk))
    except Exception as e:
        print(e)

def iterate_cache_tiles(bf_image: bfio.bfio.BioReader):

    """ This function iterates through the bfio object
        tiles and caches the information for easy access."""
    
    cache_tile = bf_image._TILE_SIZE
    for x1_cache in range(0, bf_image.X, cache_tile):
        for y1_cache in range(0, bf_image.Y, cache_tile):
            for z1_cache in range(0, bf_image.Z, cache_tile):
                for c1_cache in range(0, bf_image.C, cache_tile):
                    for t1_cache in range(0, bf_image.T, cache_tile):

                        y1_cache, y2_cache = get_dim1dim2(y1_cache, bf_image.Y, cache_tile)
                        x1_cache, x2_cache = get_dim1dim2(x1_cache, bf_image.X, cache_tile)
                        z1_cache, z2_cache = get_dim1dim2(z1_cache, bf_image.Z, cache_tile)
                        c1_cache, c2_cache = get_dim1dim2(c1_cache, bf_image.C, cache_tile)
                        t1_cache, t2_cache = get_dim1dim2(t1_cache, bf_image.T, cache_tile)

                        logger.info("Caching: " + \
                                    "X ({0:0>4}-{1:0>4}), ".format(x1_cache, x2_cache) + \
                                    "Y ({0:0>4}-{1:0>4}), ".format(y1_cache, y2_cache) + \
                                    "Z ({0:0>4}-{1:0>4}), ".format(z1_cache, z2_cache) + \
                                    "C ({0:0>4}-{1:0>4}), ".format(c1_cache, c2_cache) + \
                                    "T ({0:0>4}-{1:0>4})".format(t1_cache, t2_cache))

                        bf_image.cache = bf_image[y1_cache:y2_cache, 
                                                  x1_cache:x2_cache,
                                                  z1_cache:z2_cache, 
                                                  c1_cache:c2_cache, 
                                                  t1_cache:t2_cache]
                        
                        
                        yield (y1_cache, y2_cache, \
                               x1_cache, x2_cache, \
                               z1_cache, z2_cache, \
                               c1_cache, c2_cache, \
                               z1_cache, z2_cache, bf_image.cache)

def get_highest_resolution_volumes(bf_image: bfio.bfio.BioReader,
                                   resolution_directory: str):
    """ This function gets the most detailed pyramid and saves it in encoded 
        chunks that can be processed by Neuroglancer.

    Args:
        bf_image: the image that gets read
        resolution_directory: the directory that the images get saved into    
    """
    # get tiles of 1024
    for y1_cache, y2_cache, \
        x1_cache, x2_cache, \
        z1_cache, z2_cache, \
        c1_cache, c2_cache, \
        t1_cache, t2_cache, bf_image_cache in iterate_cache_tiles(bf_image = bf_image):

        # now its XYZ order
        bf_image_cache = np.reshape(bf_image_cache, bf_image_cache.shape[:3])
        bf_image_cache = np.transpose(bf_image_cache, (1,0,2))

        # chunk sections of the tiles
        x_chunks = list(map(get_dim1dim2, 
                            range(x1_cache, x2_cache, CHUNK_SIZE), 
                            repeat(x2_cache), 
                            repeat(CHUNK_SIZE)))
        y_chunks = list(map(get_dim1dim2, 
                            range(y1_cache, y2_cache, CHUNK_SIZE), 
                            repeat(y2_cache), 
                            repeat(CHUNK_SIZE)))
        z_chunks = list(map(get_dim1dim2,
                            range(z1_cache, z2_cache, CHUNK_SIZE),
                            repeat(z2_cache),
                            repeat(CHUNK_SIZE)))
        xyz_chunks = product(x_chunks,y_chunks,z_chunks)
        # use multiprocessing to encode every chunk
        try:
            with ThreadPoolExecutor(max_workers = os.cpu_count()-1) as executor:
                executor.map(save_resolution,
                            repeat(resolution_directory),
                            ((xyz, bf_image_cache[xyz[0][0]-x1_cache:xyz[0][1]-x1_cache,
                                                  xyz[1][0]-y1_cache:xyz[1][1]-y1_cache,
                                                  xyz[2][0]-z1_cache:xyz[2][1]-z1_cache]) for xyz in xyz_chunks))
        except Exception as e:
            print(e)


def get_volumes(bf: bfio.bfio.BioReader,
                output_directory: str,
                highest_res_directory: str,
                imagetype: str,
                datatype: np.dtype,
                num_scales: int):
    """ This functions generates volumes 
    Args:
        bf : the input image that needs to processed into pyramids
        output_directory : the directory containing all the pyramids
        highest_res_directory : the directory containing the highest resolution tiles
        imagetype : Either image or segmentation 
        datatype : the datatype of the input image, bf
        num_scales : the number of pyramid levels
    """

    bfshape = bf.shape
    get_highest_resolution_volumes(bf_image=bf, resolution_directory=highest_res_directory)

    logger.info("\n Getting the Rest of the Pyramid ...")
    for higher_scale in reversed(range(0, num_scales)):
        
        inputshape = np.ceil(np.array(bfshape)/(2**(num_scales-higher_scale-1))).astype('int')
        inputshape = [inputshape[1], inputshape[0], inputshape[2]]
        scale_directory = os.path.join(output_directory, str(higher_scale+1)) #images are read from this directory
        if not os.path.exists(scale_directory):
            os.makedirs(scale_directory)
        assert os.path.exists(scale_directory), f"Key Directory {scale_directory} does not exist"
        
        if imagetype == "image":
            ngvol.get_rest_of_the_pyramid(directory=scale_directory, input_shape=inputshape, chunk_size=chunk_size,
                                        datatype=datatype, blurring_method='average')
        else:
            ngvol.get_rest_of_the_pyramid(directory=scale_directory, input_shape=inputshape, chunk_size=chunk_size,
                                        datatype=datatype, blurring_method='mode')
        logger.info(f"Saved Encoded Volumes for Scale {higher_scale} from Key Directory {os.path.basename(scale_directory)}")
              
    
def create_plyfiles(subvolume : np.ndarray, 
                    ids : list,
                    temp_dir : str,
                    start_y : int,
                    start_x : int,
                    start_z : int):
    """
    This function generates temporary ply files of labelled segments found 
    in the subvolume. 
    
    Args:
        subvolume : A chunk of the total volume
        ids : A list of labeled segments found in the subvolume
        temp_dir : temporary directory where outputs get saved to
        start_y : The start y index of the subvolume 
        start_x : The start x index of the subvolume
        start_z : The start z index of the subvolume

    Returns:
        None, saves subvolumes into temporary directory
    """

    for iden in ids:
        vertices,faces,_,_ = measure.marching_cubes((subvolume==iden).astype("uint8"), step_size=1)
        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces) # creates mesh
        chunk_filename = '{}_{}_{}_{}.ply'.format(iden, start_x, start_y, start_z)
        export_to = os.path.join(temp_dir, chunk_filename) # saves mesh in temp directory
        root_mesh.export(export_to)
        logger.debug("Saved Segment {} as {}".format(iden, chunk_filename))

def concatenate_and_generate_meshes(iden : int,
                                    temp_dir : str,
                                    output_image : str,
                                    bit_depth : int,
                                    chunk_size : list):
    """ This function concatenates the appropriate polygons in the temporary directory
    and generates progressive meshes as defined in neurogen.
    
    Args: 
        iden : The labeled segment that we are concatenating  
        temp_dir  : The directory where all the polygon files are located
        output_image : The output directory where all of Neuroglancer's files are stored
        bit_depth : Number of bits for mesh vertex quantization. Can only be 10 or 16. 
        chunk_size : Size of chunks in temporary file
    Returns:
        None, concatenates and saves the progressive meshes into the appropriate directory
    """
    try:
        # Get the files that are relevent to the segment iden
        chunkfiles = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        logger.info('Starting Progressive Meshes for ID {}'.format(iden))
        idenfiles = [str(f) for f in chunkfiles if f.split('_')[0] == str(iden)]
        len_files = len(idenfiles)
        logger.info('ID {} is scattered amoung {} chunk(s)'.format(str(iden), len_files))

        starts = []
        stripped_files = [i.strip('.ply').split('_')[1:] for i in idenfiles]
        for fil in range(len_files):
            start = [int(trans) for trans in stripped_files[fil]]
            starts.append(start)
        start_mesh = min(starts)
        mesh_index = starts.index(start_mesh)
        mesh_fileobj = idenfiles.pop(mesh_index)

        # Get the first mesh (upper left)
        mesh1_path = os.path.join(temp_dir, mesh_fileobj)
        mesh1 = trimesh.load_mesh(file_obj=mesh1_path, file_type='ply')
        translate_start = ([1, 0, 0, start_mesh[0]],
                           [0, 1, 0, start_mesh[1]],
                           [0, 0, 1, start_mesh[2]],
                           [0, 0, 0, 1])
        mesh1.apply_transform(translate_start)
        mesh1bounds = mesh1.bounds
        logger.debug('** Loaded chunk #1: {} ---- {} bytes'.format(mesh_fileobj, os.path.getsize(mesh1_path)))

        # if there is only one mesh, then decompose
        if len_files == 1:
            num_lods = math.ceil(math.log(len(mesh1.vertices),1024))
            ngmesh.fulloctree_decomposition_mesh(mesh1, num_lods=num_lods, 
                    segment_id=iden, directory=output_image, quantization_bits=bit_depth)
        # else concatenate the meshes
        else:
            stripped_files_middle = [idy.strip('.ply').split('_')[1:] for idy in idenfiles]
            for i in range(len_files-1):
                mesh2_path = os.path.join(temp_dir, idenfiles[i])
                mesh2 = trimesh.load_mesh(file_obj=mesh2_path, file_type='ply')
                logger.debug('** Loaded chunk #{}: {} ---- {} bytes'.format(i+2, idenfiles[i], os.path.getsize(mesh2_path)))
                transformationmatrix = [int(trans) for trans in stripped_files_middle[i]]
                offset = [transformationmatrix[i]/mesh_chunk_size[i] for i in range(3)]
                middle_mesh = transformationmatrix
                translate_middle = ([1, 0, 0, middle_mesh[0] - offset[0]],
                                    [0, 1, 0, middle_mesh[1] - offset[1]],
                                    [0, 0, 1, middle_mesh[2] - offset[2]],
                                    [0, 0, 0, 1])
                mesh2.apply_transform(translate_middle)
                mesh1 = trimesh.util.concatenate(mesh1, mesh2)
            num_lods = math.ceil(math.log(len(mesh1.vertices),1024))
            ngmesh.fulloctree_decomposition_mesh(mesh1, num_lods=num_lods, 
                    segment_id=iden, directory=output_image, quantization_bits=bit_depth)
    except Exception as e:
        traceback.print_exc()

def build_pyramid(input_image : str, 
                  output_image : str, 
                  imagetype : str, 
                  mesh : bool):

    """
    This function builds the pyramids for Volume Generation and Meshes (if specified)

    Args:
        input_image : Where the input directory is located
        output_image : Where the output directory is located
        imagetype : Specifying whether we are averaging or taking the mode of the images 
                    when blurring the images for the pyramids
        mesh : Whether or not meshes are generated with segmented volumes

    Returns:
        None, generates pyramids or volumes of input data
    
    Raises:
        ValueError: If imagetype is not properly specified
    """

    try:
        with bfio.BioReader(input_image) as bf:
            bf = BioReader(input_image)
            bfshape = (bf.X, bf.Y, bf.Z, bf.C, bf.T)
            datatype = np.dtype(bf.dtype)
            logger.info("Image Shape (XYZCT) {}".format(bfshape))

            logger.info("Image Datatype {}".format(datatype))

            num_scales = np.floor(np.log2(max(bfshape[:3]))).astype('int')+1
            highest_res_directory = os.path.join(output_image, f"{num_scales}")
            if not os.path.exists(highest_res_directory):
                os.makedirs(highest_res_directory)

            

            # info file specifications
            resolution = get_resolution(phys_y=bf.physical_size_y, 
                                        phys_x=bf.physical_size_x, 
                                        phys_z=bf.physical_size_z)


            if imagetype == "segmentation":
                if mesh == False:
                    logger.info("\n Creating info file for segmentations ...")
                    file_info = nginfo.info_segmentation(directory=output_image,
                                                        dtype=datatype,
                                                        chunk_size = chunk_size,
                                                        size=(bf.X, bf.Y, bf.Z),
                                                        resolution=resolution)
                    
                else: # if generating meshes
                    
                    # Creating a temporary files for the polygon meshes -- will later be converted to Draco
                    with tempfile.TemporaryDirectory() as temp_dir:

                        # keep track of labelled segments
                        all_identities = []
                        cache_tile = bf._TILE_SIZE
                        

                        logger.info("\n Starting to Cache Section Sizes of {}".format(cache_tile))
                        # cache tiles of 1024 
                        for y1_cache, y2_cache, \
                            x1_cache, x2_cache, \
                            z1_cache, z2_cache, \
                            c1_cache, c2_cache, \
                            t1_cache, t2_cache, bf.cache in iterate_cache_tiles(bf_image = bf):

                            cached_shape = bf.cache.shape
                            # iterate through mesh chunks in cached tile
                            bf.cache = np.reshape(bf.cache, cached_shape[:3])
                            bf.cache = np.transpose(bf.cache, (1,0,2))
                            
                            # chunk sections of the tiles
                            x_chunks = list(map(get_dim1dim2, 
                                                range(x1_cache, x2_cache, MESH_CHUNK_SIZE), 
                                                repeat(x2_cache), 
                                                repeat(MESH_CHUNK_SIZE)))
                            y_chunks = list(map(get_dim1dim2, 
                                                range(y1_cache, y2_cache, MESH_CHUNK_SIZE), 
                                                repeat(y2_cache), 
                                                repeat(MESH_CHUNK_SIZE)))
                            z_chunks = list(map(get_dim1dim2,
                                                range(z1_cache, z2_cache, MESH_CHUNK_SIZE),
                                                repeat(z2_cache),
                                                repeat(MESH_CHUNK_SIZE)))
                            xyz_chunks = product(x_chunks,y_chunks,z_chunks)

                            for xyz in xyz_chunks:

                                x1_chunk, x2_chunk = xyz[0]
                                y1_chunk, y2_chunk = xyz[1]
                                z1_chunk, z2_chunk = xyz[2] 

                                volume = bf.cache[x1_chunk-x1_cache:x2_chunk-x1_cache,
                                                  y1_chunk-y1_cache:y2_chunk-y1_cache,
                                                  z1_chunk-z1_cache:z2_chunk-z1_cache]
                                volume = np.reshape(volume, volume.shape[:3])

                                ids = np.unique(volume[volume>0])
                                len_ids = len(ids)
                                logger.debug("({0:0>4}, {0:0>4}), ".format(x1_chunk, x2_chunk) + \
                                             "({0:0>4}, {0:0>4}), ".format(y1_chunk, y2_chunk) + \
                                             "({0:0>4}, {0:0>4})  ".format(z1_chunk, z2_chunk) + \
                                             "has {0:0>2} IDS".format(len_ids))

                                all_identities = np.unique(np.append(all_identities, ids))
                                if len_ids > 0:
                                    with ThreadPoolExecutor(max_workers=max([cpu_count()-1,2])) as executor:
                                        executor.submit(create_plyfiles(subvolume = volume,
                                                                        ids=ids,
                                                                        temp_dir=temp_dir,
                                                                        start_y=y1_chunk,
                                                                        start_x=x1_chunk,
                                                                        start_z=z1_chunk))

                        # concatenate and decompose the meshes in the temporary file for all segments
                        logger.info("\n Generate Progressive Meshes for segments ...")
                        all_identities = np.unique(all_identities).astype('int')
                        with ThreadPoolExecutor(max_workers=max([cpu_count()-1,2])) as executor:
                            executor.map(concatenate_and_generate_meshes, 
                                        all_identities, repeat(temp_dir), repeat(output_image), repeat(bit_depth), repeat(mesh_chunk_size)) 

                    # Once you have all the labelled segments, then create segment_properties file
                    logger.info("\n Creating info file for segmentations and meshes ...")
                    file_info = nginfo.info_mesh(directory=output_image,
                                                chunk_size=chunk_size,
                                                size=(bf.X, bf.Y, bf.Z),
                                                dtype=np.dtype(bf.dtype).name,
                                                ids=all_identities,
                                                resolution=resolution,
                                                segmentation_subdirectory="segment_properties",
                                                bit_depth=bit_depth,
                                                order="XYZ")


            if imagetype == "image":
                file_info = nginfo.info_image(directory=output_image,
                                              dtype=datatype,
                                              chunk_size = chunk_size,
                                              size=(bf.X, bf.Y, bf.Z),
                                              resolution=resolution)


            logger.info(f"\n Creating chunked volumes of {chunk_size} based on the info file ...")
            get_volumes(bf                    = bf,
                        output_directory      = output_image,
                        highest_res_directory = highest_res_directory,
                        imagetype             = imagetype,
                        datatype              = datatype,
                        num_scales            = num_scales)

            
            logger.info("Data Type: {}".format(file_info['data_type']))
            logger.info("Number of Channels: {}".format(file_info['num_channels']))
            logger.info("Number of Scales: {}".format(len(file_info['scales'])))
            logger.info("Image Type: {}".format(file_info['type']))

    except Exception as e:
        traceback.print_exc()