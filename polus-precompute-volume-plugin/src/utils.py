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

chunk_size = [64, 64, 64]
mesh_chunk_size = [256, 256, 256]

bit_depth = 10

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)

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
        chunk_filename = '{}_{}_{}_{}.ply'.format(iden, start_y, start_x, start_z)
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
                offset = [transformationmatrix[i]/chunk_size[i] for i in range(3)]
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
            bfshape = bf.shape
            datatype = np.dtype(bf.dtype)
            logger.info("Image Shape {}".format(bfshape))
            logger.info("Image Datatype {}".format(datatype))
            
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
                                                        size=bfshape[:3],
                                                        resolution=resolution)
                    
                else: # if generating meshes
                    
                    # Creating a temporary files for the polygon meshes -- will later be converted to Draco
                    with tempfile.TemporaryDirectory() as temp_dir:
                        
                        # function to get the range of chunks
                        get_dim1dim2 = lambda dimension1, dimension_size, rng_size: \
                                            (int(dimension1), int(min(dimension1+rng_size, dimension_size)))

                        # keep track of labelled segments
                        all_identities = []

                        # cache tiles of 1024 
                        for y1_cache in range(0, bf.Y, bf._TILE_SIZE):
                            for x1_cache in range(0, bf.X, bf._TILE_SIZE):
                                for z1_cache in range(0, bf.Z, bf._TILE_SIZE):
                                    for c1_cache in range(0, bf.C, bf._TILE_SIZE):
                                        for t1_cache in range(0, bf.T, bf._TILE_SIZE):

                                            y1_cache, y2_cache = get_dim1dim2(y1_cache, bf.Y, bf._TILE_SIZE)
                                            x1_cache, x2_cache = get_dim1dim2(x1_cache, bf.X, bf._TILE_SIZE)
                                            z1_cache, z2_cache = get_dim1dim2(z1_cache, bf.Z, bf._TILE_SIZE)
                                            c1_cache, c2_cache = get_dim1dim2(c1_cache, bf.C, bf._TILE_SIZE)
                                            t1_cache, t2_cache = get_dim1dim2(t1_cache, bf.T, bf._TILE_SIZE)

                                            bf.cache = bf[y1_cache:y2_cache, 
                                                          x1_cache:x2_cache, 
                                                          z1_cache:z2_cache, 
                                                          c1_cache:c2_cache, 
                                                          t1_cache:t2_cache]

                                            cached_shape = bf.cache.shape

                                            # iterate through mesh chunks in cached tile
                                            for y1_chunk in range(0, cached_shape[0], mesh_chunk_size[0]):
                                                for x1_chunk in range(0, cached_shape[1], mesh_chunk_size[1]):
                                                    for z1_chunk in range(0, cached_shape[2], mesh_chunk_size[2]):
                                                
                                                        y1_chunk, y2_chunk = get_dim1dim2(y1_chunk, cached_shape[0], mesh_chunk_size[0])
                                                        x1_chunk, x2_chunk = get_dim1dim2(x1_chunk, cached_shape[1], mesh_chunk_size[1])
                                                        z1_chunk, z2_chunk = get_dim1dim2(z1_chunk, cached_shape[2], mesh_chunk_size[2])

                                                        volume = bf.cache[y1_chunk:y2_chunk, x1_chunk:x2_chunk, z1_chunk:z2_chunk]
                                                        volume = np.reshape(volume, volume.shape[:3])
                                                        
                                                        ids = np.unique(volume[volume>0])
                                                        all_identities = np.unique(np.append(all_identities, ids))
                                                        if len(ids) > 0:
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
                                                size=bf.shape[:3],
                                                dtype=np.dtype(bf.dtype).name,
                                                ids=all_identities,
                                                resolution=resolution,
                                                segmentation_subdirectory="segment_properties",
                                                bit_depth=bit_depth,
                                                order="XYZ")

                logger.info("\n Creating volumes based on the info file ...")
                # this is written outside the if/else statement regarding meshes
                ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_image, blurring_method='mode')


            if imagetype == "image":
                file_info = nginfo.info_image(directory=output_image,
                                                    dtype=datatype,
                                                    chunk_size = chunk_size,
                                                    size=bfshape[:3],
                                                    resolution=resolution)
                                                    
                logger.info("\n Creating volumes based on the info file ...")
                ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_image, blurring_method='average')

            
            logger.info("Data Type: {}".format(file_info['data_type']))
            logger.info("Number of Channels: {}".format(file_info['num_channels']))
            logger.info("Number of Scales: {}".format(len(file_info['scales'])))
            logger.info("Image Type: {}".format(file_info['type']))

    except Exception as e:
        traceback.print_exc()