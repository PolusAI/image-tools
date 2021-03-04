import os
import logging
import traceback

import ast
import math

import trimesh
from skimage import measure
from neurogen import mesh as ngmesh

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utilities")
logger.setLevel(logging.INFO) 

def get_resolution(phys_y, phys_x, phys_z):
    """ 
    This function generates a resolution in nm 
    
    Parameters
    ----------
    phys_y : tuple
        Actual y dimension of input
    phys_x : tuple
        Actual x dimension of input
    phys_z : tuple
        Actual z dimension of input
    
    Returns
    -------
    resolution : list
        The integer values of resolution in nanometers in [Y, X, Z] order
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

def create_plyfiles(subvolume, 
                    ids,
                    temp_dir,
                    start_y,
                    start_x,
                    start_z,
                    totalbytes=None):
    """
    This function generates temporary ply files of labelled segments found 
    in the subvolume. 

    Parameters
    ----------
    subvolume : numpy array
        A chunk of the total volume
    ids : list
        A list of labeled segments found in the subvolume
    start_y : int
        The start y index of the subvolume 
    start_x : int
        The start x index of the subvolume
    start_z : int
        The start z index of the subvolume
    """

    for iden in ids:
        vertices,faces,_,_ = measure.marching_cubes((subvolume==iden).astype("uint8"), step_size=1)
        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces) # creates mesh
        chunk_filename = '{}_{}_{}_{}.ply'.format(iden, start_y, start_x, start_z)
        export_to = os.path.join(temp_dir, chunk_filename) # saves mesh in temp directory
        root_mesh.export(export_to)
        logger.debug("Saved Segment {} as {}".format(iden, chunk_filename))

def concatenate_and_generate_meshes(iden,
                                    temp_dir,
                                    output_image,
                                    bit_depth,
                                    chunk_size):
    """ This function concatenates the appropriate polygons in the temporary directory
    and generates progressive meshes as defined.

    Parameters
    ----------
    iden : int
        The labeled segment that we are concatenating  
    temp_dir  : str
        The directory where all the polygon files are located
    output_image : str
        The output directory where all of Neuroglancer's files are stored
    bit_depth : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    resolution : list
        Resolution of input data 
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
        translate_start = ([0, 1, 0, start_mesh[1]],
                            [1, 0, 0, start_mesh[0]],
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
                translate_middle = ([0, 1, 0, middle_mesh[1] - offset[1]],
                                    [1, 0, 0, middle_mesh[0] - offset[0]],
                                    [0, 0, 1, middle_mesh[2] - offset[2]],
                                    [0, 0, 0, 1])
                mesh2.apply_transform(translate_middle)
                mesh1 = trimesh.util.concatenate(mesh1, mesh2)
            num_lods = math.ceil(math.log(len(mesh1.vertices),1024))
            ngmesh.fulloctree_decomposition_mesh(mesh1, num_lods=num_lods, 
                    segment_id=iden, directory=output_image, quantization_bits=bit_depth)
    except Exception as e:
        traceback.print_exc()
