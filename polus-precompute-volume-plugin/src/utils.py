import numpy as np
import json, copy, os
import math

import logging, traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import tempfile

import trimesh
from skimage import measure

from neurogen import mesh as ngmesh
from neurogen import info as nginfo
from neurogen import volume as ngvol

from bfio.bfio import BioReader, BioWriter

chunk_size = [64,64,64]
bit_depth = 10

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
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
        translate_start = ([0, 1, 0, start_mesh[0]],
                           [1, 0, 0, start_mesh[1]],
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
                translate_middle = ([0, 1, 0, middle_mesh[0] - offset[0]],
                                    [1, 0, 0, middle_mesh[1] - offset[1]],
                                    [0, 0, 1, middle_mesh[2] - offset[2]],
                                    [0, 0, 0, 1])
                mesh2.apply_transform(translate_middle)
                mesh1 = trimesh.util.concatenate(mesh1, mesh2)
            num_lods = math.ceil(math.log(len(mesh1.vertices),1024))
            ngmesh.fulloctree_decomposition_mesh(mesh1, num_lods=num_lods, 
                    segment_id=iden, directory=output_image, quantization_bits=bit_depth)
    except Exception as e:
        traceback.print_exc()

def build_pyramid(input_image, 
                  output_image, 
                  imagetype, 
                  mesh):

    """
    This function builds the pyramids for Volume Generation and Meshes (if specified)

    Parameters
    ----------
    input_image : str
        Where the input directory is located
    output_image : str
        Where the output directory is located
    imagetype : str
        Specifying whether we are averaging or taking the mode of the images 
        when blurring the images for the pyramids
    mesh : boolean
        Whether or not meshes are generated with segmented volumes

    Returns
    -------
    Pyramids or volumes of input data
    """

    # Getting the intial information for the info file specification required by Neuroglancer 
    bf = BioReader(input_image, max_workers=max([cpu_count()-1,2]))
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
            
            # Need to iterate through chunks of the input for scalabiltiy
            xsplits = list(np.arange(0, bfshape[0], 256))
            xsplits.append(bfshape[0])
            ysplits = list(np.arange(0, bfshape[1], 256))
            ysplits.append(bfshape[1])
            zsplits = list(np.arange(0, bfshape[2], 256))
            zsplits.append(bfshape[2])

            # Keep track of the labelled segments
            all_identities = np.array([])
            totalbytes = {}
            temp_dir = os.path.join(output_image, "tempdir")

            num_scales = np.floor(np.log2(max(bfshape[:3]))).astype('int')+1
            
            logger.info("\n Iterate through input ...")
            # Creating a temporary files for the polygon meshes -- will later be converted to Draco
            with tempfile.TemporaryDirectory() as temp_dir:
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir, exist_ok=True)
                for y in range(len(ysplits)-1):
                    for x in range(len(xsplits)-1):
                        for z in range(len(zsplits)-1):
                            start_y, end_y = (ysplits[y], ysplits[y+1])
                            start_x, end_x = (xsplits[x], xsplits[x+1])
                            start_z, end_z = (zsplits[z], zsplits[z+1])
                            
                            volume = bf[start_x:end_x,start_y:end_y,start_z:end_z]
                            volume = volume.reshape(volume.shape[:3])

                            logger.info("Loaded subvolume (YXZ) {}-{}__{}-{}__{}-{}".format(start_y, end_y,
                                                                                            start_x, end_x,
                                                                                            start_z, end_z))
                            ids = np.unique(volume)
                            if (ids == [0]).all():
                                continue
                            else:
                                ids = np.delete(ids, np.where(ids==0))
                                with ThreadPoolExecutor(max_workers=8) as executor:
                                    executor.submit(create_plyfiles(subvolume = volume,
                                                                    ids=ids,
                                                                    temp_dir=temp_dir,
                                                                    start_y=start_y,
                                                                    start_x=start_x,
                                                                    start_z=start_z,
                                                                    totalbytes=totalbytes))
                                    all_identities = np.append(all_identities, ids)

                logger.info("\n Generate Progressive Meshes for segments ...")
                # concatenate and decompose the meshes in the temporary file for all segments
                all_identities = np.unique(all_identities).astype('int')
                with ThreadPoolExecutor(max_workers=4) as executor:
                    variable = [executor.submit(concatenate_and_generate_meshes, 
                                                    ide, temp_dir, output_image, bit_depth, chunk_size) 
                                                    for ide in all_identities]
            
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
                                        order="YXZ")

        logger.info("\n Creating volumes based on the info file ...")
        # this is written outside the if/else statement regarding meshes
        ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_image, blurring_method='mode')


    elif imagetype == "image":
        file_info = nginfo.info_image(directory=output_image,
                                            dtype=datatype,
                                            chunk_size = chunk_size,
                                            size=bfshape[:3],
                                            resolution=resolution)
        logger.info("\n Creating volumes based on the info file ...")
        ngvol.generate_recursive_chunked_representation(volume=bf, info=file_info, dtype=datatype, directory=output_image, blurring_method='average')
    else:
        raise ValueError("Image Type was not properly specified")
    
    logger.info("Data Type: {}".format(file_info['data_type']))
    logger.info("Number of Channels: {}".format(file_info['num_channels']))
    logger.info("Number of Scales: {}".format(len(file_info['scales'])))
    logger.info("Image Type: {}".format(file_info['type']))

