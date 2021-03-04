import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS
import bioformats
import javabridge as jutil

import numpy as np
import os
from pathlib import Path

import trimesh
from skimage import measure

from concurrent.futures import ThreadPoolExecutor
import traceback
import ast
import math

from neurogen import info as nginfo
from neurogen import mesh as ngmesh

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
        vertices,faces,_,_ = measure.marching_cubes((volume==iden).astype("uint8"), step_size=1)
        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces) # creates mesh
        chunk_filename = '{}_{}_{}_{}.ply'.format(iden, start_y, start_x, start_z)
        export_to = os.path.join(temp_dir, chunk_filename) # saves mesh in temp directory
        root_mesh.export(export_to)

def concatenate_and_generate_meshes(iden,
                                    temp_dir,
                                    output_image,
                                    bit_depth):
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
    """

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
    logger.info('** Loaded chunk #1: {} ---- {} bytes'.format(mesh_fileobj, os.path.getsize(mesh1_path)))

    # if there is only one mesh, then decompose
    if len_files == 1:
        num_lods = math.ceil(math.log(len(mesh1.vertices),1024))
        ngmesh.fulloctree_decomposition_mesh(mesh1, num_lods=num_lods, 
                segment_id=iden, directory=output_image, quantization_bits=bit_depth)
    # else concatenate the meshes
    else:
        stripped_files_middle = [idy.strip('.ply').split('_')[1:] for idy in idenfiles]
        for i in range(len_files-1):
            mesh2_path = str(Path(temp_dir).joinpath(idenfiles[i]))
            mesh2 = trimesh.load_mesh(file_obj=mesh2_path, file_type='ply')
            logger.info('** Loaded chunk #{}: {} ---- {} bytes'.format(i+2, idenfiles[i], os.path.getsize(mesh2_path)))
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



if __name__=="__main__":
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_pyramid', description='Generate a precomputed slice for Polus Viewer.')

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)

    args = parser.parse_args()
    input_dir = args.input_dir
    output_image = args.output_dir
    image = os.path.basename(output_image)
    
    try:
        # Initialize the logger    
        logging.basicConfig(format='%(asctime)s - %(name)s - {} - %(levelname)s - %(message)s'.format(image),
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("build_pyramid")
        logger.setLevel(logging.INFO) 

        logger.info("Starting to build...")
        # logger.info("Values of xyz in stack are {}".format(valsinstack))

        # Initialize the javabridge
        logger.info('Initializing the javabridge...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

        # Create the BioReader object
        logger.info('Getting the BioReader...')
        logger.info(image)

        pathbf = os.path.join(input_dir, image)
        bf = BioReader(pathbf)
        bfshape = bf.shape
        chunk_size = [256, 256, 256]
        bit_depth = 16

        ysplits = list(np.arange(0, bfshape[0], chunk_size[0]))
        ysplits.append(bfshape[0])
        xsplits = list(np.arange(0, bfshape[1], chunk_size[1]))
        xsplits.append(bfshape[1])
        zsplits = list(np.arange(0, bfshape[2], chunk_size[2]))
        zsplits.append(bfshape[2])

        all_identities = np.array([])
        totalbytes = {}
        temp_dir = os.path.join(output_image, "tempdir")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        for y in range(len(ysplits)-1):
            for x in range(len(xsplits)-1):
                for z in range(len(zsplits)-1):
                    start_y, end_y = (ysplits[y], ysplits[y+1])
                    start_x, end_x = (xsplits[x], xsplits[x+1])
                    start_z, end_z = (zsplits[z], zsplits[z+1])
                    
                    volume = bf[start_y:end_y,start_x:end_x,start_z:end_z].squeeze()
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

        executor.shutdown(wait=True)

        all_identities = np.unique(all_identities).astype('int')
        file_info = nginfo.info_mesh(directory=output_image,
                                     chunk_size=chunk_size,
                                     size=bf.shape[:3],
                                     dtype=np.dtype(bf.dtype).name,
                                     ids=all_identities,
                                     segmentation_subdirectory="segment_properties",
                                     bit_depth=bit_depth)

        logger.info("Image Shape {}".format(bf.shape))
        numscales = len(file_info['scales'])
        logger.info("number of scales: {}".format(numscales))
        logger.info("data type: {}".format(file_info['data_type']))
        logger.info("number of channels: {}".format(file_info['num_channels']))
        logger.info("number of scales: {}".format(numscales))
        logger.info("image type: {}".format(file_info['type']))

        logger.info("Labelled Segments in Input Data: {}".format(all_identities))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futuresvariable = [executor.submit(concatenate_and_generate_meshes, ide, temp_dir, output_image, bit_depth) for ide in all_identities]
        executor.shutdown(wait=True)

    except Exception as e:
        traceback.print_exc()
    finally:
        jutil.kill_vm()
