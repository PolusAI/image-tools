import numpy as np
from pathlib import Path
from skimage import measure

import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS, LOG4J
import bioformats
import javabridge as jutil

import os
import struct,json
from struct import *
import traceback
import trimesh
import math
import pandas
import shutil
import DracoPy
import iterative_density_decomposition as scalable_multires
# import scalable_multires
from os import listdir
from os.path import isfile, join
import ast
import psutil
from collections import defaultdict

# import bpy, bmesh

# Needs to be specified
input_path = Path('/home/ubuntu/3D_data')
dataname = input_path.joinpath('dA30_5_dA30.Labels.ome.tif')
output_path = Path('/home/ubuntu/mrmesh/polus-plugins/polus-precompute-mesh-plugin/src/MultipleLODs')

if output_path.exists():
    shutil.rmtree(str(output_path))
output_path.mkdir(exist_ok=True)

# Bit depth of the draco coordinates, must be 10 or 16
bit_depth = 16

#transformation matrix 
transformation_matrix=[0, 325, 0, 0, 325, 0, 0, 0,0, 0, 0, 325]

# Create the info file
with open(str(output_path.joinpath("info")), 'w') as info:
    jsoninfo = {
   "@type" : "neuroglancer_multilod_draco",
   "lod_scale_multiplier" : 1,
   "transform" : transformation_matrix,
   "vertex_quantization_bits" : bit_depth 
    }
    info.write((json.dumps(jsoninfo)))

# Start the JVM
log_config = Path(__file__).parent.joinpath("log4j.properties")
jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(LOG4J))],class_path=JARS)

Tile_Size = (1024,1024,256)
#checking to see if I login
try:
    # Load the image

    br = BioReader(dataname,backend='java')
    print('SIZE OF INPUT: ({}, {}, {})'.format(br.X, br.Y, br.Z))

    chunk_x = [[box_x,box_x + Tile_Size[0]] for box_x in range(0, br.X, Tile_Size[0])]
    chunk_x[-1][-1] = br.X
    chunk_y = [[box_y, box_y + Tile_Size[1]] for box_y in range(0, br.Y, Tile_Size[1])]
    chunk_y[-1][-1] = br.Y
    chunk_z = [[box_z, box_z + Tile_Size[2]] for box_z in range(0, br.Z, Tile_Size[2])]
    chunk_z[-1][-1] = br.Z

    num_of_chunks = len(chunk_x) * len(chunk_y) * len(chunk_z)
    print('Iterating Input through {} Chunks'.format(num_of_chunks))

    all_idens = []
    # print('Available Memory {}'.format(psutil.virtual_memory().available))
    for ch_x in chunk_x:
        for ch_y in chunk_y:
            for ch_z in chunk_z:
                volume = br[ch_y[0]:ch_y[1],ch_x[0]:ch_x[1],ch_z[0]:ch_z[1],0,0]
                IDS = np.unique(volume)
                print(" ")
                print('Tile Chunk: ({}, {}, {}) contains IDS {}'.format(ch_y, ch_x, ch_z, IDS))
                if all(v==0 for v in IDS):
                    continue
                else:
                    for iden in IDS[1:2]:
                        if iden == 0:
                            continue
                        if iden not in all_idens:
                            all_idens.append(iden)
                        print('** Processing label ID {} in section ({}, {}, {})'.format(iden, ch_x, ch_y, ch_z))
                        volume = volume.squeeze()
                        vertices,faces,_,_ = measure.marching_cubes((volume==iden).astype("uint8"), level=0, step_size=1)

                        # range goal is (32-64)
                        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        dimensions = root_mesh.bounds
                        shape = dimensions[1] - dimensions[0]
                        LOD = np.floor(np.log2(shape))
                        # print('Available Memory {}'.format(psutil.virtual_memory().available))
                        scalable_multires.generate_trimesh_chunks(mesh=root_mesh,
                                                            directory=str(output_path),
                                                            segment_id=iden,
                                                            chunks=(ch_x, ch_y, ch_z))

    temp_dir = "/home/ubuntu/mrmesh/polus-plugins/polus-precompute-mesh-plugin/src/MultipleLODs/temp_drc"
    chunkfiles = [f for f in listdir(temp_dir) if isfile(join(temp_dir, f))]
    all_idens.sort()
    print(" ")
    for ide in all_idens:
        starts = []
        print('Starting Progressive Meshes for ID {}'.format(ide))
        idenfiles = [str(f) for f in chunkfiles if f.split('_')[0] == str(ide)]
        len_files = len(idenfiles)
        print('ID {} is scattered amoung {} chunk(s)'.format(str(ide), len_files))
        stripped_files = [i.strip('.ply').split('_')[1:] for i in idenfiles]
        for fil in range(len_files):
            start = [ast.literal_eval(trans)[0] for trans in stripped_files[fil]]
            starts.append(start)
        start_mesh = min(starts)
        mesh_index = starts.index(start_mesh)
        mesh_fileobj = idenfiles.pop(mesh_index)

        mesh1_path = str(Path(temp_dir).joinpath(mesh_fileobj))
        mesh1 = trimesh.load_mesh(file_obj=mesh1_path, file_type='ply')
        translate_start = ([1, 0, 0, start_mesh[1]],
                           [0, 1, 0, start_mesh[0]],
                           [0, 0, 1, start_mesh[2]],
                           [0, 0, 0, 1])
        mesh1.apply_transform(translate_start)
        mesh1bounds = mesh1.bounds
        print('** Loaded chunk #1: {} ---- {} bytes'.format(mesh_fileobj, os.path.getsize(mesh1_path)))
        if len_files == 1:
            num_lods, df_num_lods = scalable_multires.calculate_numlods(meshvertices=mesh1.vertices, lod=0)
            fragment_info = scalable_multires.generate_multires_mesh(mesh=mesh1,
                                                    directory=str(output_path),
                                                    segment_id=ide,
                                                    dataframe = df_num_lods,
                                                    num_lods = num_lods,
                                                    quantization_bits=bit_depth)
            scalable_multires.generate_manifest_file(meshbounds=mesh1bounds, segment_id = ide, directory=str(output_path), num_lods = len(fragment_info), fragment_info=fragment_info)
        else:
            stripped_files_middle = [idy.strip('.ply').split('_')[1:] for idy in idenfiles]
            for i in range(len_files-1):
                mesh2_path = str(Path(temp_dir).joinpath(idenfiles[i]))
                mesh2 = trimesh.load_mesh(file_obj=mesh2_path, file_type='ply')
                print('** Loaded chunk #{}: {} ---- {} bytes'.format(i+2, idenfiles[i], mesh2_path))
                transformationmatrix = [ast.literal_eval(trans) for trans in stripped_files_middle[i]]
                offset = [transformationmatrix[i][0]/Tile_Size[i] for i in range(3)]
                middle_mesh = [trans[0] for trans in transformationmatrix]
                translate_middle = ([1, 0, 0, middle_mesh[1] - offset[1]],
                                    [0, 1, 0, middle_mesh[0] - offset[0]],
                                    [0, 0, 1, middle_mesh[2] - offset[2]],
                                    [0, 0, 0, 1])
                mesh2.apply_transform(translate_middle)
                mesh1 = trimesh.util.concatenate(mesh1, mesh2)
            scalable_multires.generate_multires_mesh(mesh=mesh1,
                                                    directory=str(output_path),
                                                    segment_id=ide,
                                                    nodearray = [0,0,0],
                                                    recur = 0,
                                                    num_lods = 0,
                                                    quantization_bits=bit_depth)
        print(" ")
except Exception as e:
    traceback.print_exc()
finally:
    jutil.kill_vm()