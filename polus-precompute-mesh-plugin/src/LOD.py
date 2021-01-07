import numpy as np
from pathlib import Path
from skimage import measure

import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS, LOG4J
import bioformats
import javabridge as jutil

import struct,json
from struct import *
import traceback
import trimesh
import math
import pandas
import shutil
import DracoPy
import multires_mesh

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

#checking to see if I login
try:
    # Load the image
    br = BioReader(dataname,backend='java')
    volume = br[:].squeeze()
    print(volume.shape)

    # Get the ids for each segment
    IDS = np.unique(volume)
    
    # Master draco file offset

    np.save("volume.npy", volume)
    # need to create a for loop for all the ids.
    for iden in IDS[1:2]:
        print('Processing label {}'.format(iden))
        vertices,faces,_,_ = measure.marching_cubes((volume==IDS[iden]).astype("uint8"), level=0, step_size=1)

        # range goal is (32-64)
        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        dimensions = root_mesh.bounds
        shape = dimensions[1] - dimensions[0]
        LOD = np.floor(np.log2(shape))


        multires_mesh.generate_multires_mesh(mesh=root_mesh,
                                             directory=str(output_path),
                                             segment_id=iden,
                                             num_lods=2,
                                             quantization_bits=bit_depth)


except Exception as e:
    traceback.print_exc()
finally:
    jutil.kill_vm()