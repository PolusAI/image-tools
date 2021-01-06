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

# Create two levels of detail
num_lods = 2

# Merge verticies that are closer than 1 pixel
trimesh.constants.tol.merge = 1

# Create the info file
with open(str(output_path.joinpath("info")), 'w') as info:
    jsoninfo = {
   "@type" : "neuroglancer_multilod_draco",
   "lod_scale_multiplier" : 1,
   "transform" : [0,   325,   0,   325,
                    325, 0,   0,   325,
                    0,   0, 325,   325],
   "vertex_quantization_bits" : bit_depth 
    }
    info.write((json.dumps(jsoninfo)))

# Start the JVM
log_config = Path(__file__).parent.joinpath("log4j.properties")
jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(LOG4J))],class_path=JARS)
fragoffsum = 0
Zorder = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 0, 0],
    [0, 1, 1], 
    [1, 1, 0], 
    [1, 1, 1]
]
#checking to see if I login
try:
    # Load the image
    br = BioReader(dataname,backend='java')
    volume = br[:].squeeze()
    print(volume.shape)
    # Get the ids for each segment
    IDS = np.unique(volume)
    
    # Master draco file offset
    fragment_offset = 0

    # need to create a for loop for all the ids.
    for iden in IDS[1:]:
        print('Processing label {}'.format(iden))
        
        fragment_offsets = []
        fragment_positions = []
        num_fragments_per_lod = []
        vertex_offsets = []
        lod_scales = []
        
        chunk_shape = None
        vertices,faces,_,_ = measure.marching_cubes(volume==IDS[iden], step_size=1)
        
        
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        dim = max_bounds - min_bounds

        root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        multires_mesh.generate_multires_mesh(mesh=root_mesh,
                                             directory=output_path,
                                             segment_id=1,
                                             num_lods=2,
                                             quantization_bits=bit_depth)
        # scalez = multires_mesh.Quantize(fragment_origin = min_bounds, fragment_shape=dim,input_origin=min_bounds,quantization_bits=bit_depth)
        # newvertices = scalez(vertices)
        # print("OFFSET: ", scalez.offset)
        # print("NEW SCALE: ", scalez.scale)
        # print("DIMNSION: ", dim)
        # print("Minimum Bounds: ", min_bounds)
        # print("UPPER BOUNDS", scalez.upper_bound)
        
        # print("NEWVERTICES: ", newvertices(vertices))


        # for i in range(num_lods):
        #     fragcount = 0
        #     concatmesh = 0
        #     fragment_positions.append([])
        #     fragment_offsets.append([])
        #     lod_scales.append(float(2 ** i))
            
        #     num_fragments_per_lod.append(0)
        #     vertex_offsets.append([0, 0, 0])
        
        #     root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        #     rootmeshbounds = root_mesh.bounds
        #     minrootbounds = rootmeshbounds[0]
        #     maxrootbounds = rootmeshbounds[1]

        #     zstep = dim[2]/(2 ** (num_lods - i - 1))
        #     ystep = dim[1]/(2 ** (num_lods - i - 1))
        #     xstep = dim[0]/(2 ** (num_lods - i - 1))
        #     if isinstance(chunk_shape,type(None)):
        #         chunk_shape = np.asarray([xstep,ystep,zstep]).astype(np.float32)
        #     xslice = 0
        #     yslice = 0 
        #     zslice = 0
        #     nyz, nxz, nxy = np.eye(3)
        #     # print(nyz, nxz, nxy)

        #     for z in np.arange(min_bounds[2],max_bounds[2],zstep):
        #         z_section = trimesh.intersections.slice_mesh_plane(root_mesh,
        #                                                         plane_normal = nxy,
        #                                                         plane_origin = (0.0,0.0,float(z)))
        #         z_section = trimesh.intersections.slice_mesh_plane(z_section,
        #                                                         plane_normal = -1*nxy,
        #                                                         plane_origin=(0.0,0.0,float(z + zstep)))
        #         for y in np.arange(min_bounds[1],max_bounds[1],ystep):
        #             y_section = trimesh.intersections.slice_mesh_plane(z_section,
        #                                                             plane_normal=nxz,
        #                                                             plane_origin=(0.0,float(y),0.0))
        #             y_section = trimesh.intersections.slice_mesh_plane(y_section,
        #                                                             plane_normal=nxz*-1,
        #                                                             plane_origin=(0.0,float(y + ystep),0.0))
        #             for x in np.arange(min_bounds[0],max_bounds[0],xstep):
        #                 x_section = trimesh.intersections.slice_mesh_plane(y_section,
        #                                                                 plane_normal=nyz,
        #                                                                 plane_origin=(float(x),0.0,0.0))
                                                   
        #                 x_section = trimesh.intersections.slice_mesh_plane(x_section,
        #                                                                 plane_normal=nyz*-1,
        #                                                                 plane_origin=(float(x+xstep),0.0,0.0))

                        
        #                 if len(x_section.vertices) == 0:
        #                     print("continue")
        #                     continue
        #                 fragment_positions[-1].append([       
        #                     (x-min_bounds[0]) / xstep,
        #                     (y-min_bounds[1]) / ystep,
        #                     (z-min_bounds[2]) / zstep,
        #                 ])
                        
        #                 # zmin_bounds = x_section.vertices.min(axis=0)
        #                 # zmax_bounds = x_section.vertices.max(axis=0)

        #                 # zmin_bounds = x_section.bounds[0]
        #                 # zmax_bounds = x_section.bounds[1]
        #                 # scale =  np.asarray([xstep,ystep,zstep,1]) / ((2 ** bit_depth) - 1)
        #                 # print("OG SCALE: ", scale)

        #                 # transform = np.asarray([[1, 0, 0, -x/xstep],
        #                 #                         [0, 1, 0, -y/ystep],
        #                 #                         [0, 0, 1, -z/zstep],
        #                 #                         [0, 0, 0,  1]]) / scale
        #                 # x_section.apply_transform(transform)
        #                 print("adding to draco files")
        #                 drcfile = output_path.joinpath(str(iden))
        #                 with open(str(drcfile), "ab+") as draco:
        #                     start = draco.tell()
        #                     writethis = DracoPy.encode_mesh_to_buffer(points=x_section.vertices.flatten(),
        #                                                               faces = x_section.faces.flatten(),
        #                                                               quantization_bits=bit_depth,
        #                                                               compression_level=0)
        #                     draco.write(writethis)
        #                     num_fragments_per_lod[-1] += 1
        #                     fragment_offsets[-1].append(draco.tell() - start)
        #                     fragcount = fragcount + 1

        
        # num_fragments_per_lod = np.asarray(num_fragments_per_lod).astype('<I')
        # gridorigin = min_bounds
        # manifest_file = output_path.joinpath((str(iden)+".index"))
        # vertex_offsets = np.asarray(vertex_offsets).astype('<f')
        # with open(str(manifest_file), 'wb') as index:
        #     index.write(chunk_shape.astype('<f').tobytes(order='C'))
        #     index.write(gridorigin.astype('<f').tobytes(order="C"))
        #     index.write(struct.pack("<I",num_lods))
        #     index.write(np.asarray(lod_scales).astype('<f').tobytes(order="C"))
        #     index.write(vertex_offsets.tobytes(order="C"))
        #     index.write(num_fragments_per_lod.astype('<I').tobytes(order="C"))

        #     for i in range(0, num_lods):
        #         fp = np.asarray(fragment_positions[i]).astype('<I')
        #         fp = fp.T
        #         index.write(fp.astype('<I').tobytes(order="C"))
        #         fo = np.asarray(fragment_offsets[i]).astype('<I')
        #         index.write(fo.tobytes(order="C"))

except Exception as e:
    traceback.print_exc()
finally:
    jutil.kill_vm()
