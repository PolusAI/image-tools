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

# Needs to be specified
input_path = Path('/home/ubuntu/3D_data')
dataname = input_path.joinpath('dA30_5_dA30.Labels.ome.tif')
output_path = Path('/home/ubuntu/polus-plugins/polus-precompute-mesh-plugin/src/MultipleLODs')

if output_path.exists():
    shutil.rmtree(str(output_path))
output_path.mkdir(exist_ok=True)

# Bit depth of the draco coordinates, must be 10 or 16
bit_depth = 10

# Create two levels of detail
num_lods = None

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
    for iden in IDS[25:]:
        print('Processing label {}'.format(iden))
        
        fragment_offsets = []
        fragment_positions = []
        num_fragments_per_lod = []
        vertex_offsets = []
        lod_scales = []
        
        chunk_shape = None
        # vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==IDS[iden], step_size=1)
        # min_bounds = vertices.min(axis=0)
        # max_bounds = vertices.max(axis=0)
        # dim = max_bounds - min_bounds
        # print("DIMENSION", dim)
        vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==IDS[iden], step_size=1)
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        dim = max_bounds - min_bounds

        maxvol = 128*128*128
        multdim = np.prod(dim)
        if multdim/maxvol < 1:
            num_lods = 1
        else:
            num_lods = int(np.floor(multdim/maxvol))
        print("NUMBER OF DETAILS: ", num_lods)
        print("DIMENSION: ", dim)
        
        for i in range(num_lods):
            fragcount = 0
            concatmesh = 0
            fragment_positions.append([])
            fragment_offsets.append([])
            lod_scales.append(float(2 ** i))
            
            num_fragments_per_lod.append(0)
            vertex_offsets.append([0*(2 ** i) for _ in range(3)])
        
            # Create the mesh
            # vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==IDS[iden], step_size=(i*2)+1)
            # vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==IDS[iden], step_size=1)
            # print(vert0.shape, vert1.shape)


            # getting the dimensions of the segment
            # min_bounds = vertices.min(axis=0)
            # max_bounds = vertices.max(axis=0)
            # dim = max_bounds - min_bounds
            # print("DIMENSION", dim)
            root_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            rootmeshbounds = root_mesh.bounds
            minrootbounds = rootmeshbounds[0]
            maxrootbounds = rootmeshbounds[1]
            # print("ROOT MESH BOUNDS", rootmeshbounds)
            # print("")
            zstep = dim[2]/(2 ** (num_lods - i - 1))
            ystep = dim[1]/(2 ** (num_lods - i - 1))
            xstep = dim[0]/(2 ** (num_lods - i - 1))
            # print("XYZ STEPS: ", xstep, ystep, zstep)
            if isinstance(chunk_shape,type(None)):
                chunk_shape = np.asarray([xstep,ystep,zstep]).astype(np.float32)
            xslice = 0
            yslice = 0 
            zslice = 0
            # poszdot = np.dot((0.0,0.0,1.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]
            # negzdot = np.dot((0.0,0.0,-1.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]
            # posydot = np.dot((0.0,1.0,0.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]
            # negydot = np.dot((0.0,-1.0,0.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]
            # posxdot = np.dot((1.0,0.0,0.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]
            # negxdot = np.dot((-1.0,0.0,0.0), (root_mesh.vertices - root_mesh.centroid).T)[root_mesh.faces]

            # for x in np.arrange(min_bounds[0], max_bounds[0], xstep):
            #     x_section = trimesh.split(root_mesh, only_watertight=True, ad)
            
                # print("X SECTION WATERTIGHT?", x_section.is_watertight)
            for z in np.arange(min_bounds[2],max_bounds[2],zstep):
                z_section = trimesh.intersections.slice_mesh_plane(root_mesh,
                                                                (0.0,0.0,1.0),
                                                                (0.0,0.0,float(z)))
                # print("Z Section", z_section.is_watertight)
                z_section = trimesh.intersections.slice_mesh_plane(z_section,
                                                                (0.0,0.0,-1.0),
                                                                (0.0,0.0,float(z + zstep)))
                for y in np.arange(min_bounds[1],max_bounds[1],ystep):
                    y_section = trimesh.intersections.slice_mesh_plane(z_section,
                                                                        (0.0,1.0,0.0),
                                                                        (0.0,float(y),0.0))
                    y_section = trimesh.intersections.slice_mesh_plane(y_section,
                                                                        (0.0,-1.0,0.0),
                                                                        (0.0,float(y + ystep),0.0))
                    for x in np.arange(min_bounds[0],max_bounds[0],xstep):
                        x_section = trimesh.intersections.slice_mesh_plane(y_section,
                                                                            (1.0,0.0,0.0),
                                                                            (float(x),0.0,0.0))
                        # print("X SECTION WATERTIGHT?", x_section.is_watertight)
                        # print("Sectioned x_section corners", x_section.bounds)                                                    
                        x_section = trimesh.intersections.slice_mesh_plane(x_section,
                                                                            (-1.0,0.0,0.0),
                                                                            (float(x+xstep),0.0,0.0))
                        # z_section = trimesh.Trimesh(vertices=zvertices, faces=zfaces)

                        
                        # cont = False
                        # for vals in range(3):
                        #     rootmin = rootmeshbounds[0][vals]
                        #     zmin = z_bounds[0][vals]
                        #     rootmax = rootmeshbounds[1][vals]
                        #     zmax = z_bounds[1][vals]
                        #     print("ROOT BOUNDS", rootmin, rootmax)
                        #     print("Z BOUNDS", zmin, zmax)
                        #     if zmin < rootmin:
                        #         cont = True
                        #         break
                        #     if zmax > rootmax:
                        #         cont = True
                        #         break
                        # if cont == True:
                        #     print("CONTINUED")
                        #     continue
                            
                        # print((x-min_bounds[1]) / xstep, (y-min_bounds[0]) / ystep, (z-min_bounds[2]) / zstep)
                        # print(len(z_section.vertices))
                        # if fragcount == 0 and i ==0:
                        #     continue
                        if len(x_section.vertices) == 0:
                            continue
                        fragment_positions[-1].append([       
                            (x-min_bounds[0]) / xstep,
                            (y-min_bounds[1]) / ystep,
                            (z-min_bounds[2]) / zstep,
                        ])
                        
                        # xyz = ["x", "y", "z"]
                        # xyzcors = [x,y,z]
                        # steps = [xstep, ystep, zstep]
                        # for c in range(3):
                        #     cor = (min_bounds[c] + ((xyzcors[c] * steps[c]) * (2**i)))
                        #     print("COR", xyz[c], cor)
                        zmin_bounds = x_section.vertices.min(axis=0)
                        zmax_bounds = x_section.vertices.max(axis=0)
                        # # print("VERTICES MIN AND MAX POINTS", i, fragcount)
                        # # # print("XYZ", x, y, z)
                        # print("MINIMUM",zmin_bounds)
                        # print("MAXIMUM",zmax_bounds)
                        
                        # fragpositions_tomatch = fragment_positions[-1][-1]
                        # # print("FRAGMENT POSITIONS")
                        # # print(fragpositions_tomatch)
                        # xshift = 0
                        # yshift = 0
                        # zshift = 0
                        # if fragpositions_tomatch[0] == 0:
                        #     xshift = (minrootbounds[0] - zmin_bounds[0])
                        # else:
                        #     xshift = x - zmin_bounds[0]
                        # if fragpositions_tomatch[1] == 0:
                        #     yshift = (minrootbounds[1] - zmin_bounds[1])
                        # else:
                        #     yshift = y - zmin_bounds[1]
                        # if fragpositions_tomatch[2] == 0:
                        #     zshift = (minrootbounds[2] - zmin_bounds[2])
                        # else:
                        #     zshift = z - zmin_bounds[2]
                        # shift = np.asarray([[1, 0, 0, xshift],
                        #                     [0, 1, 0, yshift],
                        #                     [0, 0, 1, zshift],
                        #                     [0, 0, 0,  1]])
                        # x_section.apply_transform(shift)

                        zmin_bounds = x_section.bounds[0]
                        zmax_bounds = x_section.bounds[1]
                        # print("AFTER TRANSFORMING", i, fragcount)
                        # print("XYZ", x, y, z)
                        # print("MINIMUM",zmin_bounds)
                        # print("MAXIMUM",zmax_bounds)
                        # print(" ")
                        scale =  np.asarray([xstep,ystep,zstep,1]) / ((2 ** bit_depth) - 1)
                        # print("SCALE", scale)
                        transform = np.asarray([[1, 0, 0, -x/xstep],
                                                [0, 1, 0, -y/ystep],
                                                [0, 0, 1, -z/zstep],
                                                [0, 0, 0,  1]]) / scale
                        x_section.apply_transform(transform)
                        # trimesh.repair.broken_faces(z_section)
                        # trimesh.repair.fill_holes(z_section)
                        # trimesh.repair.fix_inversion(z_section)
                        # trimesh.repair.fix_winding(z_section)
                        # trimesh.repair.fix_normals(z_section)
                        # print("Watertight?", z_section.is_watertight)
                        # print("")

                        # print("LEVEL OF DETAIL", i)
                        drcfile = output_path.joinpath(str(iden))
                        with open(str(drcfile), "ab+") as draco:
                            start = draco.tell()
                            # print('draco.tell: {}'.format(draco.tell()))
                            writethis = DracoPy.encode_mesh_to_buffer(points=x_section.vertices.flatten(),
                                                                      faces = x_section.faces.flatten(),
                                                                      quantization_bits=bit_depth,
                                                                      compression_level=0)
                            # draco.write(trimesh.exchange.ply.export_draco(mesh=x_section, bits=bit_depth)) # bit must match vertex_quantization_bits
                            draco.write(writethis)
                            num_fragments_per_lod[-1] += 1
                            fragment_offsets[-1].append(draco.tell() - start)
                            fragcount = fragcount + 1
                            # print("Size of file: ",draco.tell())

        
        num_fragments_per_lod = np.asarray(num_fragments_per_lod).astype('<I')
        gridorigin = min_bounds
        manifest_file = output_path.joinpath((str(iden)+".index"))
        vertex_offsets = np.asarray(vertex_offsets).astype('<f')
        with open(str(manifest_file), 'wb') as index:
            index.write(chunk_shape.astype('<f').tobytes(order='C'))
            index.write(gridorigin.astype('<f').tobytes(order="C"))
            index.write(struct.pack("<I",num_lods))
            index.write(np.asarray(lod_scales).astype('<f').tobytes(order="C"))
            index.write(vertex_offsets.tobytes(order="C"))
            index.write(num_fragments_per_lod.astype('<I').tobytes(order="C"))

            for i in range(0, num_lods):
                fp = np.asarray(fragment_positions[i]).astype('<I')
                fp = fp.T
                # print("FRAGMENT POSITION")
                # print(fp)
                index.write(fp.astype('<I').tobytes(order="C"))
                fo = np.asarray(fragment_offsets[i]).astype('<I')
                index.write(fo.tobytes(order="C"))

except Exception as e:
    traceback.print_exc()
finally:
    jutil.kill_vm()
