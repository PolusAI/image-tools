from skimage import measure
# from skimage.measure import marching_cubes
from pathlib import Path
# from bfio import BioReader
import neuroglancer
import numpy as np
import struct,json
import argparse, logging, subprocess, time, multiprocessing
from bfio import BioReader, BioWriter, JARS
import bioformats
import javabridge as jutil
from pathlib import Path
import utils
import filepattern
from filepattern import FilePattern as fp
import itertools
import numpy as np
import os
import neuroglancer

TILE_SIZE = 64
out_path = Path('neuroglancer').joinpath('output').joinpath('dA30_5_dA30.Labels.ome.tif').joinpath('meshdir')
# with open(out_path.joinpath('mesh.1.1'),'rb') as file:
#     num_vertices = struct.unpack('<I',file.read(4))
#     print('4*3*num_vertices: {}'.format(4*3*num_vertices[0]))
#     # buf = file.read(4*3*num_vertices[0])
#     # print('len(buf): {}'.format(len(buf)))
#     print(struct.unpack('<f',file.read(4)))
#     print(struct.unpack('<f',file.read(4)))
#     print(struct.unpack('<f',file.read(4)))
#     print(struct.unpack('<f',file.read(4)))
#     print(struct.unpack('<f',file.read(4)))
#     print(struct.unpack('<f',file.read(4)))
# quit()
# file_path = Path('input_volume').joinpath('dA30_5_dA30.Labels.ome.tif')
file_path = Path('/home/ec2-user/LabelingData').joinpath('dA30_5_dA30.Labels.ome.tif')
out_path = Path('neuroglancer').joinpath('output').joinpath('dA30_5_dA30.Labels.ome.tif').joinpath('meshdir')
out_path.mkdir(exist_ok=True)

log_config = Path(__file__).parent.joinpath("log4j.properties")
jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
try:
    br = BioReader(str(file_path))
    # voxel_size = [325,325,325]
    voxel_size = [np.float32(325), np.float32(325), np.float32(325)]
    # print(voxel_size)
    # quit()
    volume = br.read_image()[...,0,0]
    # print(br.physical_size_x())
    vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==1)
    vertices[:,0] *= voxel_size[0]
    vertices[:,1] *= voxel_size[1]
    vertices[:,2] *= voxel_size[2]
    print((volume==1).sum())
    # print(vertices[0])
    # vertices = vertices[:,[1,0,2]]
    # print(vertices[0])
    # print(volume.sum())
    print(vertices.shape)
    # print(faces.shape)
    print(vertices[faces].shape)
    # print(faces[0])
    # print(vertices[faces[0]])
    # print(vertices[faces[0]].reshape(-1))
    # plt.figure()
    # plt.imshow(volume[:,:,44])
    # plt.show()
    json_descriptor = '{{"fragments": ["mesh.{}.{}"]}}'
    dim=neuroglancer.CoordinateSpace(
                names=['x', 'y', 'z'],
                units=['m', 'm', 'm'],
                scales=voxel_size)
    vol = neuroglancer.LocalVolume(data=volume, dimensions=dim)
    ids = np.unique(volume)
    # print(ids)
    chunk = 0
    fragments = {}
    for z in range(0,br.num_z(),TILE_SIZE):
        z_max = min([z+TILE_SIZE,br.num_z()])
        print(z)
        for y in range(0,br.num_y(),TILE_SIZE):
            y_max = min([y+TILE_SIZE,br.num_y()])
            for x in range(0,br.num_x(),TILE_SIZE):
                x_max = min([x+TILE_SIZE,br.num_x()])
                volume = br.read_image(X=[x,x_max],Y=[y,y_max],Z=[z,z_max])[...,0,0]
                # vol = neuroglancer.LocalVolume(data=volume, dimensions=dim,mesh_options={"offset":[np.float32(x),np.float32(y),np.float32(z)]})
                ids = np.unique(volume)
                for ID in ids:
                    ID = int(ID)
                    if ID==0:
                        continue
                    try:
                        vertices,faces,_,_ = measure.marching_cubes_lewiner(volume==ID,step_size=1)
                    except RuntimeError:
                        continue
                    vertices = vertices[:,[1,0,2]].astype(np.float32)
                    vertices[:,0] = vertices[:,0]*voxel_size[0] + voxel_size[0]*np.float32(x)
                    vertices[:,1] = vertices[:,1]*voxel_size[1] + voxel_size[1]*np.float32(y)
                    vertices[:,2] = vertices[:,2]*voxel_size[2] + voxel_size[2]*np.float32(z)
                    print('ID: {}'.format(ID))
                    print('shape: {}'.format(vertices.shape))
                    print(chunk)
                    # mesh_data = vol.get_object_mesh(ID)
                    # print(type(mesh_data))
                    # num_vertices = struct.unpack('<I',mesh_data[:4])
                    # print('num_vertices: {}'.format(num_vertices[0]))
                    fragment_file = out_path.joinpath('mesh.{}.{}'.format(ID,chunk))
                    with open(str(fragment_file), 'wb') as meshfile:
                        meshfile.write(struct.pack("<I",vertices.shape[0]))
                        meshfile.write(vertices.astype('<f').tobytes(order="C"))
                        meshfile.write(faces.astype("<I").tobytes(order="C"))
                        # meshfile.write(mesh_data)
                    if ID not in fragments:
                        fragments[ID] = {'fragments': [fragment_file.name]}
                    else:
                        fragments[ID]['fragments'].append(fragment_file.name)
                chunk += 1
            print("DONE")
            # quit()
    for ID,frag in fragments.items():
        with open(
                str(out_path.joinpath('{}:0'.format(ID))), 'w') as ff:
                        ff.write(json.dumps(frag))
except Exception as e:
    jutil.kill_vm()
    print(e)
    