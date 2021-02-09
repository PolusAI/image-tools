import os, struct, json
import trimesh
from neurogen import encoder as encodethismesh
import numpy as np
from functools import cmp_to_key
from pathlib import Path
import logging
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("mesh-generation")
logger.setLevel(logging.INFO)

class Quantize():
    """
    A class used to quantize mesh vertex positions for Neuroglancer precomputed
    meshes to a specified number of bits.
    
    Based on the C++ code provided here: https://github.com/google/neuroglancer/issues/266#issuecomment-739601142

    Attributes
    ----------
    upper_bound : int 
        The largest integer used to represent a vertex position.
    scale : np.ndarray
        Array containing the scaling factors for each dimension. 
    offset : np.ndarray
        Array containing the offset values for each dimension. 
    """

    def __init__(self, fragment_origin, fragment_shape, input_origin, quantization_bits, lod):
        """
        Parameters
        ----------
        fragment_origin : np.ndarray
            Minimum input vertex position to represent.
        fragment_shape : np.ndarray
            The inclusive maximum vertex position to represent is `fragment_origin + fragment_shape`.
        input_origin : np.ndarray
            The offset to add to input vertices before quantizing them.
        quantization_bits : int
            The number of bits to use for quantization.
        """

        self.upper_bound = np.iinfo(np.uint32).max >> (np.dtype(np.uint32).itemsize*8 - quantization_bits) # if 10 then 1023, if 16 then 65535
        self.scale = self.upper_bound / fragment_shape
        # self.offset = input_origin - fragment_origin #+ 0.5/self.scale
        # self.offset = 0

        # print("\t"*lod, "UPPER BOUND: ", self.upper_bound)
        # print("\t"*lod, "SCALE: ", self.scale)
        # print("\t"*lod, "OFFSET: ", self.offset)

        # self.scale = self.upper_bound
        # self.offset = input_origin - fragment_origin + (fragment_shape[0]/(2**(lod)))/self.scale
        
        self.offset = input_origin - fragment_origin + 0.5/self.scale
        # self.offset = 0
    
    def __call__(self, vertices):
        """ Quantizes an Nx3 numpy array of vertex positions.
        
        Parameters
        ----------
        vertices : np.ndarray
            Nx3 numpy array of vertex positions.
        
        Returns
        -------
        np.ndarray
            Quantized vertex positions.
        """
        
        output = np.minimum(self.upper_bound, np.maximum(0, self.scale*(vertices + self.offset))).astype(np.uint32)
        # output = np.minimum(self.upper_bound, self.scale*(vertices + self.offset)).astype(np.uint32)
        # output = (self.scale*(vertices)).astype(np.uint32)
        return output
 

def cmp_zorder(lhs, rhs):
    """Compare z-ordering
    
    Code taken from https://en.wikipedia.org/wiki/Z-order_curve
    """
    def less_msb(x: int, y: int):
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


# def generate_mesh_decomposition(mesh, nodes_per_dim, quantization_bits, nodearray, frag, lod, maxvertex, minvertex):
#     """Decomposes and quantizes a mesh according to the desired number of nodes and bits.
    
#     A mesh is decomposed into a set of submeshes by partitioning the bounding box into
#     nodes_per_dim**3 equal subvolumes . The positions of the vertices within 
#     each subvolume are quantized according to the number of bits specified. The nodes 
#     and corresponding submeshes are sorted along a z-curve.
    
#     Parameters
#     ----------
#     mesh : trimesh.base.Trimesh 
#         A Trimesh mesh object to decompose.
#     nodes_per_dim : int
#         Number of nodes along each dimension.
#     quantization_bits : int
#         Number of bits for quantization. Should be 10 or 16.
    
#     Returns
#     -------
#     nodes : list
#         List of z-curve sorted node coordinates corresponding to each subvolume. 
#     submeshes : list
#         List of z-curve sorted meshes.
#     """

#     # Scale our mesh coordinates.
#     maxvertex = mesh.vertices.max(axis=0)
#     minvertex = mesh.vertices.min(axis=0)
#     nodearray = nodearray
#     scale = nodearray/(maxvertex- minvertex)
#     verts_scaled = scale*(mesh.vertices - minvertex) #the scaled vertices ranges from 0 to chunk_shape
#     scaled_mesh = mesh.copy()
#     scaled_mesh.vertices = verts_scaled

#     # Define plane normals and scale mesh.
#     nyz, nxz, nxy = np.eye(3)
#     res = [i*j for i,j in zip([1,1,1], nodearray)]
#     # create submeshes. 
#     submeshes = []
#     nodes = []
#     for x in range(0, nodearray[0]):
#         mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
#         mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
#         for y in range(0, nodearray[1]):
#             mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
#             mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
#             for z in range(0, nodearray[2]):
#                 mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
#                 mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
#                 # Initialize Quantizer.
#                 quantizer = Quantize(
#                     fragment_origin=np.array([x, y, z]), 
#                     fragment_shape=np.array(frag), 
#                     input_origin=np.array([0,0,0]), 
#                     quantization_bits=quantization_bits,
#                     lod = lod
#                 )
    
#                 if len(mesh_z.vertices) > 0:
#                     mesh_z.vertices = quantizer(mesh_z.vertices)
#                     submeshes.append(mesh_z)
#                     nodes.append([x,y,z])
    
#     # Sort in Z-curve order
#     submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))
            
#     return nodes, submeshes

def generate_trimesh_chunks(
    mesh,
    directory,
    segment_id,
    chunks):
    """Generates temporary chunks of the meshes by saving them in ply files

    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    directory : str
        Temporary directory to save the ply files
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    chunks: tuple
        The X, Y, Z chunk that is analyzed
    """
    chunk_filename = '{}_{}_{}_{}.ply'.format(segment_id, chunks[0], chunks[1], chunks[2])
    temp_dir = os.path.join(directory, "temp_drc")
    os.makedirs(temp_dir, exist_ok=True)
    mesh.export(os.path.join(temp_dir, chunk_filename))

def generate_manifest_file(
    meshbounds,
    segment_id,
    directory,
    num_lods,
    fragment_info,
    mesh_subdirectory = 'meshdir'
):

    grid_origin = meshbounds[0]
    shape = meshbounds[1] - meshbounds[0]

    chunk_shape = shape/[(2**(num_lods-1)) for i in range(3)]
    print(chunk_shape)
    vertex_offsets = np.zeros((num_lods,3)) 
    # vertex_offsets = np.asarray([0.5, 0.5, 0.5]) #,
    #                             #  [1.0, 1.0, 1.0],
    #                             #  [2.0, 2.0, 2.0]])
    print(vertex_offsets)
    lod_scales = np.asarray([2**i for i in range(num_lods)])
    num_fragments_per_lod = np.asarray([len(fragment_info[i]) for i in reversed(range(num_lods))])
    print(num_fragments_per_lod)


    mesh_dir = os.path.join(directory, mesh_subdirectory)
    with open(os.path.join(mesh_dir, f'{segment_id}.index'), 'wb') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())
        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))
        f.write(num_fragments_per_lod.astype('<I').tobytes())
        for lod in reversed(range(num_lods)):
            print("NEW LOD {}".format(lod))
            fragpositions = []
            fragoffsets = []
            for frag_info in fragment_info[lod]:
                fragpositions.append(frag_info)
                fragoffsets.append(fragment_info[lod][frag_info])
            fragpositions = np.asarray(fragpositions).T
            for frag_pos in fragpositions:
                f.write(np.asarray(frag_pos).astype('<I').tobytes(order='C'))
            for frag_off in fragoffsets:
                f.write(np.asarray(frag_off).astype('<I').tobytes(order='C'))
            # print(frag_info, fragment_info[lod][frag_info])
                
            # for frag_pos in np.asarray(fragment_positions[lod]).T:
            #     f.write(np.asarray(frag_pos).astype('<I').tobytes(order='C'))
            # for frag_off in fragment_offsets[lod]:
            #     f.write(np.asarray(frag_off).astype('<I').tobytes(order='C'))

def calculate_numlods(meshvertices, lod):

    reachlevel = False
    minvertices = 100000
    num_vertices = len(meshvertices)
    # if num_vertices < minvertices:
    #     reachlevel=True


    maxvertex = meshvertices.max(axis=0)
    minvertex = meshvertices.min(axis=0)

    mesh_df = pd.DataFrame(data=meshvertices, columns="x y z".split())
    count = pd.DataFrame(data=[[0, 0, 0, 0, num_vertices]], columns = "x y z LOD Count".split())
    check_lod = pd.DataFrame(columns = "x y z".split())
    while reachlevel == False:
        numsplits = int((2**lod)+1)
        xsplits = np.linspace(start=minvertex[0], stop=maxvertex[0], num=numsplits)
        ysplits = np.linspace(start=minvertex[1], stop=maxvertex[1], num=numsplits)
        zsplits = np.linspace(start=minvertex[2], stop=maxvertex[2], num=numsplits)

        new_mesh_df = mesh_df.copy()
        for x in range(numsplits-1):
            for y in range(numsplits-1):
                for z in range(numsplits-1):
                    condx = (mesh_df["x"] >= xsplits[x]) & (mesh_df["x"] < xsplits[x+1])
                    condy = (mesh_df["y"] >= ysplits[y]) & (mesh_df["y"] < ysplits[y+1])
                    condz = (mesh_df["z"] >= zsplits[z]) & (mesh_df["z"] < zsplits[z+1])

                    new_mesh_df.loc[condx,"x"] = x
                    new_mesh_df.loc[condy,"y"] = y
                    new_mesh_df.loc[condz,"z"] = z

        new_mesh_df.loc[mesh_df["x"]==xsplits[-1],"x"] = x
        new_mesh_df.loc[mesh_df["y"]==ysplits[-1],"y"] = y
        new_mesh_df.loc[mesh_df["z"]==zsplits[-1],"z"] = z
        
        count_lod = new_mesh_df.groupby(["x", "y", "z"]).size().reset_index(name='Count')
        count_lod["LOD"] = lod
        if lod > 0:
            for i in range(len(check_lod.index)):
                xcheck = check_lod["x"].iloc[i]
                ycheck = check_lod["y"].iloc[i]
                zcheck = check_lod["z"].iloc[i]

                xcond = (count_lod["x"] >= xcheck*2) & (count_lod["x"] < (xcheck*2)+2)
                ycond = (count_lod["y"] >= ycheck*2) & (count_lod["y"] < (ycheck*2)+2)
                zcond = (count_lod["z"] >= zcheck*2) & (count_lod["z"] < (zcheck*2)+2)
                append = count_lod[xcond & ycond & zcond]

                count = count.append(append)
        # append_cond = count_lod["x"] > check_lod 
        # count = count.append(count_lod)
        # print(count)
        check_cond = (count_lod["Count"] > minvertices) & (count_lod["LOD"] == lod)
        if (check_cond).any():
            lod = lod+1
            check_lod = count_lod[check_cond]
        else:
            lod = lod+1
            reachlevel = True
    count = count.reset_index()
    print(count)
    return lod, count

def generate_multires_mesh(
    mesh, 
    directory, 
    segment_id,
    dataframe,
    num_lods,
    quantization_bits=16,
    compression_level=4,
    mesh_subdirectory='meshdir',
    quantizer=None):
    
    """ Generates a Neuroglancer precomputed multiresolution mesh.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    directory : str
        Neuroglancer precomputed volume directory.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.    
    # """

    mesh_dir = os.path.join(directory, mesh_subdirectory)
    os.makedirs(mesh_dir, exist_ok=True)

    minvertices = 100000
    num_faces = mesh.faces.shape[0]
    maxvertex = mesh.vertices.max(axis=0)
    minvertex = mesh.vertices.min(axis=0)

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fill_holes()

    fragment_offsets = {}
    fragment_positions = {}
    fragment_info = {}
    
    for i in range(num_lods):
        fragment_info[i] = {}
        divide_faces = 2**(num_lods-i-1)
        new_num_faces = np.floor(num_faces/divide_faces)
        print("{} faces are divided by {} to give {} faces for LOD {}".format(num_faces, divide_faces, new_num_faces, i))

        # scale = (2**i)/(maxvertex-minvertex)
        
        

        scaled_mesh = mesh.simplify_quadratic_decimation(new_num_faces)
        scaled_mesh.remove_degenerate_faces()
        scaled_mesh.remove_duplicate_faces()
        scaled_mesh.remove_unreferenced_vertices()
        scaled_mesh.remove_infinite_values()
        scaled_mesh.fill_holes()

        scale = 1/(maxvertex-minvertex)
        verts_scaled = scale*(mesh.vertices - minvertex)
        scaled_mesh = mesh.copy()
        scaled_mesh.vertices = verts_scaled
        print("SCALED MAX: ", scaled_mesh.vertices.max(axis=0))
        print("SCALED MIN: ", scaled_mesh.vertices.min(axis=0))

        # lodframe = dataframe[(dataframe["LOD"] == i)]
        # print("LODFRAME")
        # print(lodframe)
        # xvals = lodframe["x"].to_list()
        # yvals = lodframe["y"].to_list()
        # zvals = lodframe["z"].to_list()        

        nyz, nxz, nxy = np.eye(3)
        for x in range(0,2**i):
        # for x in xvals:
            mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
            mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
            for y in range(0,2**i):
            # for y in yvals:
                mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
                mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
                for z in range(0,2**i):
                # for z in zvals:
                    mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                    mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))

                    quantizer = Quantize(
                        fragment_origin=np.array([x,y,z]), 
                        # fragment_shape=np.array([2**i,2**i,2**i]),
                        fragment_shape=np.array([1,1,1]),
                        input_origin=np.array([0,0,0]), 
                        quantization_bits=quantization_bits,
                        lod = i
                    )

                    if len(mesh_z.vertices) > 0:
                        print("X Y Z loop: {} {} {} has {} vertices".format(x, y, z, len(mesh_z.vertices)))
                        mesh_z.vertices = quantizer(mesh_z.vertices)
                        print("MAX", mesh_z.vertices.max(axis=0))
                        print("MIN", mesh_z.vertices.min(axis=0))
                        mesh_dir = os.path.join(directory, mesh_subdirectory)
                        os.makedirs(mesh_dir, exist_ok=True)

                        # mesh.vertices = quantizer(scaled_mesh.vertices)
                        with open(os.path.join(mesh_dir, f'{segment_id}'), 'ab') as f:
                            draco = encodethismesh.encode_mesh(mesh_z,compression_level=compression_level)
                            f.write(draco)
                            fragment_info[i][(x,y,z)] = len(draco)

    print(fragment_info)
    return fragment_info
    
    # if recur == 0:
    #     mesh.remove_degenerate_faces()
    #     mesh.remove_duplicate_faces()
    #     mesh.remove_unreferenced_vertices()
    #     mesh.remove_infinite_values()
    #     mesh.fill_holes()

    # minvertices = 100000
    # num_of_vertices = len(mesh.vertices)

    # print("\t"*recur, "Recursion {}/{}: {} -- {}".format(recur, num_lods, num_of_vertices,nodearray))
    # if num_of_vertices > 0:
    #     maxvertex = mesh.vertices.max(axis=0)
    #     minvertex = mesh.vertices.min(axis=0)
    #     scale = 2/(maxvertex-minvertex)
    #     verts_scaled = scale*(mesh.vertices - minvertex) 
    #     scaled_mesh = mesh.copy()
    #     scaled_mesh.vertices = verts_scaled #the scaled vertices ranges from 0 to 2
    

    # nyz, nxz, nxy = np.eye(3)
    # for x in range(0,2):
    #     mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
    #     mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
    #     for y in range(0,2):
    #         mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
    #         mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
    #         for z in range(0,2):
    #             mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
    #             mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
    #             print("NUMBER OF VERTICES IN SPLIT {}".format(len(mesh_z.vertices)))
    # #             if num_lods < recur + 1:
    # #                 num_lods = recur + 1
    # #             quantizer = Quantize(
    # #                 fragment_origin=np.array([x,y,z]), 
    # #                 fragment_shape=np.array([2,2,2]),
    # #                 input_origin=np.array([0,0,0]), 
    # #                 quantization_bits=quantization_bits,
    # #                 lod = recur + 1
    # #             )

    # # if recur == 0:
    # #     quantizer = Quantize(
    # #                         fragment_origin=np.array([0,0,0]), 
    # #                         fragment_shape=np.array([2, 2, 2]), 
    # #                         input_origin=np.array([0,0,0]), 
    # #                         quantization_bits=quantization_bits,
    # #                         lod = recur
    # #                     )


    # # if len(mesh.vertices) > 0:
    # #     simplified_mesh = mesh.copy()
    # #     # print("\t"*recur, "MIN Vertices", scaled_mesh.vertices.min(axis=0))
    # #     # print("\t"*recur, "MAX Vertices", scaled_mesh.vertices.max(axis=0))
    # #     num_faces = simplified_mesh.faces.shape[0]
    # #     dividefaces = (num_lods - recur + 1)
    # #     new_num_faces = int(num_faces//dividefaces)

    # #     simplified_mesh.vertices = quantizer(scaled_mesh.vertices)
    # #     print("\t"*recur, "MIN Vertices", simplified_mesh.vertices.min(axis=0))
    # #     print("\t"*recur, "MAX Vertices", simplified_mesh.vertices.max(axis=0))
    # #     simplified_mesh = simplified_mesh.simplify_quadratic_decimation(new_num_faces)

    # #     simplified_mesh.remove_degenerate_faces()
    # #     simplified_mesh.remove_duplicate_faces()
    # #     simplified_mesh.remove_unreferenced_vertices()
    # #     simplified_mesh.remove_infinite_values()
    # #     simplified_mesh.fill_holes()


    # #     print("\t"*recur,"{} is divided by {} for LOD {}".format(num_faces, dividefaces, recur))


    # #     mesh_dir = os.path.join(directory, mesh_subdirectory)
    # #     os.makedirs(mesh_dir, exist_ok=True)

    # #     # mesh.vertices = quantizer(scaled_mesh.vertices)
    # #     with open(os.path.join(mesh_dir, f'{segment_id}'), 'ab') as f:
    # #         draco = encodethismesh.encode_mesh(simplified_mesh,compression_level=compression_level)
    # #         f.write(draco)


    # #     if recur in fragment_offsets.keys():
    # #         update = {int(recur): (fragment_offsets[recur]).append(len(draco))}
    # #     else:
    # #         fragment_offsets[recur] = [len(draco)]
        
    # #     if recur in fragment_positions.keys():
    # #         update = {int(recur): (fragment_positions[recur]).append(nodearray)}
    # #     else:
    # #         fragment_positions[recur] = [(nodearray)]
    # #     print("")

    

