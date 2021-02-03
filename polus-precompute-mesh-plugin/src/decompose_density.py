import os, struct, json
import trimesh
from neurogen import encoder as encodethismesh
import numpy as np
from functools import cmp_to_key
from pathlib import Path
import logging
from collections import defaultdict

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
        self.scale = np.floor(self.upper_bound / fragment_shape)
        # self.scale = self.upper_bound
        self.offset = input_origin - fragment_origin + (fragment_shape/(2**(lod)))/self.scale
        # self.offset = input_origin - fragment_origin + 0.5/self.scale
    
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
    fragment_offsets,
    fragment_positions,
    mesh_subdirectory = 'meshdir'
):

    # maxvertex = meshbounds[1]
    # minvertex = meshbounds[0]
    grid_origin = meshbounds[0]
    shape = meshbounds[1] - meshbounds[0]

    chunk_shape = shape/[(2**(num_lods-1)) for i in range(3)]
    print(chunk_shape)
    vertex_offsets = np.zeros((num_lods,3))
    print(vertex_offsets)
    lod_scales = np.asarray([2**i for i in range(num_lods)])
    num_fragments_per_lod = np.asarray([len(fragment_offsets[i]) for i in reversed(range(num_lods))])
    for i in reversed(range(num_lods)):
        print(i)
        print("LEVEL OF DETAIL {}: FRAGMENT OFFSETS: {}".format(i, fragment_offsets[i]))
        print("LEVEL OF DETAIL {}: FRAGMENT POSITIONS: {}".format(i, fragment_positions[i]))

    mesh_dir = os.path.join(directory, mesh_subdirectory)
    with open(os.path.join(mesh_dir, f'{segment_id}.index'), 'wb') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())
        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))
        f.write(num_fragments_per_lod.astype('<I').tobytes())
        for lod in reversed(range(num_lods)):
            for frag_pos in np.asarray(fragment_positions[lod]).T:
                f.write(np.asarray(frag_pos).astype('<I').tobytes(order='C'))
            for frag_off in fragment_offsets[lod]:
                f.write(np.asarray(frag_off).astype('<I').tobytes(order='C'))

    

def generate_multires_mesh(
    mesh, 
    directory, 
    segment_id, 
    nodearray,
    recur,
    num_lods,
    fragment_offsets,
    fragment_positions,
    scalevalue,
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
    """

    
    # print("ORIGINAL", num_of_vertices)
    # Scale our mesh coordinates.
    

    minvertices = 10000
    num_of_vertices = len(mesh.vertices)

    print("\t"*recur, "Recursion {}: {} -- {}".format(recur, num_of_vertices,nodearray))
    if num_of_vertices > 0:
        maxvertex = mesh.vertices.max(axis=0)
        minvertex = mesh.vertices.min(axis=0)
        # if recur != 0:
        #     scale = 2**(recur+1)/(maxvertex-minvertex)
        #     verts_scaled = scale*(mesh.vertices - minvertex) #the scaled vertices ranges from 0 to chunk_shape
        #     scaled_mesh = mesh.copy()
        #     scaled_mesh.vertices = verts_scaled
        # else:
        scale = 2/(maxvertex-minvertex)
        verts_scaled = scale*(mesh.vertices - minvertex) #the scaled vertices ranges from 0 to chunk_shape
        scaled_mesh = mesh.copy()
        scaled_mesh.vertices = verts_scaled
    
    if minvertices < num_of_vertices:
        nyz, nxz, nxy = np.eye(3)
        for x in range(0,2):
            mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
            mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
            for y in range(0,2):
                mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
                mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
                for z in range(0,2):
                    mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                    mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
                    print(len(mesh_z.vertices))
                    if num_lods < recur:
                        num_lods = recur + 1
                    new_nodearray = [x+(nodearray[0]*2),y+(nodearray[1]*2),z+(nodearray[2]*2)]
                    print("FRAGMENT SHAPE VALUE: {}, scalevalue {}".format(2**(recur+1), scalevalue))
                    quantizer = Quantize(
                        fragment_origin=np.array([0,0,0]), 
                        # fragment_shape=np.array([2**(recur+1), 2**(recur+1), 2**(recur+1)]), 
                        fragment_shape=np.array([1,1,1]),
                        input_origin=np.array(nodearray), 
                        quantization_bits=quantization_bits,
                        lod = recur + 1
                    )
                    
                    num_lods, fragment_offsets, fragment_positions = generate_multires_mesh(mesh=mesh_z, directory=directory, segment_id=segment_id, nodearray = new_nodearray, recur = recur+1, num_lods = num_lods, 
                                            fragment_offsets = fragment_offsets, fragment_positions=fragment_positions, scalevalue = 2**(recur+1),
                                            quantization_bits=quantization_bits, compression_level=compression_level, mesh_subdirectory=mesh_subdirectory, quantizer = quantizer)

    
            
            # fragment_offsets.update(update)
        # print("after", fragment_offsets)
        # print("SIZE OF FILE", os.path.getsize(os.path.join(mesh_dir, f'{segment_id}')))x
    if recur == 0:
        quantizer = Quantize(
                            fragment_origin=np.array([0,0,0]), 
                            fragment_shape=np.array([2, 2, 2]), 
                            input_origin=np.array([0,0,0]), 
                            quantization_bits=quantization_bits,
                            lod = recur
                        )
    if len(mesh.vertices) > 0:
        mesh.vertices = quantizer(scaled_mesh.vertices)
        mesh_dir = os.path.join(directory, mesh_subdirectory)
        os.makedirs(mesh_dir, exist_ok=True)
        # print("MAX VERTICES", mesh.vertices.max(axis=0))
        with open(os.path.join(mesh_dir, f'{segment_id}'), 'ab') as f:
            draco = encodethismesh.encode_mesh(mesh,compression_level=compression_level)
            f.write(draco)
            # print("\t"*recur, "Recursion {}: {} (vertices: {})".format(recur, len(draco),len(mesh.vertices)))
        if recur in fragment_offsets.keys():
            update = {int(recur): (fragment_offsets[recur]).append(len(draco))}
        else:
            fragment_offsets[recur] = [len(draco)]
        
        if recur in fragment_positions.keys():
            update = {int(recur): (fragment_positions[recur]).append(nodearray)}
        else:
            fragment_positions[recur] = [(nodearray)]

    return num_lods, fragment_offsets, fragment_positions

