import os, struct, json
from pathlib import Path
import trimesh
from neurogen import encoder
import numpy as np
from functools import cmp_to_key
import logging, argparse, time
import traceback

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("decomposition")
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

    def __init__(self, fragment_origin, fragment_shape, input_origin, quantization_bits):
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
        self.upper_bound = np.iinfo(np.uint32).max >> (np.dtype(np.uint32).itemsize*8 - quantization_bits)
        self.scale = self.upper_bound / fragment_shape
        self.offset = input_origin - fragment_origin + 0.5/self.scale
    
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


def generate_mesh_decomposition(mesh, nodes_per_dim, quantization_bits):
    """Decomposes and quantizes a mesh according to the desired number of nodes and bits.
    
    A mesh is decomposed into a set of submeshes by partitioning the bounding box into
    nodes_per_dim**3 equal subvolumes . The positions of the vertices within 
    each subvolume are quantized according to the number of bits specified. The nodes 
    and corresponding submeshes are sorted along a z-curve.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    nodes_per_dim : int
        Number of nodes along each dimension.
    quantization_bits : int
        Number of bits for quantization. Should be 10 or 16.
    
    Returns
    -------
    nodes : list
        List of z-curve sorted node coordinates corresponding to each subvolume. 
    submeshes : list
        List of z-curve sorted meshes.
    """

    # Scale our mesh coordinates.
    scale = nodes_per_dim/(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    verts_scaled = scale*(mesh.vertices - mesh.vertices.min(axis=0))

    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = verts_scaled

    # Define plane normals and scale mesh.
    nyz, nxz, nxy = np.eye(3)
    
    # create submeshes. 
    submeshes = []
    nodes = []
    for x in range(0, nodes_per_dim):
        mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
        mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
        for y in range(0, nodes_per_dim):
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
            mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
            for z in range(0, nodes_per_dim):
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
                
                # Initialize Quantizer.
                quantizer = Quantize(
                    fragment_origin=np.array([x, y, z]), 
                    fragment_shape=np.array([1, 1, 1]), 
                    input_origin=np.array([0,0,0]), 
                    quantization_bits=quantization_bits
                )
    
                if len(mesh_z.vertices) > 0:
                    mesh_z.vertices = quantizer(mesh_z.vertices)
                    submeshes.append(mesh_z)
                    nodes.append([x,y,z])
    
    # Sort in Z-curve order
    submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))
            
    return nodes, submeshes

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

def generate_multires_mesh(
    mesh, 
    directory, 
    segment_id, 
    num_lods,
    quantization_bits=16, 
    compression_level=5, 
    mesh_subdirectory='meshdir'):
    """ Generates a Neuroglancer precomputed multiresolution mesh.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    directory : str
        Neuroglancer precomputed volume directory.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    num_lods : int
        Number of levels of detail to generate.
    transformation_matrix: np.ndarray
        A 3x4 numpy array representing a coordinate transform matrix. 
        If None, identity is used.
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.    
    """


    logger.info("(ID: {}) {} Level of Details".format(segment_id, num_lods))
    # Define key variables. 
    lods = np.arange(0, num_lods)
    chunk_shape = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))/2**lods.max()
    grid_origin = mesh.vertices.min(axis=0)
    lod_scales = np.array([2**lod for lod in lods])
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])

    # Clean up mesh.
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fill_holes()

    # Create directory
    mesh_dir = os.path.join(directory, mesh_subdirectory)
    os.makedirs(mesh_dir, exist_ok=True)

    fragment_offsets = []
    fragment_positions = []
    # Write fragment binaries.
    with open(os.path.join(mesh_dir, f'{segment_id}'), 'wb') as f:
        ## We create scales from finest to coarsest.
        for scale in lod_scales[::-1]:

            level = np.log2(scale)

            # Decimate mesh and clean. Decrease number of faces by scale sqaured.
            divideby_faces = (lod_scales.max()/scale)**2
            num_faces = int(mesh.faces.shape[0]//divideby_faces)
            scaled_mesh = mesh.simplify_quadratic_decimation(num_faces)
            scaled_mesh.remove_degenerate_faces()
            scaled_mesh.remove_duplicate_faces()
            scaled_mesh.remove_unreferenced_vertices()
            scaled_mesh.remove_infinite_values()
            scaled_mesh.fill_holes()
            
            nodes, submeshes = generate_mesh_decomposition(scaled_mesh, scale, quantization_bits)

            lod_offsets = []
            for submesh in submeshes:
                # Only write non-empty meshes.
                if len(submesh.vertices) > 0:
                    draco = encoder.encode_mesh(submesh,compression_level=compression_level)
                    f.write(draco)
                    lod_offsets.append(len(draco))
                else:
                    lod_offsets.append(0)

            fragment_positions.append(np.array(nodes))
            fragment_offsets.append(np.array(lod_offsets))
            logger.info(("(ID: {}) Level {} is simplified by {}".format(segment_id, level + 1, divideby_faces)))

    num_fragments_per_lod = np.array([len(nodes) for nodes in fragment_positions])

    try:
        # Write manifest file.
        indexfile = Path(mesh_dir).joinpath(f'{segment_id}.index')
        with open(indexfile, 'ab') as f:
            f.write(chunk_shape.astype('<f').tobytes())
            f.write(grid_origin.astype('<f').tobytes())
            f.write(struct.pack('<I', num_lods))
            f.write(lod_scales.astype('<f').tobytes())
            f.write(vertex_offsets.astype('<f').tobytes(order='C'))
            f.write(num_fragments_per_lod.astype('<I').tobytes())
            for frag_pos, frag_offset in zip(fragment_positions, fragment_offsets):
                f.write(frag_pos.T.astype('<I').tobytes(order='C'))
                f.write(frag_offset.astype('<I').tobytes(order='C'))
        f.close()
    except Exception as e:
        traceback.print_exc()