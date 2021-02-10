import os, struct, json
import trimesh
import DracoPy
import numpy as np
from functools import cmp_to_key
import pandas as pd
from neurogen import encoder


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

def final_mesh_decomposition(mesh, nodes_per_dim, quantization_bits, lodframe):

    scale = nodes_per_dim/(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    verts_scaled = scale*(mesh.vertices - mesh.vertices.min(axis=0))

    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = verts_scaled

    # create submeshes. 
    submeshes = []
    nodes = []

    # Define plane normals and scale mesh.
    nyz, nxz, nxy = np.eye(3)
    # print("LODFRAME")
    # print(lodframe)
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
                if (len(mesh_z.vertices) > 0):
                    condx = (lodframe["x"] == x)
                    condy = (lodframe["y"] == y)
                    condz = (lodframe["z"] == z)
                    if ((condx & condy) & condz).any():
                        if (lodframe.loc[((condx & condy)&condz), "lessthan"]).item() == True:
                            mesh_z.vertices = quantizer(mesh_z.vertices)
                            submeshes.append(mesh_z)
                            nodes.append([x,y,z])

    return nodes,submeshes

def generate_mesh_decomposition(mesh, nodes_per_dim, quantization_bits, lodframe):
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
    # print("LODFRAME")
    # print(lodframe)
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
                if (len(mesh_z.vertices) > 0):
                    # condx = (x in lodframe["x"].to_list())
                    # condy = (y in lodframe["y"].to_list())
                    # condz = (z in lodframe["z"].to_list())
                    # lodframe[(condx & condy)&condz]
                    condx = (lodframe["x"] == x)
                    condy = (lodframe["y"] == y)
                    condz = (lodframe["z"] == z)
                    if ((condx & condy) & condz).any():
                        # if (lodframe.loc[((condx & condy)&condz), "lessthan"]).item() == False:
                        mesh_z.vertices = quantizer(mesh_z.vertices)
                        submeshes.append(mesh_z)
                        nodes.append([x,y,z])
                        # lodframe[((condx & condy) & condz), "draco"] = 6
                        # print(lodframe)
    # Sort in Z-curve order
    # if nodes:
    # submeshes, nodes = zip(*sorted(zip(submeshes, nodes), key=cmp_to_key(lambda x, y: cmp_zorder(x[1], y[1]))))


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


def calculate_numlods(meshvertices, lod):

    reachlevel = False
    minvertices = 13000
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

        check_cond = (count_lod["Count"] > minvertices) & (count_lod["LOD"] == lod)
        if (check_cond).any():
            lod = lod+1
            check_lod = count_lod[check_cond]
        else:
            lod = lod+1
            reachlevel = True
    count["lessthan"] = np.where(count["Count"] > minvertices, False, True)
    count.loc[:, 'draco'] = 0
    count = count.reset_index(drop=True)
    return minvertices, lod, count

def generate_multires_mesh(
    mesh, 
    directory, 
    segment_id,
    minimum,
    dataframe,
    num_lods,
    transformation_matrix=None,
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

    if transformation_matrix is None:
        transformation_matrix = np.array([1,0,0,0,0,1,0,0,0,0,1,0]).reshape(3,4)

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
    filez = open(os.path.join(mesh_dir, f'{segment_id}'), 'x')
    with open(os.path.join(mesh_dir, f'{segment_id}'), 'r+b') as f:
        ## We create scales from finest to coarsest.
        f.seek(0,0)
        for scale in lod_scales:
            lod = np.log2(scale)
            lodframe = dataframe[(dataframe["LOD"] == lod)]
            # print("LOD {} has scale value of {}".format(lod, scale))
            # Decimate mesh and clean. Decrease number of faces by scale sqaured.
            num_faces = int(mesh.faces.shape[0]//(lod_scales.max()/scale)**2)
            scaled_mesh = mesh.simplify_quadratic_decimation(num_faces)
            scaled_mesh.remove_degenerate_faces()
            scaled_mesh.remove_duplicate_faces()
            scaled_mesh.remove_unreferenced_vertices()
            scaled_mesh.remove_infinite_values()
            scaled_mesh.fill_holes()
            
            nodes, submeshes = generate_mesh_decomposition(scaled_mesh, scale, quantization_bits, lodframe)
            lod_offsets = []

            f.seek(0,0)
            content = f.read()
            f.seek(0,0)

            for submesh in range(len(submeshes)):
                if len(submeshes[submesh].vertices) > 0: # Only write non-empty meshes.
                    draco = encoder.encode_mesh(submeshes[submesh],compression_level=compression_level)
                    f.write(draco)
                    val = nodes[submesh]
                    dataframe.loc[(dataframe.LOD == lod) & (dataframe.x == val[0]) & (dataframe.y == val[1]) & (dataframe.z == val[2]), 'draco'] = len(draco)
                    lod_offsets.append(len(draco))
                else:
                    lod_offsets.append(0)

            f.write(content)

            fragment_positions.append(np.array(nodes))
            fragment_offsets.append(np.array(lod_offsets))
        
        # with open(os.path.join(mesh_dir, f'{segment_id}'), 'r+b') as f:
            
        #     lodframe_true = dataframe[(dataframe["lessthan"]==True)]
        #     leftover_lods = lodframe_true["LOD"].unique()
        #     for lod in leftover_lods:
        #         scale = 2**lod
                
        #         nodes, submeshes = final_mesh_decomposition(mesh, scale, quantization_bits, lodframe=lodframe_true[(lodframe_true["LOD"] == lod)])
        #         for submesh in range(len(submeshes)):
        #             if len(submeshes[submesh].vertices) > 0: # Only write non-empty meshes.
        #                 val = nodes[submesh]
        #                 conditions = (dataframe.LOD == lod) & (dataframe.x == val[0]) & (dataframe.y == val[1]) & (dataframe.z == val[2])
        #                 indexofrow = dataframe[conditions].index.values[0]
        #                 offset = dataframe.iloc[:indexofrow,6].sum()
        #                 f.seek(0,0)
        #                 firsthalf = f.read(offset)
        #                 secondhalf = f.read()
        #                 draco = encoder.encode_mesh(submeshes[submesh],compression_level=compression_level)

        #                 f.seek(0,0)
        #                 f.write(firsthalf)
        #                 f.write(draco)
        #                 f.write(secondhalf)

        #                 # print("VALUE {} is located at {} index and has offset of {}".format(val, indexofrow, offset))
        #                 dataframe.loc[conditions, 'draco'] = len(draco)

    num_fragments_per_lod = dataframe.groupby("LOD").count()["x"].to_list()
    num_fragments_per_lod = np.flip(np.asarray(num_fragments_per_lod))

    print("SUM: {}".format(dataframe["draco"].sum()))
    # Add mesh subdir to the main info file.
    with open(os.path.join(directory, 'info'), 'r+') as f:
        info = json.loads(f.read())
        f.seek(0)
        info['mesh'] = mesh_subdirectory
        json.dump(info, f)
    
    # Write manifest file.
    with open(os.path.join(mesh_dir, f'{segment_id}.index'), 'wb') as f:
        f.write(chunk_shape.astype('<f').tobytes())
        f.write(grid_origin.astype('<f').tobytes())
        f.write(struct.pack('<I', num_lods))
        f.write(lod_scales.astype('<f').tobytes())
        f.write(vertex_offsets.astype('<f').tobytes(order='C'))
        f.write(num_fragments_per_lod.astype('<I').tobytes())
        for lod in reversed(range(num_lods)):
            lod_frame = dataframe[(dataframe["LOD"] == lod)]
            x = lod_frame["x"].to_list()
            y = lod_frame["y"].to_list()
            z = lod_frame["z"].to_list()
            off = lod_frame["draco"].to_list()
            f.write(np.asarray(x).astype('<I').tobytes(order='C'))
            f.write(np.asarray(y).astype('<I').tobytes(order='C'))
            f.write(np.asarray(z).astype('<I').tobytes(order='C'))
            f.write(np.asarray(off).astype('<I').tobytes(order='C'))
        #     offset = lodframe["draco"].to_list()
        # zipped = zip(reversed(fragment_positions), reversed(fragment_offsets))
        # # zipped.reverse()
        # for frag_pos, frag_offset in zipped:
        #     f.write(frag_pos.T.astype('<I').tobytes(order='C'))
        #     f.write(frag_offset.astype('<I').tobytes(order='C'))
    
    # Write mesh info file. We override the file here. 
    # But that's fine since it never changes.
    with open(os.path.join(mesh_dir, 'info'), 'w') as f:
        info = {
            '@type': 'neuroglancer_multilod_draco',
            'vertex_quantization_bits': quantization_bits,
            'transform': transformation_matrix.flatten().tolist(),
            'lod_scale_multiplier': 1
        }
        json.dump(info, f)