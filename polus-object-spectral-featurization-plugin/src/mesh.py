import scipy 
import logging
import trimesh
import numpy as np
from typing import Tuple
from bfio import BioReader
from skimage import measure


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('mesh')
logger.setLevel(logging.INFO)


def mesh_spectral_features(vertices, faces, k = 50, scale_invariant = False):
    i = np.concatenate([faces[:,0], faces[:,0], faces[:,1], faces[:,1], faces[:,2], faces[:,2]])
    j = np.concatenate([faces[:,1], faces[:,2], faces[:,0], faces[:,2], faces[:,0], faces[:,1]])
    
    idx = np.c_[i,j]
    idx = np.unique(idx, axis=0)
    
    xi = vertices[idx[:,0]]
    xj = vertices[idx[:,1]]
    
    dij = np.sum((xi - xj)**2, axis=1)
    t = 2*dij.max()
        
    v = np.exp(-dij/t)
    W = scipy.sparse.csr_matrix((v, (idx[:,0], idx[:,1])))
    D = scipy.sparse.diags(np.asarray(W.sum(axis=1)).flatten(), 0)
    L = D - W
        
    # Make sure we calculate enough eigenvalues. First we calculate some extra 
    # eigenvalues in case there are many separate components. 
    max_iter = 10
    enough_eigvals = False
    k_orig = k
    n_singular = 0
    etol = 1e-3
    while enough_eigvals is not True and max_iter > 0:
        
        try:
            # Even though we're after the smallest eigenvalues (which='SM'), we use 
            # shift-invert mode (sigma=0) and solve for the largest. This is more efficient. 
            # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
            eigvals = scipy.sparse.linalg.eigsh(L, sigma=0, which='LM', k=k + 10, return_eigenvectors=False)
        except RuntimeError:
            # Runtime error is most likely due to singularity. Add small delta to matrix and try again. 
            # This will burn through an iteration since we don't want to do this forever.
            eps = scipy.sparse.diags(np.zeros((W.shape[0],)) + etol, 0)
            L += eps
            n_singular += 1
            logger.warning('Graph laplacian likely singular. Perturbing matrix, but this may introduce inaccuracies.')
            max_iter -= 1
            continue

        # Remove zero eigenvalues.
        eigvals = eigvals[~np.isclose(eigvals, 0, atol=1e-5 + etol*n_singular)]
        eigvals.sort()
    
        if len(eigvals) >= k_orig:
            eigvals = eigvals[:k_orig]
            enough_eigvals = True
        else:
            max_iter -= 1
            k += k - len(eigvals) + 1 # Add one for good measure.

    if max_iter <= 0:
        logger.error('Could not solve for the desired number of eigenvalues. Please check the number of connected components in your graph.')
    
    if scale_invariant:
        eigvals /= eigvals[0]

    return eigvals[:k_orig]


def mesh_and_featurize_image(
    image: BioReader,
    chunk_size: Tuple[int, int, int] = (256, 256, 256),
    num_features : int = 50, 
    scale_invariant : bool = False,
    limit_mesh_size : int = None):
    """ Mesh and generate spectral features for all ROIs in a 3D image.

    Image is initially scanned for all ROIs and corresponding bounding boxes. 
    Each ROI is then loaded in its entirety and meshed. The mesh is then used 
    to generate spectral features using the graph Laplacian. 

    Inputs:
        image - BioReader handle to the image
        chunk_size - Size of chunks used for image traversal 
        num_features - Number of spectral features to calculate
        scale_invariant - Specify if the calculated features should be scale invariant
        limit_mesh_size - If specified, the number of faces in generated meshes are limited
                          to the supplied value
    Outputs:
        labels - The label IDs of each ROI
        features - An N x num_features matrix containing the spectral features for each ROI
    """
    
    # Store minimum and maximum bounds of every object.
    min_bounds = {}
    max_bounds = {}
    
    # Go through the image in chunks and determine extents of each ROI. 
    for y in range(0, image.Y, chunk_size[1]):
        for x in range(0, image.X, chunk_size[0]):
            for z in range(0, image.Z, chunk_size[2]):
                
                x_step = np.min([x + chunk_size[0], image.X])
                y_step = np.min([y + chunk_size[1], image.Y])
                z_step = np.min([z + chunk_size[2], image.Z])

                chunk = np.squeeze(image[y:y_step, x:x_step, z:z_step]) 
                
                labels = np.unique(chunk[chunk > 0]) 
                
                for label in labels:
                    coords = np.argwhere(chunk == label)
                    # Add a one pixel padding so long as we're not on a boundary. 
                    curr_min = np.min(coords, axis=0) + np.array([y, x, z]) - 1
                    curr_min = np.maximum([0, 0, 0], curr_min)
                    
                    curr_max = np.max(coords, axis=0) + np.array([y, x, z]) + 1
                    curr_max = np.minimum([image.Y, image.X, image.Z], curr_max)
                    
                    if label not in min_bounds: 
                        min_bounds[label] = curr_min
                    else:
                        min_bounds[label] = np.min(np.stack([min_bounds[label], curr_min]), axis=0)
                    
                    if label not in max_bounds:
                        max_bounds[label] = curr_max
                    else:
                        max_bounds[label] = np.max(np.stack([max_bounds[label], curr_max]), axis=0)
    

    # Get subvolume for each ROI and generate mesh. 
    # Note: we assume labels in min and max dicts are the same.
    labels = list(min_bounds.keys())
    features = np.zeros((len(labels), num_features))
    for i, (label, min_bounds, max_bounds) in enumerate(zip(labels, min_bounds.values(), max_bounds.values())): 
        subvol = image[min_bounds[0]:max_bounds[0], min_bounds[1]:max_bounds[1], min_bounds[2]:max_bounds[2]]
        subvol = np.squeeze(subvol)

        verts, faces, _, _ = measure.marching_cubes((subvol == label).astype(np.uint8), 0, allow_degenerate=False)

        if limit_mesh_size is not None and faces.shape[0] > limit_mesh_size: 
            mesh_obj = trimesh.Trimesh(verts, faces).simplify_quadratic_decimation(limit_mesh_size)
            mesh_obj.remove_degenerate_faces()
            mesh_obj.remove_duplicate_faces()
            mesh_obj.remove_unreferenced_vertices()
            mesh_obj.remove_infinite_values()
            mesh_obj.fill_holes()

            verts = np.asarray(mesh_obj.vertices)
            faces = np.asarray(mesh_obj.faces)

        logger.info(f'Featurizing ROI {label} ({i + 1}/{len(labels)}) with {verts.shape[0]} vertices.')

        feats = mesh_spectral_features(verts, faces, k=num_features, scale_invariant=scale_invariant)
        features[i] = feats

    return labels, features
