import os, sys
import unittest
import numpy as np
from bfio import BioReader
from skimage import measure


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../src'))

import mesh


class TestScalableFeaturization(unittest.TestCase):
    def test_scalable_features(self):
        # First, load entire image and calculate features. 
        bunny_file = os.path.join(dir_path, 'test_data/bunny.ome.tif')
        with BioReader(bunny_file) as br:
            ref_img = np.squeeze(br[:])
        
        verts, faces, _, _ = measure.marching_cubes((ref_img == 255).astype(np.uint8), 0, allow_degenerate=False)
        ref_features = mesh.mesh_spectral_features(verts, faces, k=50, scale_invariant=False)

        self.assertEquals(len(ref_features), 50)

        # Now calculate features using chunking. 
        with BioReader(bunny_file) as br:
            labels, features = mesh.mesh_and_featurize_image(br, chunk_size=(20, 20, 20), num_features=50, scale_invariant=False)
        
        self.assertTrue(
            np.allclose(ref_features, features.flatten(), atol=1.e-3)
        )
        

if __name__ == '__main__':
    unittest.main()