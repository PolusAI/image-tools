import unittest

from src.utils import Tile, UnTile

import torch

class TileTest(unittest.TestCase):
    
    def test_tile(self):
        
        params = [
            ((5,1,1080,1080),(512,512)),
            ((5,3,1080,1080),(512,512)),
            ((5,1,1080,1000),(512,512)),
            ((5,1,1080,1024),(512,512)),
            ((5,1,480,480),(512,512)),
            ((5,1,1080,1080),(300,512)),
            ((5,1,1080,1000),(512,101))
        ]
        
        for input_shape,tile_size in params:
            
            with self.subTest(input_shape=input_shape,tile_size=tile_size):
        
                tile = Tile(tile_size=tile_size)
                untile = UnTile(tile_size=tile_size)
                
                image = torch.rand(5,1,1080,1080,dtype=torch.float32)
                
                image_tiled,input_shape = tile(image)
                
                image_untiled = untile(image_tiled,input_shape)
                
                assert torch.all(image==image_untiled)


if __name__=="__main__":
    unittest.main()
    