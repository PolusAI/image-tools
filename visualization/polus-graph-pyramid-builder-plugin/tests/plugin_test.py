import unittest

import tempfile

import random

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.Pyramid as pyramid
import src.Dataset as dataset
import src.Figure as figure

import itertools
from PIL import Image


class PluginTest(unittest.TestCase):

    def test_pyramid(self):
        
        self.nfeats  = 5
        self.ngraphs = self.nfeats*self.nfeats

        self.CHUNK_SIZE = 1024

        self.axisnames = [f"{combo[0]}_{combo[1]}" for combo in itertools.product(list(range(5)), repeat=2)]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_base = "temp_basedir"

            # initialize pyramid object
            graphPyramid = pyramid.GraphPyramid(output_dir=temp_dir, output_name=temp_base, \
                                                ngraphs=self.ngraphs, axisnames=self.axisnames, \
                                                CHUNK_SIZE=self.CHUNK_SIZE)
            
            self.assertTrue(graphPyramid.ngraphs == self.ngraphs)
            self.assertTrue(graphPyramid.fig_dim[0] * graphPyramid.fig_dim[1] == self.ngraphs)
            self.assertTrue([dim*self.CHUNK_SIZE for dim in graphPyramid.fig_dim] == graphPyramid.sizes)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, f"{temp_base}.dzi")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, f"{temp_base}.csv")))

            self.assertTrue(int(os.path.basename(graphPyramid.bottom_pyramidDir)) == graphPyramid.num_scales)

            # need to save objects to base pyramid dir
            ngraph_counter = 1
            for axisname in self.axisnames:
                img = np.ones((self.CHUNK_SIZE, self.CHUNK_SIZE, 4), dtype=np.uint8) * ngraph_counter
                img_pillow = Image.fromarray(img)
                img_pillow.save(os.path.join(graphPyramid.bottom_pyramidDir, f"{axisname}.{graphPyramid.image_extension}"))
                ngraph_counter += 1

            self.assertTrue(len(os.listdir(graphPyramid.bottom_pyramidDir)) == self.ngraphs)

            graphPyramid.build_thepyramid()

            img_dims = [1024, 1024]
            for scale in reversed(range(graphPyramid.num_scales+1)):

                scale_dir = os.path.join(temp_dir, f"{temp_base}_files/{scale}")
                zero_zero_imgpath = os.path.join(scale_dir, f"0_0.png")

                self.assertTrue(os.path.exists(zero_zero_imgpath))

                img_zero = np.array(Image.open(zero_zero_imgpath))
                if len(os.listdir(scale_dir)) >= 4:
                    self.assertTrue(img_zero.shape[0] == self.CHUNK_SIZE)
                    self.assertTrue(img_zero.shape[1] == self.CHUNK_SIZE)

                # the last image should have dimensions of (1,1), otherwise something is wrong
                if scale == 0:
                    self.assertTrue(img_zero.shape[1] == 1)
                    self.assertTrue(img_zero.shape[1] == 1)

    def test_loglinear_conversions(self):

        # this is what we expect to see after the conversion
        expected_data = {"Column1" : random.sample(range(-30, 30), 20)}
        
        # this converts expected_data
        logdata = dataset.LogData(expected_data)

        # this convert the logarithmic data back to the original input
            # this is important for formatting the graphs
        actual_data = figure.convert_tolinear(logdata.dataframe)

        # need numpy arrays to test whether they are close enough
        expected_array = np.array(expected_data["Column1"])
        actual_array   = np.array(actual_data["Column1"])

        self.assertTrue(np.allclose(expected_array, actual_array, atol=.003))


if __name__=="__main__":
    unittest.main()