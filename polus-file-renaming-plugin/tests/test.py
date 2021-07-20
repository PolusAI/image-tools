import unittest
from pathlib import Path
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../src'))

import main

class TestFileRenaming(unittest.TestCase):

    def test_convert_filename_test(self):
        """
        Args:
            input_image: img_x01_y01_TXRED.tif
            fpattern: img_x{row:dd}_y{col:dd}_{channel:c+}.tif
            outpattern: newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif

        Returns:
            temp_fp: newdata_x001_y001_c_rgxstart_[0-9]{3}_rgxmid_TXRED_rgxend_.tif
        """
        test_cases = [
            (
                ("img_x01_y01_TXRED.tif"),
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                ("newdata_x001_y001_c_rgxstart_[0-9]{3}_rgxmid_TXRED_rgxend_.tif")
            ),
            (
                ("r001_z000_y(00-14)_x0(00-21)_c00000.ome.tif"),
                ("r{row:ddd}_z{z:ddd}_y(00-14)_x0(00-21)_c{col:ddddd}.ome.tif"),
                ("newdata_r{row:dd}_z{z:dd}_y(00-14)_x0(00-21)_c{col:ddd}.ome.tif"),
                ("newdata_r01_z00_y(00-14)_x0(00-21)_c000.ome.tif"),
            ),
            (
                ("r01_x05_y23_z_01-30__mask.ome.tif"),
                ("r{dd}_x{dd}_y{dd}_z_01-30__mask.ome.tif"),
                ("newdata_{dddd}_x{ddd}_y{ddd}_z_01-30__mask.ome.tif"),
                (None)
            ),
            (
                ("filename_with_no_match.tif"),
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                ("newdata_x001_y001_c_rgxstart_[0-9]{3}_rgxmid_TXRED_rgxend_.tif")
            ),
            #: User writes filepattern that file doesn't match
            (
                ("img_x01_y01_TXRED.ome.tif"),
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                ("newdata_x001_y001_c_rgxstart_[0-9]{3}_rgxmid_TXRED_rgxend_.tif")
            ),
        ]
        for test_case in test_cases:
            #: Inputs: input_image, fpattern, outpattern, temp_fp
            (from_val1, from_val2, from_val3, to_val) = test_case
            result = main.translate_regex(from_val1, from_val2, from_val3)
            self.assertEqual(result, to_val) 
         
    def test_translate_regex_test(self):
        """
    
        Args:
            from_val: ['img_x', 'row:dd', '_y', 'col:dd', '_', 'channel:c+', '.tif']
        Returns:
            to_val: ['img_x', '[0-9]{2}', '_y', '[0-9]{2}', '_', '[a-zA-Z]*', '.tif']
        """
        test_cases = [
            (
                (['img_x', 'row:dd', '_y', 'col:dd', '_', 'channel:c+', '.tif']), 
                (['img_x', '[0-9]{2}', '_y', '[0-9]{2}', '_', '[a-zA-Z]*', '.tif']),
            ),
            (
                (["a"]), 
                (["a"]),
            ),
            (
                (["a:dd"]), 
                (["[0-9]{2}"]),
            )
        ]
        
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.translate_regex(from_val)
            self.assertEqual(result, to_val)
            
    def format_output_digit_test(self):
        """
        match_in_input: 01 OR TXRED 
        rgx_match_in: [0-9]{2} OR [a-zA-Z]* 
        rgx_match_out:[0-9]{3} OR [0-9]{2}
        """
        print("shortest function start")
        test_cases = [
            (
                (["01", "[0-9]{2}", "[0-9]{3}"]),
                (["TXRED", "[a-zA-Z]*", "[0-9]{2}"]),
                ()
            ),
        ]
        
        for test_case in test_cases:
            (from_val1, from_val2, from_val3, to_val) = test_case
            result = main.format_output_digit(from_val1, from_val2, from_val3)
            self.assertEqual(result, to_val)

        print("shortest function end")
    def test_str_to_num(self):
        """
        Args:
            input_file: ../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif
            chan_data_dict_sorted  {1: {PosixPath('../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_DAPI_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_DAPI_rgxend_.tif'}, 2: {PosixPath('../tests/test_data/image_collection_1/img_x01_y01_GFP.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_GFP_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_GFP_rgxend_.tif'}, 3: {PosixPath('../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.tif'}}
        
        Returns:
            final_filename  newdata_x001_y001_c03.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.tif
        """
        
        test_cases = [
            (
                [
                    "../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif", 
                    {
                        1: {Path('../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_DAPI_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_DAPI_rgxend_.tif'}, 
                        2: {Path('../tests/test_data/image_collection_1/img_x01_y01_GFP.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_GFP_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_GFP_rgxend_.tif'}, 
                        3: {Path('../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif'): 'newdata_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.tif'}
                    }, 
                    "newdata_x001_y001_c03.ti.ome.tifD_x001_y001_c_rgxstart_[0-9]{2}_rgxmid_TXRED_rgxend_.tif"
                ]
            ), # Test case 1
        ]
if __name__ == '__main__':
    unittest.main()