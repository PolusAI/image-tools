import unittest
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../src'))

import main

class TestFileRenaming(unittest.TestCase):

    def test_convert_filename(self):
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
        ]
        for test_case in test_cases:
            #: Inputs: input_image, fpattern, outpattern, temp_fp
            (from_val1, from_val2, from_val3, to_val) = test_case
            result = main.translate_regex(
                from_val1, from_val2, from_val3, to_val)
            self.assertEqual(result, to_val) 
         
    def test_translate_regex(self):
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
        ]
        
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.translate_regex(from_val, to_val)
            self.assertEqual(result, to_val)

if __name__ == '__main__':
    unittest.main()