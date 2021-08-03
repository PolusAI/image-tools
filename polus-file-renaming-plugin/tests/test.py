import unittest
from pathlib import Path
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../src"))
import main
from pathlib import Path
import re


class TestFileRenaming(unittest.TestCase):
    
    def test_pattern_to_regex_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ({"row": "(?P<row>[0-9][0-9])", "col": "(?P<col>[0-9][0-9])", "channel": "(?P<channel>[a-zA-Z]+)"}),
            ),
            #: Test case 2
            (
                ("img_x{row:c+}.tif"),
                ({"row": "(?P<row>[a-zA-Z]+)"})
            )
        ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.pattern_to_regex(from_val)
            self.assertEqual(result, to_val)
    
    def test_pattern_to_raw_f_string_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ({"row": "(?P<row>[0-9][0-9])", "col": "(?P<col>[0-9][0-9])", "channel": "(?P<channel>[a-zA-Z]+)"}),
                ("img_x(?P<row>[0-9][0-9])_y(?P<col>[0-9][0-9])_(?P<channel>[a-zA-Z]+).tif")
            ),
            #: Test case 2
            (
                ("img_x{row:c+}.tif"),
                ({"row": "(?P<row>[a-zA-Z]+)"}),
                ("img_x(?P<row>[a-zA-Z]+).tif")
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.pattern_to_raw_f_string(from_val1, from_val2)
            self.assertEqual(result, to_val)
            
    def test_pattern_to_fstring_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                ("newdata_x{row:03d}_y{col:03d}_c{channel:03d}.tif")
                ),
            #: Test case 2
            (
                ("newdata_x{row:c+}.tif"),
                ("newdata_x{row:s}.tif")
            )
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.pattern_to_fstring(from_val)
            self.assertEqual(result, to_val)
            
    def test_replace_cat_label_returns_unique_key_list_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                (["channel"])
                ),
            #: Test case 2
            (
                ("img_x{row:c+}.tif"),
                ("newdata_x{row:c+}.tif"),
                ([]) 
                )            
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.replace_cat_label(from_val1, from_val2)
            self.assertEqual(result, to_val)     
            
    def test_gen_all_matches_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x(?P<row>[0-9][0-9])_y(?P<col>[0-9][0-9])_(?P<channel>[a-zA-Z]+).tif"),
                ([Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")]),
                ([{"row": "01", "col": "01", "channel": "DAPI", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif")}, {"row": "01", "col": "01", "channel": "GFP", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif")}, {"row": "01", "col": "01", "channel": "TXRED", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")}])
                ),
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.gen_all_matches(from_val1, from_val2)
            self.assertEqual(result, to_val)
    
    def test_gen_all_matches_bad_pattern_invalid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x(?P<row>[a-zA-Z]+).tif"),
                ([Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")]),
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(AttributeError, main.gen_all_matches, from_val1, from_val2)
    
    def test_gen_all_matches_if_duplicate_named_grp_given_invalid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("x(?P<row>[0-9][0-9])_y(?P<row>[0-9][0-9])_c(?P<channel>[a-zA-Z]+).ome.tif"),
                ([Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif"), Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")])
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(ValueError, main.gen_all_matches, from_val1, from_val2)
    
    def test_numstrvalue_to_int_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ({"row": "01", "col": "01", "channel": "DAPI", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif")}),
                ({"row": 1, "col": 1, "channel": "DAPI", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif")})
                ),
            #: Test case 2
            (
                ({"row": "2", "col": "01", "channel": "TXRED", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")}),
                ({"row": 2, "col": 1, "channel": "TXRED", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")})
                ),
            #: Test case 3
            (
                ({"row": "0001", "col": "0001", "channel": "GFP", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif")}),
                ({"row": 1, "col": 1, "channel": "GFP", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif")})
                )
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.numstrvalue_to_int(from_val)
            self.assertEqual(result, to_val)

    def test_non_numstr_value_to_int_returns_category_index_dict_valid_input(self):
        test_cases = [
            (
                ("channel"),
                [
                    {"row": 1, "col": 1, "channel": "DAPI", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif")}, 
                    {"row": 1, "col": 1, "channel": "GFP", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif")}, 
                    {"row": 1, "col": 1, "channel": "TXRED", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")}
                    ],
                
                ({"DAPI": 0, "GFP": 1, "TXRED": 2})
                )   
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.non_numstr_value_to_int(from_val1, from_val2)
            self.assertEqual(result, to_val)
    
    def test_non_numstr_value_to_int_returns_error_invalid_input(self):
        test_cases = [
            #: Test case 1
            (
                (2),
                [
                    {"row": 1, "col": 1, "channel": "DAPI", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_DAPI.tif")}, 
                    {"row": 1, "col": 1, "channel": "GFP", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_GFP.tif")}, 
                    {"row": 1, "col": 1, "channel": "TXRED", "fname": Path("../tests/test_data/image_collection_1/img_x01_y01_TXRED.tif")}
                    ],
                ),
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(KeyError, main.non_numstr_value_to_int, from_val1, from_val2)
     
            
if __name__ == "__main__":
    unittest.main()