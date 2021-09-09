import unittest, json
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../src"))
import main
from pathlib import Path

class TestFileRenaming(unittest.TestCase):
    
    """ Verify VERSION is correct """
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    
    def setUp(self):
        with open(Path(__file__).with_name('file_rename_test.json'),'r') as fr:
            self.data = json.load(fr)
            
    def test_numeric_fixed_width(self):
        inp_pattern = '00{one:d}0{two:dd}-{three:d}-00100100{four:d}.tif'
        out_pattern = 'output{one:dd}0{two:ddd}-{three:dd}-00100100{four:dd}.tif'
        inp_files = self.data['robot']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        self.assertTrue(result)

    def test_alphanumeric_fixed_width(self):
        inp_pattern = 'S1_R{one:d}_C1-C11_A1_y0{two:dd}_x0{three:dd}_c0{four:dd}.ome.tif'
        out_pattern = 'output{one:dd}_C1-C11_A1_y0{two:ddd}_x0{three:ddd}_c0{four:ddd}.ome.tif'
        inp_files = self.data['brain']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_alphanumeric_variable_width(self):
        inp_pattern = 'S1_R{one:d}_C1-C11_A1_y{two:d+}_x{three:d+}_c{four:d+}.ome.tif'
        out_pattern = 'output{one:dd}_C1-C11_A1_y{two:d+}_x{three:d+}_c{four:d+}.ome.tif'
        inp_files = self.data['variable']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_parenthesis(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_({chan:c+}).tif"
        out_pattern = "output{row:dd}_{col:ddd}_{chan:dd}.tif"
        inp_files = self.data['parenthesis']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_two_chan_to_digit(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_{chan:c+}_{ychan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}_{ychan:ddd}.tif"
        inp_files = self.data['two_chan']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
    
    def test_three_chan_to_digit(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_{chan:c+}_{ychan:c+}_{alphachan:ccc}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}_{ychan:ddd}_{alphachan:dddd}.tif"
        inp_files = self.data['three_chan']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
    
    def test_three_char_chan(self):
        inp_pattern = "img x{row:dd} y{col:dd} {chan:ccc}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:ccc}.tif"
        inp_files = self.data['three_char_chan']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_varied_digits(self):
        inp_pattern = "p{p:d}_y{y:d}_r{r:d+}_c{c:d+}.ome.tif"
        out_pattern = "p{p:dd}_y{y:dd}_r{r:dddd}_c{c:ddd}.ome.tif"
        inp_files = self.data['tissuenet-val-labels-45-C']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_spaces(self):
        """Ensure non-alphanumeric chars are handled properly
        (spaces only)"""
        inp_pattern = "img x{row:dd} y{col:dd} {chan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}.tif"
        inp_files = self.data['non_alphanum_int']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_non_alphanum_float(self):
        """Ensure non-alphanumeric chars are handled properly
        (spaces, periods, commas, brackets)"""
        inp_pattern = "img x{row:dd}.{other:d+} y{col:dd} {chan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_ {other:d+} {chan:dd}.tif"
        inp_files = self.data['non_alphanum_float']
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)
        
    def test_dashes_parentheses(self):
        """Ensure non-alphanumeric chars are handled properly
        (dashes, parentheses)"""
        inp_pattern = "0({mo:dd}-{day:dd})0({mo2:dd}-{day2:dd})-({a:d}-{b:d})-{col:ddd}.ome.tif"
        out_pattern = "0({mo:ddd}-{day:ddd})0{mo2:dd}-{day2:dd})-({a:dd}-{b:dd})-{col:ddd}.ome.tif"
        inp_files = self.data["kph-kirill"]
        out_dir = ""
        result = main.main(inp_pattern, out_pattern, inp_files, out_dir)
        print("result: ", result)

    def test_pattern_to_regex_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                (
                    {
                        "row": "(?P<row>[0-9][0-9])", 
                        "col": "(?P<col>[0-9][0-9])", 
                        "channel": "(?P<channel>[a-zA-Z]+)"
                        }
                    ),
                ),
            #: Test case 2
            (
                ("img_x{row:c+}.tif"),
                ({"row": "(?P<row>[a-zA-Z]+)"})
                ),
            #: Test case 3
            (
                (""),
                ({})
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
                (
                    {
                        "row": "(?P<row>[0-9][0-9])",
                        "col": "(?P<col>[0-9][0-9])",
                        "channel": "(?P<channel>[a-zA-Z]+)"
                        }
                    ),
                ("img_x(?P<row>[0-9][0-9])_y(?P<col>[0-9][0-9])_(?P<channel>[a-zA-Z]+).tif")
            ),
            #: Test case 2
            (
                ("img_x{row:c+}.tif"),
                ({"row": "(?P<row>[a-zA-Z]+)"}),
                ("img_x(?P<row>[a-zA-Z]+).tif")
                ),
            #: Test case 3
            (
                ("img_x01.tif"),
                ({}),
                ("img_x01.tif")
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
                ),
            #: Test case 3
            (
                ("newdata_x01.tif"),
                ("newdata_x01.tif")
                )
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.pattern_to_fstring(from_val)
            self.assertEqual(result, to_val)
            
    def test_replace_cat_label_returns_unique_keys_valid_input(self):
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
                ),
            #: Test case 3
            (
                ("img_x01.tif"),
                ("newdata_x01.tif"),
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
                (
                    [
                        "img_x01_y01_DAPI.tif",
                        "img_x01_y01_GFP.tif",
                        "img_x01_y01_TXRED.tif"
                        ]
                    ),
                (
                    [
                        {
                            "row": "01", 
                            "col": "01", 
                            "channel": "DAPI", 
                            "fname": "img_x01_y01_DAPI.tif"
                            },
                        {
                            "row": "01", 
                            "col": "01", 
                            "channel": "GFP", 
                            "fname": "img_x01_y01_GFP.tif"
                            },
                        {
                            "row": "01",
                            "col": "01",
                            "channel": "TXRED",
                            "fname": "img_x01_y01_TXRED.tif"
                            }
                        ]
                    )
                ),
            #: Test case 2
            (
                ("img_x01.tif"),
                (["img_x01.tif"]),
                ([{'fname': 'img_x01.tif'}])
                )
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
                (
                    [
                        "img_x01_y01_DAPI.tif",
                        "img_x01_y01_GFP.tif",
                        "img_x01_y01_TXRED.tif"
                        ]
                    ),
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(
                AttributeError, main.gen_all_matches, from_val1, from_val2
                )
    
    def test_gen_all_matches_duplicate_namedgrp_invalid_input(self):
        test_cases = [
            #: Test case 1
            (
                ("x(?P<row>[0-9][0-9])_y(?P<row>[0-9][0-9])_c(?P<channel>[a-zA-Z]+).ome.tif"),
                (
                    [
                        "img_x01_y01_DAPI.tif",
                        "img_x01_y01_GFP.tif",
                        "img_x01_y01_TXRED.tif"
                        ]
                    )
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(
                ValueError, main.gen_all_matches, from_val1, from_val2
                )
    
    def test_numstrvalue_to_int_valid_input(self):
        test_cases = [
            #: Test case 1
            (
                (
                    {
                        "row": "01",
                        "col": "01",
                        "channel": "DAPI",
                        "fname": "img_x01_y01_DAPI.tif"
                        }
                    ),
                (
                    {
                        "row": 1,
                        "col": 1,
                        "channel": "DAPI",
                        "fname": "img_x01_y01_DAPI.tif"
                        }
                    )
                ),
            #: Test case 2
            (
                (
                    {
                        "row": "2",
                        "col": "01",
                        "channel": "TXRED",
                        "fname": "img_x01_y01_TXRED.tif"
                        }
                    ),
                (
                    {
                        "row": 2,
                        "col": 1,
                        "channel": "TXRED",
                        "fname": "img_x01_y01_TXRED.tif"
                        }
                    )
                ),
            #: Test case 3
            (
                (
                    {
                        "row": "0001",
                        "col": "0001",
                        "channel": "GFP",
                        "fname": "img_x01_y01_GFP.tif"
                        }
                    ),
                (
                    {
                        "row": 1,
                        "col": 1,
                        "channel": "GFP",
                        "fname": "img_x01_y01_GFP.tif"
                        }
                    )
                )
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.numstrvalue_to_int(from_val)
            self.assertEqual(result, to_val)

    def test_cat_to_int_returns_cat_index_dict_valid_input(self):
        test_cases = [
            (
                ("channel"),
                [
                    {
                        "row": 1,
                        "col": 1,
                        "channel": "DAPI",
                        "fname": "img_x01_y01_DAPI.tif"
                        }, 
                    {
                        "row": 1,
                        "col": 1, 
                        "channel": "GFP", 
                        "fname": "img_x01_y01_GFP.tif"
                        }, 
                    {
                        "row": 1, 
                        "col": 1, 
                        "channel": "TXRED", 
                        "fname": "img_x01_y01_TXRED.tif"
                        }
                    ],
                
                ({"DAPI": 0, "GFP": 1, "TXRED": 2})
                )   
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.non_numstr_value_to_int(from_val1, from_val2)
            self.assertEqual(result, to_val)
    
    def test_cat_to_int_returns_error_invalid_input(self):
        test_cases = [
            #: Test case 1
            (
                (2),
                [
                    {
                        "row": 1, 
                        "col": 1, 
                        "channel": "DAPI", 
                        "fname": "img_x01_y01_DAPI.tif"
                        }, 
                    {
                        "row": 1, 
                        "col": 1, 
                        "channel": "GFP", 
                        "fname": "img_x01_y01_GFP.tif"
                        }, 
                    {
                        "row": 1, 
                        "col": 1, 
                        "channel": "TXRED", 
                        "fname": "img_x01_y01_TXRED.tif"
                        }
                    ],
                ),
            ]
        for test_case in test_cases:
            (from_val1, from_val2) = test_case
            self.assertRaises(
                KeyError, main.non_numstr_value_to_int, from_val1, from_val2
                )
     
if __name__ == "__main__":
    unittest.main()