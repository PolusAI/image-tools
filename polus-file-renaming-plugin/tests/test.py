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
            
    def test_duplicate_channels_to_digit(self):
        test_cases = [
            (
                ("r{row:ddd}_c{col:ddd}_{chan:ccc}.ome.tif"), 
                ("output_r{row:dddd}_c{col:dddd}_{chan:ddd}.ome.tif")
                ),
            ]
        for test_case in test_cases:
            (inp_pattern, out_pattern) = test_case
            inp = self.data["duplicate_channels_to_digit"]
            out = ""
            result = main.main(inp, inp_pattern, out, out_pattern)
            self.assertTrue(result)
    
    def test_duplicate_channels_to_digit_non_spec_digit_len(self):
        test_cases = [
            (
                ("r{row:ddd}_c{col:ddd}_{chan:ccc}.ome.tif"), 
                ("output_r{row:dddd}_c{col:dddd}_{chan:d+}.ome.tif")
                ),
            ]
        for test_case in test_cases:
            (inp_pattern, out_pattern) = test_case
            inp = self.data["duplicate_channels_to_digit"]
            out = ""
            result = main.main(inp, inp_pattern, out, out_pattern)
            self.assertTrue(result)
            
    def test_invalid_input_raises_error(self):
        test_cases = [
            (
                ("r.ome.tif"), 
                ("output_r{row:dddd}_c{col:dddd}_{chan:d+}.ome.tif")
                ),
            ]
        for test_case in test_cases:
            (inp_pattern, out_pattern) = test_case
            inp = self.data["duplicate_channels_to_digit"]
            out = ""
            #result = main.main(inp, inp_pattern, out, out_pattern)
            self.assertRaises(
                KeyError, main.main, inp, inp_pattern, out, out_pattern
                )
                 
    def test_non_alphanum_inputs_percentage_sign(self):
        test_cases = [
            (
                ("%{row:ddd}_c{col:ddd}_z{z:d+}.ome.tif"), 
                ("%{row:dddd}_col{col:dddd}_z{z:d+}.ome.tif")
                ),
            ]
        for test_case in test_cases:
            (inp_pattern, out_pattern) = test_case
            inp = self.data["percentage_file"]
            out = ""
            result = main.main(inp, inp_pattern, out, out_pattern)
            self.assertTrue(result)
    
    def test_bleed_through_fixed_chars(self):
        test_cases = [
            (
                ("r{row:ddd}_c{col:ddd}_z{z:d+}.ome.tif"), 
                ("output_row{row:dddd}_col{col:dddd}_z{z:d+}.ome.tif")
                ),
            (
                ("r{row:ddd}_c{col:ddd}_z{z:d+}.ome.tif"), 
                ("output_row{row:dddd}_col{col:dddd}_z{z:ddd}.ome.tif")
                ),
            (
                ("r{row:ddd}_c{col:ddd}_z{z:ddd}.ome.tif"), 
                ("output_row{row:dddd}_col{col:dddd}_z{z:d+}.ome.tif")
                ),
            ]
        for test_case in test_cases:
            (inp_pattern, out_pattern) = test_case
            inp = self.data["bleed_through_estimation_fixed_chars"]
            out = ""
            result = main.main(inp, inp_pattern, out, out_pattern)
            self.assertTrue(result)
        
    def test_numeric_fixed_width(self):
        inp_pattern = '00{one:d}0{two:dd}-{three:d}-00100100{four:d}.tif'
        out_pattern = 'output{one:dd}0{two:ddd}-{three:dd}-00100100{four:dd}.tif'
        inp = self.data['robot']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)

    def test_alphanumeric_fixed_width(self):
        inp_pattern = 'S1_R{one:d}_C1-C11_A1_y0{two:dd}_x0{three:dd}_c0{four:dd}.ome.tif'
        out_pattern = 'output{one:dd}_C1-C11_A1_y0{two:ddd}_x0{three:ddd}_c0{four:ddd}.ome.tif'
        inp = self.data['brain']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_alphanumeric_variable_width(self):
        inp_pattern = 'S1_R{one:d}_C1-C11_A1_y{two:d+}_x{three:d+}_c{four:d+}.ome.tif'
        out_pattern = 'output{one:dd}_C1-C11_A1_y{two:d+}_x{three:d+}_c{four:d+}.ome.tif'
        inp = self.data['variable']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_parenthesis(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_({chan:c+}).tif"
        out_pattern = "output{row:dd}_{col:ddd}_{chan:dd}.tif"
        inp = self.data['parenthesis']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_two_chan_to_digit(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_{chan:c+}_{ychan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}_{ychan:ddd}.tif"
        inp = self.data['two_chan']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
    
    def test_three_chan_to_digit(self):
        inp_pattern = "img_x{row:dd}_y{col:dd}_{chan:c+}_{ychan:c+}_{alphachan:ccc}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}_{ychan:ddd}_{alphachan:dddd}.tif"
        inp = self.data['three_chan']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
    
    def test_three_char_chan(self):
        inp_pattern = "img x{row:dd} y{col:dd} {chan:ccc}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:ccc}.tif"
        inp = self.data['three_char_chan']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_varied_digits(self):
        inp_pattern = "p{p:d}_y{y:d}_r{r:d+}_c{c:d+}.ome.tif"
        out_pattern = "p{p:dd}_y{y:dd}_r{r:dddd}_c{c:ddd}.ome.tif"
        inp = self.data['tissuenet-val-labels-45-C']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_spaces(self):
        """Ensure non-alphanumeric chars are handled properly
        (spaces only)
        
        """
        inp_pattern = "img x{row:dd} y{col:dd} {chan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_{chan:dd}.tif"
        inp = self.data['non_alphanum_int']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_non_alphanum_float(self):
        """Ensure non-alphanumeric chars are handled properly
        (spaces, periods, commas, brackets)
        
        """
        inp_pattern = "img x{row:dd}.{other:d+} y{col:dd} {chan:c+}.tif"
        out_pattern = "output{row:ddd}_{col:ddd}_ {other:d+} {chan:dd}.tif"
        inp = self.data['non_alphanum_float']
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)
        self.assertTrue(result)
        
    def test_dashes_parentheses(self):
        """Ensure non-alphanumeric chars are handled properly
        (dashes, parentheses)
        
        """
        inp_pattern = "0({mo:dd}-{day:dd})0({mo2:dd}-{day2:dd})-({a:d}-{b:d})-{col:ddd}.ome.tif"
        out_pattern = "0({mo:ddd}-{day:ddd})0{mo2:dd}-{day2:dd})-({a:dd}-{b:dd})-{col:ddd}.ome.tif"
        inp = self.data["kph-kirill"]
        out = ""
        result = main.main(inp, inp_pattern, out, out_pattern)        
        self.assertTrue(result)

    def test_map_pattern_grps_to_regex_valid_input(self):
        test_cases = [
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
            
            (("img_x{row:c+}.tif"), ({"row": "(?P<row>[a-zA-Z]+)"})), ((""), ({}))
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.map_pattern_grps_to_regex(from_val)
            self.assertEqual(result, to_val)
            
    def test_convert_to_regex_valid_input(self):
        test_cases = [
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

            (
                ("img_x{row:c+}.tif"),
                ({"row": "(?P<row>[a-zA-Z]+)"}),
                ("img_x(?P<row>[a-zA-Z]+).tif")
                ),

            (
                ("img_x01.tif"),
                ({}),
                ("img_x01.tif")
            )
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.convert_to_regex(from_val1, from_val2)
            self.assertEqual(result, to_val)
            
    def test_specify_len_valid_input(self):
        test_cases = [
            (
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                ("newdata_x{row:03d}_y{col:03d}_c{channel:03d}.tif")
                ),

            (
                ("newdata_x{row:c+}.tif"),
                ("newdata_x{row:s}.tif")
                ),

            (
                ("newdata_x01.tif"),
                ("newdata_x01.tif")
                )
            ]
        for test_case in test_cases:
            (from_val, to_val) = test_case
            result = main.specify_len(from_val)
            self.assertEqual(result, to_val)
            
    def test_get_char_to_digit_grps_returns_unique_keys_valid_input(self):
        test_cases = [
            (
                ("img_x{row:dd}_y{col:dd}_{channel:c+}.tif"),
                ("newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif"),
                (["channel"])
                ),

            (
                ("img_x{row:c+}.tif"),
                ("newdata_x{row:c+}.tif"),
                ([]) 
                ),

            (
                ("img_x01.tif"),
                ("newdata_x01.tif"),
                ([])
                )       
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.get_char_to_digit_grps(from_val1, from_val2)
            self.assertEqual(result, to_val)
            
    def test_extract_named_grp_matches_valid_input(self):
        test_cases = [
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

            (
                ("img_x01.tif"),
                (["img_x01.tif"]),
                ([{'fname': 'img_x01.tif'}])
                )
            ]
        for test_case in test_cases:
            (from_val1, from_val2, to_val) = test_case
            result = main.extract_named_grp_matches(from_val1, from_val2)
            self.assertEqual(result, to_val)
    
    def test_extract_named_grp_matches_bad_pattern_invalid_input(self):
        test_cases = [
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
                AttributeError, main.extract_named_grp_matches, from_val1, from_val2
                )
    
    def test_extract_named_grp_matches_duplicate_namedgrp_invalid_input(self):
        test_cases = [
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
                ValueError, main.extract_named_grp_matches, from_val1, from_val2
                )
    
    def test_str_to_int_valid_input(self):
        test_cases = [
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
            result = main.str_to_int(from_val)
            self.assertEqual(result, to_val)

    def test_letters_to_int_returns_cat_index_dict_valid_input(self):
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
            result = main.letters_to_int(from_val1, from_val2)
            self.assertEqual(result, to_val)
    
    def test_letters_to_int_returns_error_invalid_input(self):
        test_cases = [
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
                KeyError, main.letters_to_int, from_val1, from_val2
                )
     
if __name__ == "__main__":
    unittest.main()