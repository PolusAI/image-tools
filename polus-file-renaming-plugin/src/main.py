import logging
import argparse
from pathlib import Path
import re
import shutil

"""
This is the order in which functions are called:

pattern_to_regex(): Convert pattern to regex; include named regex group
pattern_to_raw_f_string(): Create raw f strings (ex: (?P<row>[a-zA-Z]+))
pattern_to_fstring(): Convert outpattern to format string (ex: {row:03d})
replace_cat_label(): Map chars to nums if input is char and out is digit
gen_all_matches(): Get matches from input pattern and input filename
numstrvalue_to_int(): Convert numeric str dictionary values to integers
non_numstr_value_to_int(): Make dictionary of category labels and numbers.
"""

def pattern_to_regex(pattern:str) -> dict:
    """Add named regular expression group, ?p<>, and make pattern regex.
    
    Ultimately, this function builds a dictionary where the key is the 
    named regex group and the value is the regex.
    
    Args:
        pattern: file pattern provided by the user
    Returns:
        rgx_patterns: named group and regex value dictionary
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug("pattern_to_regex() inputs: {}".format(pattern))
    #: Make list of (named group, user file pattern) tuples
    patterns = re.findall(r'\{(\w+):([dc+]+)\}', pattern)
    pattern_map = {
    'd' : r'[0-9]',
    'c' : r'[a-zA-Z]',
    '+' : '+'
    }
    rgx_patterns = {}
    for var, pat in patterns:
        pp = ''.join([pattern_map[p] for p in pat])
        rgx_patterns[var] = fr'(?P<{var}>{pp})'
    logger.debug("pattern_to_regex() returns {}.".format(rgx_patterns))
    return rgx_patterns

def pattern_to_raw_f_string(pattern:str, rgx_patts:dict)->str:
    """
    Create f strings, a cleaner version of .format().
    
    Using the file pattern and the dictionary with regex patterns, where
    the key is a named group and the value is a regex, we replace the 
    file pattern with a file pattern that can be read by the re library.
        
    Args:
        pattern: file pattern provided by the user
        rgx_patts: named group and regex value dictionary
    Returns:
        rgx_pattern: file pattern that is readable by re library
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug(
        "pattern_to_raw_f_string() inputs: {}, {}". format(pattern, rgx_patts)
        )
    rgx_pattern = pattern
    for named_grp, regex_str in rgx_patts.items():
        #: Using the prefix "fr" creates raw f-strings
        rgx_pattern = re.sub(fr'\{{{named_grp}:.*?\}}', regex_str, rgx_pattern)
    logger.debug("pattern_to_raw_f_string() returns {}.".format(rgx_pattern))
    return rgx_pattern

def pattern_to_fstring(out_pattern:str)->str:
    """
    Convert outpattern to format string, :03d. 
    
    For example, "newdata_x{row:ddd}" returns "new_data{row:03d}".
  
    Args:
        out_pattern: output file pattern provided by the user
    
    Returns:
        out_pattern_fstring: output file pattern converted to f-string
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug("pattern_to_fstring() inputs: {}".format(out_pattern))
    #: Make list of (named groups, user file pattern) tuples 
    out_patterns = re.findall(r'\{(\w+):([dc+]+)\}', out_pattern)
    f_string_dict = {}
    for key, value in out_patterns:
        temp_value = value[:1]
        if "+" not in value and temp_value == "c":
            temp_value = "s"
            f_string_dict[key] = "{" + key + ":" + str(len(value)) + temp_value + "}"
        # Prepend "0" to field to enable 0-padding of numeric types
        elif "+" not in value and temp_value != "c":
            f_string_dict[key] = "{" + key + ":0" + str(len(value)) + temp_value + "}"
        elif "+" in value and temp_value == "c":
            temp_value = "s"
            f_string_dict[key] = "{" + key  + ":" + temp_value + "}"
        # Prepend "0" to field to enable 0-padding of numeric types
        elif "+" in value and temp_value != "c":
            f_string_dict[key] = "{" + key  + ":0" + temp_value + "}"
    out_pattern_fstring = out_pattern
    for named_group, fstring in f_string_dict.items():
        out_pattern_fstring = re.sub(fr'\{{{named_group}:.*?\}}', fstring, out_pattern_fstring)
    logger.debug("pattern_to_f_string() returns {}.".format(out_pattern_fstring))
    return out_pattern_fstring

def replace_cat_label(inp_pattern:str, out_pattern:str)->list:
    """Return categorical labels if inpatt is char and outpatt is digit.
    
    Args:
        inp_pattern: user input pattern
        out_pattern: user output pattern
    Returns:
        unique_keys: list of unique category labels to make numeric
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug(
        "replace_cat_label() inputs: {}, {}".format(inp_pattern, out_pattern))
    #: Generate list [('row', 'dd'), ('col', 'dd'), ('channel', 'c+')]
    in_patts = re.findall(r'\{(\w+):([dc+]+)\}', inp_pattern)
    out_patts = re.findall(r'\{(\w+):([dc+]+)\}', out_pattern)
    
    #: If input file pattern is c and output is d, list unique key
    unique_keys = list(set([
        inp_grp for (inp_grp,inp_rgx) in in_patts 
        for (out_grp, out_rgx) in out_patts  
        if inp_rgx.startswith("c") and out_rgx.startswith("d")
        ]))
    logger.debug("replace_cat_label() returns {}.".format(unique_keys))
    return unique_keys

def gen_all_matches(rgx_pattern:str, inp_files:list)->dict:
    """
    Get matches from input pattern and input filename
    
    Generate a list of dictionaries, where each dictionary key is the 
    named regular expression capture group and the value is the
    corresponding match from the filename.
    
    Args:
        rgx_pattern: input pattern as f string (has ?P for rgx groups)
        inp_files: list of files in input directory
    
    Returns:
        grp_match_dict_list: k=capture grps, v=match, last item has filename info
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug("gen_all_matches() inputs: {}, {}".format(rgx_pattern, inp_files))
    grp_match_dict_list =[]
    #: Build list of dicts, where key is capture group and value is match
    for inp_file in inp_files:
        try:
            grp_match_dict = re.match(rgx_pattern, inp_file.name).groupdict()
            #: Add filename information to dictionary
            grp_match_dict["fname"] = inp_file
            grp_match_dict_list.append(grp_match_dict)
        except AttributeError:
            logger.error(
                "File pattern does not match one or more files. See README for pattern rules.")
            raise AttributeError(
                "File pattern does not match one or more files. See README for pattern rules.") 
    logger.debug("gen_all_matches() returns {}.".format(grp_match_dict_list))
    return grp_match_dict_list

def numstrvalue_to_int(ngrp_match_dict:dict)->dict:
    """Convert numeric str dictionary values to integers, if applicable
    Args:
        ngrp_match_dict: named groups to match dict, last value=filename
    Returns:
        ngrp_nummatch_dict: input dict, with numeric str values to int
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    logger.debug("numstrvalue_to_int() inputs: {}".format(ngrp_match_dict))
    ngrp_nummatch_dict = {}
    for key, value in ngrp_match_dict.items():
        try:
            ngrp_nummatch_dict[key] = int(value)
        except Exception:
            ngrp_nummatch_dict[key] = value
    logger.debug("numstrvalue_to_int() returns {}.".format(ngrp_nummatch_dict))
    return ngrp_nummatch_dict

def non_numstr_value_to_int(c_to_d_category:str, all_matches:list)->dict:
    """
    Make dictionary of category labels and numbers.
    This is used to later perform string to digit datatype conversion.
    
    Args:
        category: category group name to convert from char to digit
        all_matches: list of dicts, k=capture grps, v=match, last item has fname info
    Returns:
        cat_index_dict: dict key=category name, value=index after sorting
    """
    #: Return reference to logger instance
    logger = logging.getLogger("main")
    #: Generate list of strings belonging to the given category (element).
    cat_str_list = sorted(list(set([x[c_to_d_category] for x in all_matches])))
    cat_index_dict = dict()
    for i in range(0, len(cat_str_list)):
        cat_index_dict[cat_str_list[i]] = i
    logger.debug("non_numstr_value_to_int() returns {}.".format(cat_index_dict))
    return cat_index_dict

if __name__ == "__main__":
    #: Initialize the logger
    logging.basicConfig(
        format = "%(asctime)s - %(name) - 8s - %(levelname) - 8s - %(message)s",
        datefmt = "%d-%b-%y %H:%M:%S"
        )
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    #: Set up the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog="main",
        description="""
            Renames files using patterns, such as dd or c+"""
        )
    parser.add_argument(
        "--inpDir",
        dest = "inpDir", type=str, 
        help="Input image collection to be processed by this plugin",
        required=True
        )
    parser.add_argument(
        "--outDir", dest="outDir", type=str, 
        help="Output image collection of renamed files",
        required=True
        )
    parser.add_argument(
        "--filePattern", dest="filePattern", type=str, 
        help="Filename pattern used to separate data",
        required=True
        )
    parser.add_argument(
        "--outFilePattern", dest="outFilePattern", type=str, 
        help="Desired filename pattern used to rename and separate data",
        required=True
        )
    #: Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    #: outDir is the new collection
    outDir = args.outDir
    #: Input pattern is the regex pattern input by user
    filePattern = args.filePattern
    #: Output pattern is regex pattern expected by user
    outFilePattern = args.outFilePattern
    
    """Inputs"""
    inp_files = list(Path(inpDir).iterdir())
    inp_pattern = filePattern
    out_pattern = outFilePattern
    
    """Process Inputs"""
    #: Generate dict of named regular expression groups
    regex_patterns = pattern_to_regex(inp_pattern)
    #: Update file pattern to be regex-compatible, using dict of regex_patterns
    inp = pattern_to_raw_f_string(inp_pattern, regex_patterns)
    #: Convert out pattern to fstring to be used for str formatting
    out_pattern_fstring = pattern_to_fstring(out_pattern)
    #: List category names where input, output patterns are c, d)
    category_list = replace_cat_label(inp_pattern, out_pattern)
    #: List dict for each filename, capture groups, and corresponding matches
    all_matches = gen_all_matches(inp, inp_files)
    #: Iterate through matches
    for i in range(0, len(all_matches)):
        tmp_match = all_matches[i]
        #: Convert numeric str dictionary values to int, if applicable
        all_matches[i] = numstrvalue_to_int(tmp_match)
    char_to_num = dict()
    for c_to_d_category in category_list:
        char_to_num[c_to_d_category] = non_numstr_value_to_int(
            c_to_d_category, all_matches)
    #: Convert tmp_match to integers, if applicable
    for c_to_d_category in category_list:
        for i in range(0, len(all_matches)):
            if all_matches[i].get(c_to_d_category):
                all_matches[i][c_to_d_category] = char_to_num[c_to_d_category][
                    all_matches[i][c_to_d_category]]
    for match in all_matches:
        out_name = Path(outDir).resolve() / out_pattern_fstring.format(**match)
        logger.info(f'old file name {match["fname"]} and new file name {out_name}')
        #: Copy renamed file to output directory
        shutil.copy2(match["fname"], out_name)