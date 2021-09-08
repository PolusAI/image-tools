import argparse
import logging
from pathlib import Path
import re
import shutil

"""
This is the order in which functions are called:

main(): Receives parsed inputs and copies renamed files to new dir
pattern_to_regex(): Convert pattern to regex; include named regex group
pattern_to_raw_f_string(): Create raw f strings (ex: (?P<row>[a-zA-Z]+))
pattern_to_fstring(): Convert outpattern to format string (ex: {row:03d})
replace_cat_label(): Map chars to nums if input is char and out is digit
gen_all_matches(): Get matches from input pattern and input filename
numstrvalue_to_int(): Convert numeric str dictionary values to integers
non_numstr_value_to_int(): Make dictionary of category labels and numbers.
"""

def main(inp_pattern, out_pattern, inp_files, outDir):
    """
    Use parsed inputs to rename and copy files to a new directory.
    
    Args:
        inp_pattern: Input pattern given by user
        out_pattern: Output pattern given by user
        inp_files: List of files in input image collection
    """
    """Process Inputs"""
    #: Initialize the logger
    logging.basicConfig(
        format = "%(asctime)s - %(name) - 8s - %(levelname) - 8s - %(message)s",
        datefmt = "%d-%b-%y %H:%M:%S"
        )
    #: Get reference to logger instance
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    #: Get folder name in which first given file resides
    if type(inp_files[0]) == str: #: Used for testing with json
        parent_path = ""
    else:
        parent_path = inp_files[0].parent
        #: Convert pathlib path files in list to string
        inp_files = [str(inp_file.name) for inp_file in inp_files]
    #: Generate dict of named regular expression groups
    logger.debug("pattern_to_regex() inputs: {}".format(inp_pattern))
    # Escape any existing parentheses and periods
    inp_pattern = inp_pattern.replace('(','\(').replace(')','\)')
    inp_pattern = inp_pattern.replace('.', '\.')
    regex_patterns = pattern_to_regex(inp_pattern)
    logger.debug("pattern_to_regex() returns {}.".format(regex_patterns))
    
    #: Convert file pattern regex, using dict of regex_patterns
    logger.debug(
        "pattern_to_raw_f_string() inputs: {}, {}". format(
            inp_pattern, regex_patterns)
        )
    inp = pattern_to_raw_f_string(inp_pattern, regex_patterns)
    logger.debug("pattern_to_raw_f_string() returns {}.".format(inp))
    #: Convert out pattern to fstring to be used for str formatting
    logger.debug("pattern_to_fstring() inputs: {}".format(out_pattern))
    out_pattern_fstring = pattern_to_fstring(out_pattern)
    logger.debug("pattern_to_fstring() returns {}.".format(out_pattern_fstring))
    #: List category names where input, output patterns are c, d)
    logger.debug(
        "replace_cat_label() inputs: {}, {}".format(inp_pattern, out_pattern))
    category_list = replace_cat_label(inp_pattern, out_pattern)
    logger.debug("replace_cat_label() returns {}.".format(category_list))
    #: List dict for each filename, capture grps & corresponding matches
    logger.debug("gen_all_matches() inputs: {}, {}".format(inp, inp_files))
    all_matches = gen_all_matches(inp, inp_files)
    logger.debug("gen_all_matches() returns {}.".format(all_matches))
    #: Iterate through matches
    for i in range(0, len(all_matches)):
        tmp_match = all_matches[i]
        #: Convert numeric str dictionary values to int, if applicable
        logger.debug("numstrvalue_to_int() inputs: {}".format(tmp_match))
        all_matches[i] = numstrvalue_to_int(tmp_match)
        logger.debug("numstrvalue_to_int() returns {}.".format(all_matches[i]))
    char_to_num = dict()
    for c_to_d_category in category_list:
        char_to_num[c_to_d_category] = non_numstr_value_to_int(
            c_to_d_category, all_matches)
        logger.debug("non_numstr_value_to_int() returns {}.".format(char_to_num[c_to_d_category]))
    #: Convert tmp_match to integers, if applicable
    for c_to_d_category in category_list:
        for i in range(0, len(all_matches)):
            if all_matches[i].get(c_to_d_category):
                all_matches[i][c_to_d_category] = char_to_num[c_to_d_category][
                    all_matches[i][c_to_d_category]]
    output_dict = {}
    for match in all_matches:
        #: outDir entered as an empty string for testing purposes
        if outDir != "":
            out_name = Path(outDir).resolve() / out_pattern_fstring.format(**match)
            old_file_name = parent_path / match['fname']
            shutil.copy2(old_file_name, out_name)
        #: If running on WIPP
        else:
            out_name = out_pattern_fstring.format(**match)
            old_file_name = match['fname']
        logger.info(f"Old file name {old_file_name} and new file name {out_name}")
        #: Copy renamed file to output directory
        output_dict[old_file_name] = out_name
    #: Save old and new file names to dict (used for testing)
    return output_dict

def pattern_to_regex(pattern:str) -> dict:
    """Add named regular expression group, ?p<>, and make pattern regex.
    
    Ultimately, this function builds a dictionary where the key is the 
    named regex group and the value is the regex.
    
    Args:
        pattern: file pattern provided by the user
    Returns:
        rgx_patterns: named group and regex value dictionary
    """
    #: Make list of (named group, user file pattern) tuples
    patterns = re.findall(r"\{(\w+):([dc+]+)\}", pattern)
    pattern_map = {
        "d" : r"[0-9]",
        "c" : r"[a-zA-Z]",
        "+" : "+"
        }
    rgx_patterns = {}
    for var, pat in patterns:
        pp = "".join([pattern_map[p] for p in pat])
        rgx_patterns[var] = fr"(?P<{var}>{pp})"
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
    rgx_pattern = pattern
    for named_grp, regex_str in rgx_patts.items():
        #: Using the prefix "fr" creates raw f-strings
        rgx_pattern = re.sub(fr"\{{{named_grp}:.*?\}}", regex_str, rgx_pattern)
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
    #: Make list of (named groups, user file pattern) tuples 
    out_patterns = re.findall(r"\{(\w+):([dc+]+)\}", out_pattern)
    f_string_dict = {}
    for key, value in out_patterns:
        # Get the length of the string if not variable width
        str_len = "" if "+" in value else str(len(value))
        # Set the formatting value
        temp_value = "s" if value[0] == "c" else "d"
        # Prepend a 0 for padding digit format
        if temp_value == "d":
            str_len = "0" + str_len
        f_string_dict[key] = "{" + key + ":" + str_len + temp_value + "}"  
    out_pattern_fstring = out_pattern
    for named_group, fstring in f_string_dict.items():
        out_pattern_fstring = re.sub(fr"\{{{named_group}:.*?\}}", fstring, out_pattern_fstring)
    return out_pattern_fstring

def replace_cat_label(inp_pattern:str, out_pattern:str)->list:
    """Return categorical labels based on the input and output patterns.
    
    If the input pattern is a character and the output pattern is a 
    digit, this generates a list of the string patterns captured that
    match this criteria.
    
    Args:
        inp_pattern: user input pattern
        out_pattern: user output pattern
    Returns:
        unique_keys: list of unique category labels to make numeric
    """
    #: Generate list [("row", "dd"), ("col", "dd"), ("channel", "c+"")]
    matched_in_patterns = re.findall(r"\{(\w+):([dc+]+)\}", inp_pattern)
    matched_out_patterns = re.findall(r"\{(\w+):([dc+]+)\}", out_pattern)
    unique_keys = []
    #: If input file pattern is c and output is d, list unique key
    unique_keys = list(set([
        inp_grp for (inp_grp,inp_rgx) in matched_in_patterns 
        for (out_grp, out_rgx) in matched_out_patterns  
        if inp_rgx.startswith("c") and out_rgx.startswith("d")
        ]))
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
    grp_match_dict_list =[]
    #: Build list of dicts, where key is capture group and value is match
    for inp_file in inp_files:
        try:
            grp_match_dict = re.match(rgx_pattern, inp_file).groupdict()
            #: Add filename information to dictionary
            grp_match_dict["fname"] = inp_file
            grp_match_dict_list.append(grp_match_dict)
        except AttributeError as e:
            logger.error(e)
            logger.error("File pattern does not match one or more files. See README for pattern rules.")
            
            raise AttributeError("File pattern does not match one or more files. Check that each named group in your file pattern is unique or see README for pattern rules.")
        except Exception as e:
            if str(e).startswith("redefinition of group name"):
                logger.error("Ensure that named groups in your file patterns have unique names. Original error: {}".format(e))
                raise ValueError("Ensure that named groups in your file patterns have unique names ({})".format(e))
            else:
                raise ValueError("Something went wrong. See README for pattern rules. Original error: {}".format(e))
    return grp_match_dict_list

def numstrvalue_to_int(ngrp_match_dict:dict)->dict:
    """Convert numeric str dictionary values to integers, if applicable
    Args:
        ngrp_match_dict: named groups to match dict, last value=filename
    Returns:
        ngrp_nummatch_dict: input dict, with numeric str values to int
    """
    ngrp_nummatch_dict = {}
    for key, value in ngrp_match_dict.items():
        try:
            ngrp_nummatch_dict[key] = int(value)
        except Exception:
            ngrp_nummatch_dict[key] = value
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
    #: Generate list of strings belonging to the given category (element).
    cat_str_list = sorted(list(set([x[c_to_d_category] for x in all_matches])))
    cat_index_dict = dict()
    for i in range(0, len(cat_str_list)):
        cat_index_dict[cat_str_list[i]] = i
    return cat_index_dict

if __name__ == "__main__":
    #: Initialize the logger
    logging.basicConfig(
        format = "%(asctime)s - %(name) - 8s - %(levelname) - 8s - %(message)s",
        datefmt = "%d-%b-%y %H:%M:%S"
        )
    #: Get reference to logger instance
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
    #: Pass parsed inputs to main()
    main(filePattern, outFilePattern, inp_files, outDir)