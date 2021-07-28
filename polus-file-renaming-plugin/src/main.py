import logging
import argparse
from pathlib import Path
import re
import shutil
def pattern_to_regex(pattern:str) -> dict:
    """Here, we add in a "named regular expression (capture) group: ?p<>
    
    Looks at pattern. Finds word: dd or cc or d+
    
    findall() finds all the matches and returns them as a list of strings, 
    with each string representing one match
    Source: https://developers.google.com/edu/python/regular-expressions#findall)
    
    
    Args:
        pattern: "img_x{row:dd}_y{col:dd}_{channel:c+}.tif"
    Returns:
        regex_patterns: {'row': '(?P<row>[0-9][0-9])', 'col': '(?P<col>[0-9][0-9])','channel': '(?P<channel>[a-zA-Z]+)'}
    """
    patterns = re.findall(r'\{(\w+):([dc+]+)\}', pattern)
    pattern_map = {
    'd' : r'[0-9]',
    'c' : r'[a-zA-Z]',
    '+' : '+'
    }
    regex_patterns = {}
    for var, pat in patterns:
        pp = ''.join([pattern_map[p] for p in pat])
        regex_patterns[var] = fr'(?P<{var}>{pp})'
    
    return regex_patterns


def pattern_to_raw_f_string(pattern:str, regex_patterns:dict)->str:
    """
    Here we create an f strings
    (cleaner in python 3.7 than using .format())
    You can create raw f-strings by using the prefix “fr” 
    Source: https://cito.github.io/blog/f-strings/
    
    Args:
        pattern: "img_x{row:dd}_y{col:dd}_{channel:c+}.tif"
        regex_patterns: {'row': '(?P<row>[0-9][0-9])', 'col': '(?P<col>[0-9][0-9])', 'channel': '(?P<channel>[a-zA-Z]+)'}
    Returns:
        inp: "img_x(?P<row>[0-9][0-9])_y(?P<col>[0-9][0-9])_(?P<channel>[a-zA-Z]+).tif"
    """
    
    inp = pattern
    for k, v in regex_patterns.items():
        inp = re.sub(fr'\{{{k}:.*?\}}', v, inp)
    return inp

def gen_all_matches(inp:str, inp_files:list)->dict:
    """
    Get matches from input pattern and input filename
    
    Generate a list of dictionaries, where each dictionary 
    is the named regular expression capture group and 
    corresponding match.
    
    Args:
        inp: input pattern as f string (contains ?P for capturing regex capture groups)
    
    Returns:
        all_matches: list of dicts, where keys are regex capture group names and correspnding matches from filename
    
    """
    all_matches =[]
    for x in inp_files:
        tmp = re.match(inp, x.name).groupdict()
        tmp["name"] = x
        all_matches.append(tmp)
    return all_matches

def pattern_to_fstring(out_pattern:str)->str:
    
    """
    Convert outpattern to format string, :03d 
    Args:
        out_patterns: 'newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.tif'
    
    Returns:
        out_pattern_fstring: 'newdata_x{row:03d}_y{col:03d}_c{channel:03d}.tif'
    """
    out_patterns = re.findall(r'\{(\w+):([dc+]+)\}', out_pattern)
    f_string_dict = {}
    for key, value in out_patterns:
        temp_value = value[:1]
        if "+" not in value:
            if temp_value == "c":
                temp_value = "s"
                f_string_dict[key] = "{" + key + ":" + str(len(value)) + temp_value + "}"
            else:
                # Preceding the width field by a zero ('0') character enables sign-aware zero-padding for numeric types
                f_string_dict[key] = "{" + key + ":0" + str(len(value)) + temp_value + "}"
        else:
            if temp_value == "c":
                temp_value = "s"
                f_string_dict[key] = "{" + key  + ":" + temp_value + "}"
            else:
                f_string_dict[key] = "{" + key  + ":0" + temp_value + "}"
    
    out_pattern_fstring = out_pattern
    for named_group, fstring in f_string_dict.items():
        out_pattern_fstring = re.sub(fr'\{{{named_group}:.*?\}}', fstring, out_pattern_fstring)
    return out_pattern_fstring

def convert_match_to_int(tmp_match):
    """Convert tmp_match to integers, if applicable
    """
    new_tmp_match = {}
    for key, value in tmp_match.items():
        try:
            new_tmp_match[key] = int(value)
        except Exception:
            new_tmp_match[key] = value
    return new_tmp_match

def replace_cat_label(inp_pattern:str, out_pattern:str):
    """This function replaces with the categorical label
    
    Args:
        inp_pattern
        out_pattern:
    Returns:
        
    """
    #: Generate list [('row', 'dd'), ('col', 'dd'), ('channel', 'c+')]
    in_pattts = re.findall(r'\{(\w+):([dc+]+)\}', inp_pattern)
    out_pattts = re.findall(r'\{(\w+):([dc+]+)\}', out_pattern)
    
    #: ['channel'] If input is c and output starts with d, store unique key in list
    my_list = list(set([a for (a,b) in in_pattts for (c,d) in out_pattts  if b.startswith("c") and d.startswith("d")]))
    
    return my_list

def dict_str_to_digit(element, all_matches):
    """
    Perform string to digit datatype conversion
    """
    #tmp_match2 = new_tmp_match
    #: find the index of item in the list
    set_list = sorted(list(set([x[element] for x in all_matches])))
    indices = dict()
    for i in range(0, len(set_list)):
        indices[set_list[i]] = i
    return indices

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
    #: Label categories where input, output patterns are c, d)
    category_list = replace_cat_label(inp_pattern, out_pattern)
    #: List dict for each filename, capture groups, and corresponding matches
    all_matches = gen_all_matches(inp, inp_files)
    for i in range(0, len(all_matches)):
        tmp_match = all_matches[i]
        all_matches[i] = convert_match_to_int(tmp_match)
    char_to_num = dict()
    for element in category_list:
        char_to_num[element] = dict_str_to_digit(element, all_matches)
    #: Convert tmp_match to integers, if applicable
    for element in category_list:
        for i in range(0, len(all_matches)):
            if all_matches[i].get(element):
                all_matches[i][element] = char_to_num[element][all_matches[i][element]]
    for match in all_matches:
        new_name = Path(outDir).resolve() / out_pattern_fstring.format(**match)
        logger.info(f'old name {match["name"]} and new name {new_name}')
        shutil.copy2(match["name"], new_name)