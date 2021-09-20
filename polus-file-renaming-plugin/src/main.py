import argparse, logging
import re
import shutil
import os
from pathlib import Path
from typing import Union

"""This is the order in which functions are called:

main(): Receives parsed inputs and copies renamed files to new dir
map_pattern_grps_to_regex(): Get group names from pattern. Rewrite in regex.
convert_to_regex(): Integrate regex into original file pattern using map.
specify_len(): Update pattern to output correct digit/character length
get_char_to_digit_grps(): List named groups where inp pattern=c & out pattern=d.
extract_named_grp_matches(): Get matches belonging to named grps from files
str_to_int(): Convert numbers in a dictionary from str to int.
letters_to_int(): Make dictionary of category labels and numbers.

"""

#: Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')
# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

def map_pattern_grps_to_regex(file_pattern:str) -> dict:
    """Get group names from pattern. Convert patterns (c+ or dd) to regex.
    
    Args:
        file_pattern: File pattern, with special characters escaped
        
    Returns:
        rgx_patterns: The key is a named regex group. The value is regex.
        
    """
    logger.debug("pattern_to_regex() inputs: {}".format(file_pattern))
    #: Extract the group name and associated pattern (ex: {row:dd})
    group_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", file_pattern)
    pattern_map = {
        "d" : r"[0-9]",
        "c" : r"[a-zA-Z]",
        "+" : "+"
        }
    rgx_patterns = {}
    for group_name, groups_pattern in group_and_pattern_tuples:
        rgx = "".join([pattern_map[pattern] for pattern in groups_pattern])
        #: ?P<foo> is included to specify that foo is a named group.
        rgx_patterns[group_name] = fr"(?P<{group_name}>{rgx})"
    logger.debug("pattern_to_regex() returns {}".format(rgx_patterns))
    
    return rgx_patterns

def convert_to_regex(file_pattern:str, extracted_rgx_patterns:dict)->str:
    """Integrate regex into original file pattern.
    
    The extracted_rgx_patterns helps replace simple patterns (ie. dd, c+) 
    with regex in the correct location, based on named groups.
    
    Args:
        file_pattern: file pattern provided by the user
        extracted_rgx_patterns: named group and regex value dictionary
        
    Returns:
        new_pattern: file pattern converted to regex
    
    """
    logger.debug(
        "convert_to_regex() inputs: {}, {}". format(
            file_pattern, extracted_rgx_patterns
            )
        )
    rgx_pattern = file_pattern
    for named_grp, regex_str in extracted_rgx_patterns.items():
        #: The prefix "fr" creates raw f-strings, which act like format()
        rgx_pattern = re.sub(fr"\{{{named_grp}:.*?\}}", regex_str, rgx_pattern)
    logger.debug("convert_to_regex() returns {}".format(rgx_pattern))
    return rgx_pattern

def specify_len(out_pattern:str)->str:
    """Update output file pattern to output correct number of digits.
    
    After extracting group names and associated patterns from the 
    outFilePattern, integrate format strings into the file pattern to 
    accomplish.
    
    Example:
        "newdata_x{row:ddd}" becomes "new_data{row:03d}".
  
    Args:
        out_pattern: output file pattern provided by the user
    
    Returns:
        new_out_pattern: file pattern converted to format string
        
    """
    logger.debug("specify_len() inputs: {}".format(out_pattern))
    #: Extract the group name and associated pattern (ex: {row:dd})
    group_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", out_pattern)
    grp_rgx_dict = {}
    #: Convert simple file patterns to format strings (ex: ddd becomes :03d). 
    for group_name, groups_pattern in group_and_pattern_tuples:
        # Get the length of the string if not variable width
        s_len = "" if "+" in groups_pattern else str(len(groups_pattern))
        # Set the formatting value
        temp_pattern = "s" if groups_pattern[0] == "c" else "d"
        # Prepend a 0 for padding digit format
        if temp_pattern == "d":
            s_len = "0" + s_len
        grp_rgx_dict[group_name] = "{" + group_name + ":" + s_len + temp_pattern + "}"  
    new_out_pattern = out_pattern
    for named_group, format_str in grp_rgx_dict.items():
        new_out_pattern = re.sub(
            fr"\{{{named_group}:.*?\}}", format_str, new_out_pattern
            )
    logger.debug("specify_len() returns {}".format(new_out_pattern))
    
    return new_out_pattern


def get_char_to_digit_grps(inp_pattern:str, out_pattern:str)->list:
    """Return group names where input and output datatypes differ.
        
    If the input pattern is a character and the output pattern is a 
    digit, return the named group associated with those patterns.
    
    Args:
        inp_pattern: Original input pattern
        out_pattern: Original output pattern
        
    Returns:
        special_categories: Named groups with c to d conversion or [None]
        
    """
    logger.debug(
        "get_char_to_digit_grps() inputs: {}, {}".format(inp_pattern, out_pattern))
    #: Extract the group name and associated pattern (ex: {row:dd})
    ingrp_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", inp_pattern)
    outgrp_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", out_pattern)
    
    #: Get group names where input pattern is c and output pattern is d
    special_categories = []
    for out_grp_name in dict(outgrp_and_pattern_tuples).keys():
        if dict(ingrp_and_pattern_tuples)[out_grp_name].startswith("c") and \
            dict(outgrp_and_pattern_tuples)[out_grp_name].startswith("d"):
                special_categories.append(out_grp_name)
    logger.debug("get_char_to_digit_grps() returns {}".format(special_categories))
    return special_categories

def extract_named_grp_matches(rgx_pattern:str, inp_files:list)->list:
    """Store matches from the substrings from each filename that vary.
    
    Loop through each file. Apply the regex pattern to each 
    filename. When a match occurs for a named group, add that match to 
    a dictionary, where the key is the named (regex capture) group and 
    the value is the corresponding match from the filename.
    
    Args:
        rgx_pattern: input pattern in regex format
        inp_files: list of files in input directory
    
    Returns:
        grp_match_dict_list: list of dictionaries containing str matches
        
    """
    logger.debug(
        "extract_named_grp_matches() inputs: {}, {}".format(rgx_pattern, inp_files)
        )
    grp_match_dict_list =[]
    #: Build list of dicts, where key is capture group and value is match
    for filename in inp_files:
        try:
            grp_match_dict = re.match(rgx_pattern, filename).groupdict()
            #: Add filename information to dictionary
            grp_match_dict["fname"] = filename
            grp_match_dict_list.append(grp_match_dict)
        except AttributeError as e:
            logger.error(e)
            logger.error(
                "File pattern does not match one or more files. "
                "See README for pattern rules."
                )
            raise AttributeError(
                "File pattern does not match one or more files. "
                "Check that each named group in your file pattern is unique. "
                "See README for pattern rules."
                )
        except Exception as e:
            if str(e).startswith("redefinition of group name"):
                logger.error(
                    "Ensure that named groups in file patterns are unique. "
                    "({})".format(e)
                    )
                raise ValueError(
                    "Ensure that named groups in file patterns are unique. "
                    "({})".format(e)
                    )
            else:
                raise ValueError(
                    "Something went wrong. See README for pattern rules. "
                    "({})".format(e)
                    )
    logger.debug("extract_named_grp_matches() returns {}".format(grp_match_dict_list))
    
    return grp_match_dict_list

def str_to_int(dictionary:dict)->dict:
    """If a number in the dictionary is in str format, convert to int.
    
    Args:
        dictionary: contains group, match, and filename info.
    
    Returns:
        fixed_dictionary: input dict, with numeric str values to int
    
    """
    logger.debug("str_to_int() inputs: {}".format(dictionary))
    fixed_dictionary = {}
    for key, value in dictionary.items():
        try:
            fixed_dictionary[key] = int(value)
        except Exception:
            fixed_dictionary[key] = value
    logger.debug("str_to_int() returns {}".format(fixed_dictionary))
    return fixed_dictionary

def letters_to_int(named_grp:str, all_matches:list)->dict:
    """Alphabetically number matches for the given named group for all files.
    
    Make a dictionary where each key is a match for each filename and 
    the corresponding value is a number indicating its alphabetical rank.
    
    Args:
        named_grp: Group with c in input pattern and d in out pattern
        all_matches: list of dicts, k=grps, v=match, last item=file name
    
    Returns:
        cat_index_dict: dict key=category name, value=index after sorting
    
    """
    logger.debug(
        "letters_to_int() inputs: {}, {}".format(named_grp, all_matches))
    #: Generate list of strings belonging to the given category (element).
    alphabetized_matches = sorted(list(set([
        namedgrp_match_dict[named_grp] for namedgrp_match_dict in all_matches
        ])))
    str_alphabetindex_dict = {}
    for i in range(0, len(alphabetized_matches)):
        str_alphabetindex_dict[alphabetized_matches[i]] = i
    logger.debug("letters_to_int() returns {}".format(str_alphabetindex_dict))
    return str_alphabetindex_dict

def main(inpDir: Union[Path, str],
         filePattern: str,
         outDir: Union[Path, str],
         outFilePattern: Path,
         ) -> dict:
    """Use parsed inputs to rename and copy files to a new directory.
    
    First, the script converts the input filePattern to regex.
    Next, it converts the output file pattern using format strings.
    Then, it replaces matched letters with a digit if needed.
    Finally, it copies the renamed file to a new directory.
    
    See README for pattern rules.
    
    Args:
        inpDir: Path to image collection. Empty str given for testing
        filePattern: Input file pattern
        outDir: Path to image collection storing copies of renamed files
        outFilePattern: Output file pattern
    
    Returns:
        output_dict: Dictionary of in to out file names, for testing

    """
    #: Used for testing json filenames with test.py
    if type(inpDir) == list:
        inp_files = inpDir
        parent_path = ""
    #: Used for processing WIPP files
    else:
        parent_path = list(inpDir.iterdir())[0].parent
        inp_files = [str(inp_file.name) for inp_file in list(inpDir.iterdir())]
    chars_to_escape = ['(', ")", ".", "[", "]", "$"]
    for char in chars_to_escape:
        filePattern = filePattern.replace(char, ("\\" + char))
    groupname_regex_dict= map_pattern_grps_to_regex(filePattern)
    #: Integrate regex from dictionary into original file pattern
    inp_pattern_rgx = convert_to_regex(filePattern, groupname_regex_dict)
    
    #: Integrate format strings into outFilePattern to specify digit/char len
    out_pattern_fstring = specify_len(outFilePattern)
    
    #: List named groups where input pattern=char & output pattern=digit
    char_to_digit_categories = get_char_to_digit_grps(filePattern, outFilePattern)
    
    #: List a dictionary (k=named grp, v=match) for each filename
    all_grp_matches = extract_named_grp_matches(inp_pattern_rgx, inp_files)
    #: Convert numbers from strings to integers, if applicable
    for i in range(0, len(all_grp_matches)):
        tmp_match = all_grp_matches[i]
        all_grp_matches[i] = str_to_int(tmp_match)

    #: Populate dict if any matches need to be converted from char to digit
    #: Key=named group, Value=Int representing matched chars
    numbered_categories = {}
    for named_grp in char_to_digit_categories:
        numbered_categories[named_grp] = letters_to_int(
            named_grp, all_grp_matches
            )
    # Check named groups that need c->d conversion
    for named_grp in char_to_digit_categories:
        for i in range(0, len(all_grp_matches)):
            if all_grp_matches[i].get(named_grp):
                #: Replace original matched letter with new digit
                all_grp_matches[i][named_grp] = numbered_categories[named_grp][
                    all_grp_matches[i][named_grp]]
    
    output_dict = {}
    for match in all_grp_matches:
        #: If running on WIPP
        if outDir != "":
            #: Apply str formatting to change digit or char length
            out_name = outDir.resolve() / out_pattern_fstring.format(**match)
            old_file_name = parent_path / match['fname']
            shutil.copy2(old_file_name, out_name)
        #: Enter outDir as an empty string for testing purposes
        elif outDir == "":
            out_name = out_pattern_fstring.format(**match)
            old_file_name = match['fname']
        logger.info(f"Old name {old_file_name} & new name {out_name}")
        #: Copy renamed file to output directory
        output_dict[old_file_name] = out_name
    #: Save old and new file names to dict (used for testing)
    return output_dict

if __name__ == "__main__":
    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a WIPP plugin.')
    
    # Input arguments
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--outFilePattern', dest='outFilePattern', type=str,
                        help='Desired filename pattern used to rename and separate data', required=True)
    
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args()
    inpDir = Path(args.inpDir)
    if (inpDir.joinpath('images').is_dir()):
        # Switch to images folder if present
        inpDir = inpDir.joinpath('images').absolute()
    logger.info('inpDir = {}'.format(inpDir))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(outDir))
    outFilePattern = args.outFilePattern
    logger.info('outFilePattern = {}'.format(outFilePattern))
        
    main(inpDir=inpDir,
        filePattern=filePattern,
        outDir=outDir, 
        outFilePattern=outFilePattern)