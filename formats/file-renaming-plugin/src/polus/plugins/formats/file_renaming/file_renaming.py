"""File Renaming."""
import logging
import os
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Import environment variables
POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")


def map_pattern_grps_to_regex(file_pattern: str) -> dict:
    """Get group names from pattern. Convert patterns (c+ or dd) to regex.

    Args:
        file_pattern: File pattern, with special characters escaped.
    Returns:
        rgx_patterns: The key is a named regex group. The value is regex.
    """
    logger.debug(f"pattern_to_regex() inputs: {file_pattern}")
    #: Extract the group name and associated pattern (ex: {row:dd})
    group_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", file_pattern)
    pattern_map = {"d": r"[0-9]", "c": r"[a-zA-Z]", "+": "+"}
    rgx_patterns = {}
    for group_name, groups_pattern in group_and_pattern_tuples:
        rgx = "".join([pattern_map[pattern] for pattern in groups_pattern])
        #: ?P<foo> is included to specify that foo is a named group.
        rgx_patterns[group_name] = rf"(?P<{group_name}>{rgx})"
    logger.debug(f"pattern_to_regex() returns {rgx_patterns}")

    return rgx_patterns


def convert_to_regex(file_pattern: str, extracted_rgx_patterns: Dict) -> str:
    """Integrate regex into original file pattern.

    The extracted_rgx_patterns helps replace simple patterns (ie. dd, c+)
    with regex in the correct location, based on named groups.

    Args:
        file_pattern: file pattern provided by the user.
        extracted_rgx_patterns: named group and regex value dictionary.
    Returns:
        new_pattern: file pattern converted to regex.
    """
    logger.debug(f"convert_to_regex() inputs: {file_pattern}, {extracted_rgx_patterns}")
    rgx_pattern = file_pattern
    for named_grp, regex_str in extracted_rgx_patterns.items():
        #: The prefix "fr" creates raw f-strings, which act like format()
        rgx_pattern = re.sub(rf"\{{{named_grp}:.*?\}}", regex_str, rgx_pattern)
    logger.debug(f"convert_to_regex() returns {rgx_pattern}")
    return rgx_pattern


def specify_len(out_pattern: str) -> str:
    """Update output file pattern to output correct number of digits.

    After extracting group names and associated patterns from the
    outFilePattern, integrate format strings into the file pattern to
    accomplish.

    Example:
        "newdata_x{row:ddd}" becomes "new_data{row:03d}".

    Args:
        out_pattern: output file pattern provided by the user.
    Returns:
        new_out_pattern: file pattern converted to format string.
    """
    logger.debug(f"specify_len() inputs: {out_pattern}")
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
            rf"\{{{named_group}:.*?\}}", format_str, new_out_pattern
        )
    logger.debug(f"specify_len() returns {new_out_pattern}")

    return new_out_pattern


def get_char_to_digit_grps(inp_pattern: str, out_pattern: str) -> List[str]:
    """Return group names where input and output datatypes differ.

    If the input pattern is a character and the output pattern is a
    digit, return the named group associated with those patterns.

    Args:
        inp_pattern: Original input pattern.
        out_pattern: Original output pattern.

    Returns:
        special_categories: Named groups with c to d conversion or [None].
    """
    logger.debug(f"get_char_to_digit_grps() inputs: {inp_pattern}, {out_pattern}")
    #: Extract the group name and associated pattern (ex: {row:dd})
    ingrp_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", inp_pattern)
    outgrp_and_pattern_tuples = re.findall(r"\{(\w+):([dc+]+)\}", out_pattern)

    #: Get group names where input pattern is c and output pattern is d
    special_categories = []
    for out_grp_name in dict(outgrp_and_pattern_tuples).keys():
        if dict(ingrp_and_pattern_tuples)[out_grp_name].startswith("c") and dict(
            outgrp_and_pattern_tuples
        )[out_grp_name].startswith("d"):
            special_categories.append(out_grp_name)
    logger.debug(f"get_char_to_digit_grps() returns {special_categories}")
    return special_categories


def extract_named_grp_matches(
    rgx_pattern: str, inp_files: List
) -> List[Dict[str, Union[str, Any]]]:
    """Store matches from the substrings from each filename that vary.

    Loop through each file. Apply the regex pattern to each
    filename. When a match occurs for a named group, add that match to
    a dictionary, where the key is the named (regex capture) group and
    the value is the corresponding match from the filename.

    Args:
        rgx_pattern: input pattern in regex format.
        inp_files: list of files in input directory.

    Returns:
        grp_match_dict_list: list of dictionaries containing str matches.
    """
    logger.debug(f"extract_named_grp_matches() inputs: {rgx_pattern}, {inp_files}")
    grp_match_dict_list = []
    #: Build list of dicts, where key is capture group and value is match
    for filename in inp_files:
        try:
            d = re.match(rgx_pattern, filename)
            if d is None:
                break
            grp_match_dict = d.groupdict()
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
    logger.debug(f"extract_named_grp_matches() returns {grp_match_dict_list}")

    return grp_match_dict_list


def str_to_int(dictionary: dict) -> dict:
    """If a number in the dictionary is in str format, convert to int.

    Args:
        dictionary: contains group, match, and filename info.

    Returns:
        fixed_dictionary: input dict, with numeric str values to int.
    """
    logger.debug(f"str_to_int() inputs: {dictionary}")
    fixed_dictionary = {}
    for key, value in dictionary.items():
        try:
            fixed_dictionary[key] = int(value)
        except Exception:
            fixed_dictionary[key] = value
    logger.debug(f"str_to_int() returns {fixed_dictionary}")
    return fixed_dictionary


def letters_to_int(named_grp: str, all_matches: list) -> Dict:
    """Alphabetically number matches for the given named group for all files.

    Make a dictionary where each key is a match for each filename and
    the corresponding value is a number indicating its alphabetical rank.

    Args:
        named_grp: Group with c in input pattern and d in out pattern.
        all_matches: list of dicts, k=grps, v=match, last item=file name.
    Returns:
        cat_index_dict: dict key=category name, value=index after sorting.
    """
    logger.debug(f"letters_to_int() inputs: {named_grp}, {all_matches}")
    #: Generate list of strings belonging to the given category (element).
    alphabetized_matches = sorted(
        list({namedgrp_match_dict[named_grp] for namedgrp_match_dict in all_matches})
    )
    str_alphabetindex_dict = {}
    for i in range(0, len(alphabetized_matches)):
        str_alphabetindex_dict[alphabetized_matches[i]] = i
    logger.debug(f"letters_to_int() returns {str_alphabetindex_dict}")
    return str_alphabetindex_dict
