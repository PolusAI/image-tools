"""File Renaming."""
import logging
import os
import pathlib
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from sys import platform
from typing import Any
from typing import Optional

import filepattern as fp
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

if platform == "linux" or platform == "linux2":
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    NUM_THREADS = max(cpu_count() // 2, 2)


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
            rf"\{{{named_group}:.*?\}}",
            format_str,
            new_out_pattern,
        )
    logger.debug(f"specify_len() returns {new_out_pattern}")

    return new_out_pattern


def get_char_to_digit_grps(inp_pattern: str, out_pattern: str) -> list[str]:
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
    for out_grp_name in dict(outgrp_and_pattern_tuples):
        if dict(ingrp_and_pattern_tuples)[out_grp_name].startswith("c") and dict(
            outgrp_and_pattern_tuples,
        )[out_grp_name].startswith("d"):
            special_categories.append(out_grp_name)
    logger.debug(f"get_char_to_digit_grps() returns {special_categories}")
    return special_categories


def str_to_int(dictionary: dict) -> dict:
    """If a number in the dictionary is in str format, convert to int.

    Args:
        dictionary: contains group, match, and filename info.

    Returns:
        fixed_dictionary: input dict, with numeric str values to int.
    """
    fixed_dictionary = {}
    for key, value in dictionary.items():
        try:
            fixed_dictionary[key] = int(value)
        except Exception:  # noqa: BLE001
            fixed_dictionary[key] = value
    logger.debug(f"str_to_int() returns {fixed_dictionary}")
    return fixed_dictionary


def letters_to_int(named_grp: str, all_matches: list) -> dict:
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
        {namedgrp_match_dict[named_grp] for namedgrp_match_dict in all_matches},
    )
    str_alphabetindex_dict = {}
    for i in range(0, len(alphabetized_matches)):
        str_alphabetindex_dict[alphabetized_matches[i]] = i
    logger.debug(f"letters_to_int() returns {str_alphabetindex_dict}")
    return str_alphabetindex_dict


def rename(  # noqa: C901
    inp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    file_pattern: str,
    out_file_pattern: str,
    map_directory: Optional[bool] = False,
) -> None:
    """Scalable Extraction of Nyxus Features.

    Args:
        inp_dir : Path to image collection.
        out_dir : Path to image collection storing copies of renamed files.
        file_pattern : Input file pattern.
        out_file_pattern : Output file pattern.
        map_directory : Mapping of folder name.
    """
    logger.info("Start renaming files")

    files = fp.FilePattern(inp_dir, file_pattern, recursive=True)

    if len(files) == 0:
        msg = f"Please define filePattern: {file_pattern} again!!"
        raise ValueError(
            msg,
        )

    inp_files: list[Any] = [file[0] for file in files()]
    fpaths: list[str] = [file[1] for file in files()]

    #: Integrate format strings into outFilePattern to specify digit/char len
    out_pattern_fstring = specify_len(out_file_pattern)

    #: List named groups where input pattern=char & output pattern=digit
    char_to_digit_categories = get_char_to_digit_grps(file_pattern, out_file_pattern)

    #: Convert numbers from strings to integers, if applicable
    for i in range(0, len(inp_files)):
        tmp_match = inp_files[i]
        inp_files[i] = str_to_int(tmp_match)

    #: Populate dict if any matches need to be converted from char to digit
    #: Key=named group, Value=Int representing matched chars
    numbered_categories = {}
    for named_grp in char_to_digit_categories:
        numbered_categories[named_grp] = letters_to_int(named_grp, inp_files)
    # Check named groups that need c->d conversion
    for named_grp in char_to_digit_categories:
        for i in range(0, len(inp_files)):
            if inp_files[i].get(named_grp):
                #: Replace original matched letter with new digit
                inp_files[i][named_grp] = numbered_categories[named_grp][
                    inp_files[i][named_grp]
                ]
    # To create a dictionary mapping for folder names,
    # The keys represent folder names and the values represent corresponding mappings.
    check_dir_var = bool([d for d in inp_files if "directory" in list(d.keys())])
    if map_directory:
        if check_dir_var is False:
            logger.error("directory variable is not included in filepattern correctly")

        else:
            subdirs = sorted({d["directory"] for d in inp_files if d["directory"]})
            map_dirs = [f"d{i}" for i in range(1, len(subdirs) + 1)]
            map_dict = dict(zip(subdirs, map_dirs))

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        threads = []
        for match, p in zip(inp_files, fpaths):
            if check_dir_var is True:
                # Apply str formatting to change digit or char length
                out_name = out_pattern_fstring.format(**match)
                if map_directory:
                    try:
                        out_path = pathlib.Path(
                            out_dir,
                            f"{map_dict[match['directory']]}_{out_name}",
                        )
                    except ValueError:
                        logger.error(
                            f"{match['directory']} is not provided in filePattern",
                        )

                if not map_directory:
                    try:
                        out_path = pathlib.Path(
                            out_dir,
                            f"{ match['directory']}_{out_name}",
                        )
                    except ValueError:
                        logger.error(
                            f"{match['directory']} is not provided in filePattern",
                        )

                old_file_name = pathlib.Path(inp_dir, p[0])
                threads.append(executor.submit(shutil.copy2, old_file_name, out_path))

            if check_dir_var is False and not map_directory:
                try:
                    # Apply str formatting to change digit or char length
                    out_name = out_pattern_fstring.format(**match)
                    out_path = pathlib.Path(out_dir, out_name)
                    old_file_name = pathlib.Path(inp_dir, p[0])
                    threads.append(
                        executor.submit(shutil.copy2, old_file_name, out_path),
                    )
                except ValueError:
                    logger.error(
                        f"filePattern:{file_pattern} is incorrectly defined!!!",
                    )

        for f in tqdm(
            as_completed(threads),
            total=len(threads),
            mininterval=5,
            desc="Renaming images",
            initial=0,
            unit_scale=True,
            colour="cyan",
        ):
            f.result()
