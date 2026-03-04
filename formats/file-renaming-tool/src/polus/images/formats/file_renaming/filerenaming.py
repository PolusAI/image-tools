"""File Renaming."""
import logging
import os
import pathlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

import filepattern as fp
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


def get_num_threads() -> int:
    """Return thread count from NUM_THREADS env or a safe I/O-bound default."""
    try:
        if env := os.getenv("NUM_THREADS"):
            return max(1, int(env))
    except ValueError:
        pass

    return min(32, (os.cpu_count() or 1) * 4)


NUM_THREADS = get_num_threads()


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


def str_to_int(dictionary: dict[str, Any]) -> dict[str, Any]:
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
        except (ValueError, TypeError):
            fixed_dictionary[key] = value
    logger.debug(f"str_to_int() returns {fixed_dictionary}")
    return fixed_dictionary


def letters_to_int(named_grp: str, all_matches: list[dict[str, Any]]) -> dict[str, int]:
    """Alphabetically number matches for the given named group for all files.

    Make a dictionary where each key is a match for each filename and
    the corresponding value is a number indicating its alphabetical rank,
    with single-letter keys sorted first, followed by double-letter keys.

    Args:
        named_grp: Group with c in input pattern and d in out pattern.
        all_matches: list of dicts, k=grps, v=match, last item=file name.

    Returns:
        cat_index_dict: dict key=category name, value=index after sorting.
    """
    logger.debug(f"letters_to_int() inputs: {named_grp}, {all_matches}")

    # Generate a set of unique matches for the given group
    matches = {namedgrp_match_dict[named_grp] for namedgrp_match_dict in all_matches}

    # Sort with single-letter keys first, then double-letter keys
    alphabetized_matches = sorted(matches, key=lambda x: (len(x) > 1, x))

    # Create a dictionary mapping each match to its alphabetical rank
    str_alphabetindex_dict = {match: i for i, match in enumerate(alphabetized_matches)}

    logger.debug(f"letters_to_int() returns {str_alphabetindex_dict}")
    return str_alphabetindex_dict


def _prepare_file_matches(
    inp_dir: pathlib.Path,
    file_pattern: str,
    out_file_pattern: str,
    map_directory: bool | None = False,
) -> tuple[list[Any], list[str], str, bool, dict[str, Any] | None]:
    """Validate inputs and prepare transformed file matches.

    Returns:
        Tuple of (inp_files, fpaths, out_pattern_fstring, check_dir_var, map_dict)
    """
    # Check if the directory is empty without creating a full list
    file_count = sum(1 for _ in inp_dir.iterdir())
    if file_count == 0:
        msg = f"Input directory is empty: {file_count} files found."
        raise ValueError(msg)

    logger.info(f"Number of files found: {file_count}")

    recursive = bool(map_directory)
    files = fp.FilePattern(inp_dir, file_pattern, recursive=recursive)

    if len(files) == 0:
        msg = f"Please define filePattern: {file_pattern} again!"
        raise ValueError(msg)

    inp_files: list[Any] = [file[0] for file in files()]
    fpaths: list[str] = [file[1] for file in files()]

    # Integrate format strings into outFilePattern to specify digit/char len
    out_pattern_fstring = specify_len(out_file_pattern)

    # List named groups where input pattern=char & output pattern=digit
    char_to_digit_categories = get_char_to_digit_grps(file_pattern, out_file_pattern)

    # Convert numbers from strings to integers, if applicable
    for i in range(len(inp_files)):
        inp_files[i] = str_to_int(inp_files[i])

    # Populate dict if any matches need to be converted from char to digit
    # Key=named group, Value=Int representing matched chars
    numbered = {grp: letters_to_int(grp, inp_files) for grp in char_to_digit_categories}

    # Check named groups that need c->d conversion
    for named_grp in char_to_digit_categories:
        for i in range(len(inp_files)):
            if inp_files[i].get(named_grp):
                #: Replace original matched letter with new digit
                inp_files[i][named_grp] = numbered[named_grp][inp_files[i][named_grp]]

    # To create a dictionary mapping for folder names,
    # The keys represent folder names and the values represent corresponding mappings.
    check_dir_var = bool([d for d in inp_files if "directory" in list(d.keys())])
    map_dict = None
    if map_directory:
        if not check_dir_var:
            logger.error("directory variable is not included in filepattern correctly")
        else:
            subdirs = sorted({d["directory"] for d in inp_files if d["directory"]})
            map_dict = dict(zip(subdirs, [f"d{i}" for i in range(1, len(subdirs) + 1)]))

    return inp_files, fpaths, out_pattern_fstring, check_dir_var, map_dict


class _ResolvePathArgs:
    """Arguments for _resolve_output_path."""

    def __init__(  # noqa: PLR0913
        self,
        match: dict[str, Any],
        out_dir: pathlib.Path,
        out_pattern_fstring: str,
        check_dir_var: bool,
        map_directory: bool | None,
        map_dict: dict[str, Any] | None,
    ) -> None:
        self.match = match
        self.out_dir = out_dir
        self.out_pattern_fstring = out_pattern_fstring
        self.check_dir_var = check_dir_var
        self.map_directory = map_directory
        self.map_dict = map_dict


def _resolve_output_path(args: _ResolvePathArgs) -> pathlib.Path | None:
    """Resolve the output file path for a single file match."""
    # Apply str formatting to change digit or char length
    out_name = args.out_pattern_fstring.format(**args.match)

    if not args.check_dir_var:
        return pathlib.Path(args.out_dir, out_name)

    directory = args.match.get("directory")
    if args.map_directory:
        if not args.map_dict or directory not in args.map_dict:
            logger.error(f"{directory} is not provided in filePattern")
            return None
        return pathlib.Path(args.out_dir, f"{args.map_dict[directory]}_{out_name}")

    return pathlib.Path(args.out_dir, f"{directory}_{out_name}")


def rename(
    inp_dir: pathlib.Path,
    out_dir: pathlib.Path,
    file_pattern: str,
    out_file_pattern: str,
    map_directory: bool | None = False,
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

    (
        inp_files,
        fpaths,
        out_pattern_fstring,
        check_dir_var,
        map_dict,
    ) = _prepare_file_matches(inp_dir, file_pattern, out_file_pattern, map_directory)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        threads = []
        for match, p in zip(inp_files, fpaths):
            try:
                args = _ResolvePathArgs(
                    match,
                    out_dir,
                    out_pattern_fstring,
                    check_dir_var,
                    map_directory,
                    map_dict,
                )
                out_path = _resolve_output_path(args)
                if out_path is None:
                    continue
                old_file_name = pathlib.Path(inp_dir, p[0])
                threads.append(executor.submit(shutil.copy2, old_file_name, out_path))
            except ValueError:
                logger.error(f"filePattern:{file_pattern} is incorrectly defined!!!")

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
