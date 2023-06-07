"""File Renaming."""
import json
import os
import logging
import pathlib
import shutil
from typing import Any, Optional

import typer

from polus.plugins.formats.file_renaming import file_renaming as fr

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.plugins.formats.file_renaming")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))

@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collections",
    ),
    file_pattern: str = typer.Option(
        ".+", "--filePattern", help="Filename pattern used to separate data"
    ),
    out_dir: pathlib.Path = typer.Option(
        ..., "--outDir", help="Path to image collection storing copies of renamed files"
    ),
    out_file_pattern: str = typer.Option(
        ".+",
        "--outFilePattern",
        help="Desired filename pattern used to rename and separate data",
    ),
    preview: Optional[bool] = typer.Option(
        False, "--preview", help="Output a JSON preview of files"
    ),
) -> None:
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
    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {file_pattern}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"outFilePattern = {out_file_pattern}")

    inp_dir = inp_dir.resolve()
    out_dir = out_dir.resolve()

    assert (
        inp_dir.exists()
    ), f"{inp_dir} does not exists!! Please check input path again"
    assert (
        out_dir.exists()
    ), f"{out_dir} does not exists!! Please check output path again"
 
    inp_files = [str(inp_file.name) for inp_file in inp_dir.iterdir() if not inp_file.name.startswith(".")]

    assert len(inp_files) != 0, f"Please define {file_pattern} again!! As it is not parsing files correctly"

    chars_to_escape = ["(", ")", "[", "]", "$", "."]
    for char in chars_to_escape:
        file_pattern = file_pattern.replace(char, ("\\" + char))

    if "\.*" in file_pattern:
        file_pattern = file_pattern.replace("\.*", (".*"))
    if "\.+" in file_pattern:
        file_pattern = file_pattern.replace("\.+", (".+"))

    groupname_regex_dict = fr.map_pattern_grps_to_regex(file_pattern)

    # #: Integrate regex from dictionary into original file pattern
    inp_pattern_rgx = fr.convert_to_regex(file_pattern, groupname_regex_dict)
    

    # #: Integrate format strings into outFilePattern to specify digit/char len
    out_pattern_fstring = fr.specify_len(out_file_pattern)
    

    #: List named groups where input pattern=char & output pattern=digit
    char_to_digit_categories = fr.get_char_to_digit_grps(file_pattern, out_file_pattern)
    print(out_pattern_fstring)

    #: List a dictionary (k=named grp, v=match) for each filename
    all_grp_matches = fr.extract_named_grp_matches(inp_pattern_rgx, inp_files)
    

    #: Convert numbers from strings to integers, if applicable
    for i in range(0, len(all_grp_matches)):
        tmp_match = all_grp_matches[i]
        all_grp_matches[i] = fr.str_to_int(tmp_match)

    #: Populate dict if any matches need to be converted from char to digit
    #: Key=named group, Value=Int representing matched chars
    numbered_categories = {}
    for named_grp in char_to_digit_categories:
        numbered_categories[named_grp] = fr.letters_to_int(named_grp, all_grp_matches)
    # Check named groups that need c->d conversion
    for named_grp in char_to_digit_categories:
        for i in range(0, len(all_grp_matches)):
            if all_grp_matches[i].get(named_grp):
                #: Replace original matched letter with new digit
                all_grp_matches[i][named_grp] = numbered_categories[named_grp][
                    all_grp_matches[i][named_grp]
                ]

    output_dict = {}
    for match in all_grp_matches:
        #: If running on WIPP
        if out_dir != "":
            #: Apply str formatting to change digit or char length
            out_name = out_dir.resolve() / out_pattern_fstring.format(**match)
            old_file_name = inp_dir / match["fname"]
            shutil.copy2(old_file_name, out_name)
        #: Enter outDir as an empty string for testing purposes
        elif out_dir == "":
            out_name = out_pattern_fstring.format(**match)
            old_file_name = match["fname"]
        logger.info(f"Old name {old_file_name} & new name {out_name}")
        #: Copy renamed file to output directory
        output_dict[old_file_name] = out_name
    #: Save old and new file names to dict (used for testing)

    if preview:
        with open(pathlib.Path(out_dir, "preview.json"), "w") as jfile:
            out_json: dict[str, Any] = {
                "filepattern": out_file_pattern,
                "outDir": [],
            }
            for file in out_dir.iterdir():
                if file.is_file() and file.suffix != ".json":
                    out_name = file.name
                    out_json["outDir"].append(out_name)
            json.dump(out_json, jfile, indent=2)


if __name__ == "__main__":
    app()
