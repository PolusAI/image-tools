import regex as re
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import shutil


def add_curly_brackets(pattern):
    #: This function changes (text) to ({text})
    pattern_mod = ""
    for i in pattern:
        if i == "(":
            pattern_mod = pattern_mod + i + "{"
        elif i == ")":
            pattern_mod = pattern_mod + "}" + i
        else:
            pattern_mod = pattern_mod + i
    return pattern_mod

def make_digit_channel_dict(input_images):
    """#: Create dictionary {0: 'DAPI', 1: 'GFP', 2: 'TXRED'}
    Args:
        input_images:  [PosixPath('image_collection_1/img_x01_y01_DAPI.tif'), PosixPath('image_collection_1/img_x01_y01_TXRED.tif'), PosixPath('image_collection_1/img_x01_y01_GFP.tif')]
    Returns:
        gfp_dict: {0: 'DAPI', 1: 'GFP', 2: 'TXRED'}
        """
    # Convert input string to list, ['img', '01', '01', 'DAPI.tif']. Convert to ['img', '001', '001', 'DAPI.tif']
    new_images = []
    list2 = []
    #: Make list of lists and convert to df
    for image in input_images:
        new_images.append(image.name)
        list1 = ["_".join(x.split()) for x in re.split(r'[_]', image.name) if x.strip()] #: ['img', '01', '01', 'DAPI.tif']
        list2.append(list1)
    df = pd.DataFrame(list2)
    #: Sort last column alphabetically and reset index
    df.sort_values(by=df.columns[-1], ascending=True, inplace=True)
    df.reset_index(inplace=True)
    chan_series = df[df.columns[-1]]
    chan_series  = chan_series.str.rstrip(to_strip=".tif")
    #: Make dict where key is a number, value is channel. Values are sorted.
    gfp_dict = chan_series.to_dict()
    return gfp_dict

def pad_it(output_format, data, input_format):
    """
    output_format is regex like d+, dd, ff
    data is a number like 100
    input_format = inputs dataformats
        
    Args:
        output_format: {}
        data: {}
        input_format: {} ddd 01 dd
        data2:  001
        
        output_format: {}
        data: {}
        input_format: {} nan _y nan
        data2:  _y
        
        output_format: {}
        data: {}
        input_format: {} ddd 01 dd
        data2:  001
        
        output_format: {}
        data: {}
        input_format: {} nan _ nan
        data2:  _
        
        output_format: {}
        data: {}
        input_format: {} ddddd GFP .*
        data2:  starttempnum_GFP_endtempnum_regexstart_ddddd_regexend_
        
        output_format: {}
        data: {}
        input_format: {} nan .tif nan
        data2:  .tif
        
    Returns:
        data2: See examples above
    """
    data2 = None
    digits_list = ["d", "dd", "ddd", "dddd", "ddddd"]
    try:
        #: Filters out 'nan' and non-numeric string values
        data = int(data)
        output_format = str(output_format)
        pad_digit = "{:0" + str(len(output_format)) + str(output_format[0]) + "}"
        tmp_string = "{}".format(pad_digit)
        data2 = tmp_string.format(data)
    except ValueError:
        #: Handle cases input is character and output is digit
        data2 = ""
        if input_format == ".*" and output_format in digits_list:
            data2 = "starttempnum_" + data + "_endtempnum" + "_regexstart_" + output_format + "_regexend_"  
        else:
            data2 = data 
    return data2

def build_var_digits_dict(pattern_as_list):
    """
    Build dict: key=match group id, value=regex pattern
    
    Args:
        pattern_as_list:  ['newdata_x', '{var_x:ddd}', '_y', '{var_y:ddd}', '_c', '{var_name:ddddd}', '.tif']
    Returns:
        var_digits_dict {'var_x': 'ddd', 'var_y': 'ddd', 'var_name': 'ddddd'}
    """
    #: Build dictionary {'var_x': 'dd', 'var_y': 'dd', 'var_name': 'c+'}
    var_digits_dict = {}
    for i in range (0, len(pattern_as_list)):
        if pattern_as_list[i].startswith("{"):
            #: Split string into list by : and remove curly brackets
            var_digit_list = [", ".join(x.split()) for x in re.split(r'[:]', pattern_as_list[i]) if x.strip()]
            var_digits_dict[var_digit_list[0][1:]] = var_digit_list[1][:-1]
    return var_digits_dict

def build_psuedoregex_trueregex_dict(input_pattern_as_list, var_digits_dict):
    """
    Build pseudoregex true regex dict
    
    Args:
        input_pattern_as_list: ['img_x', '{var_x:dd}', '_y', '{var_y:dd}', '_', '{var_name:.*}', '.tif']
        var_digits_dict: {'var_x': 'ddd', 'var_y': 'ddd', 'var_name': 'ddddd'}
        
    Returns:
        pseudoregex_trueregex_dict: {'newdata_x': 'newdata_x', 'var_x:ddd': '(\\d\\d\\d)', '_y': '_y', 'var_y:ddd': '(\\d\\d\\d)', '_c': '_c', 'var_name:ddddd': '(\\d\\d\\d\\d\\d)', '.tif': '.tif'}
    """
    new_pattern = ""
    pseudoregex_trueregex_dict = {}
    for i in range (0, len(input_pattern_as_list)):
        if input_pattern_as_list[i].startswith("{"):
            input_pattern_as_list[i] = input_pattern_as_list[i][1:-1]
            for each_key, each_value in var_digits_dict.items():
                new_val = ""
                if input_pattern_as_list[i].startswith(each_key):
                    backslash = "\\"
                    for char in each_value:
                        if char != "." and char != "*":
                            new_val = new_val + backslash + char
                        elif char == "." or char == '*':
                            new_val = new_val + char
                    new_val = "(" + new_val + ")"
                    new_pattern = new_pattern + new_val
                    pseudoregex_trueregex_dict[input_pattern_as_list[i]] = new_val
        elif input_pattern_as_list[i].startswith("{") == False: 
            new_patt = "(" + input_pattern_as_list[i] + ")"
            pseudoregex_trueregex_dict[input_pattern_as_list[i]] = input_pattern_as_list[i]
            new_pattern = new_pattern + new_patt

    return pseudoregex_trueregex_dict

def get_values(df_col_as_list):
    """
    Args:
        df_col_as_list: ['img_x', 'var_x:dd', '_y', 'var_y:dd', '_', 'var_name:.*', '.tif']
        
    Returns:
        values_list: [nan, 'dd', nan, 'dd', nan, '.*', nan]
    """
    values_list = []
    for item in df_col_as_list:
        if ":" not in item:
            values_list.append(np.nan)
        else:
            x = item.split(':')[-1]
            values_list.append(x) 
    return values_list

def build_dataframe(pseudoregex_trueregex_in_dict, pseudoregex_trueregex_out_dict, input_image, match_groups):
    """
    Create intermediate filename to look like img_x001_y001_starttempnum_TXRED_endtempnum_regexstart_ddddd_regexend_.tif
    
    Args:
        pseudoregex_trueregex_in_dict: {'img_x': 'img_x', 'var_x:dd': '(\\d\\d)', '_y': '_y', 'var_y:dd': '(\\d\\d)', '_': '_', 'var_name:.*': '(.*)', '.tif': '.tif'}
        pseudoregex_trueregex_out_dict: {'newdata_x': 'newdata_x', 'var_x:ddd': '(\\d\\d\\d)', '_y': '_y', 'var_y:ddd': '(\\d\\d\\d)', '_c': '_c', 'var_name:ddddd': '(\\d\\d\\d\\d\\d)', '.tif': '.tif'}
        input_image: img_x01_y01_GFP.tif
        match_groups: ['img_x', '01', '_y', '01', '_', 'GFP', '.tif']
    
    Returns:
        final_str: img_x001_y001_starttempnum_TXRED_endtempnum_regexstart_ddddd_regexend_.tif
    """
    df1 = pd.Series(pseudoregex_trueregex_in_dict).to_frame('psy_in_val').reset_index()
    df1.rename(columns={"index": "psy_in_key"}, inplace=True)
    df2 = pd.Series(pseudoregex_trueregex_out_dict).to_frame('psy_out_val').reset_index()
    df2.rename(columns={"index": "psy_out_key"}, inplace=True)
    df = pd.concat([df1, df2], axis=1)
    #: Build new column from values in psy_out_key
    #: Source https://stackoverflow.com/questions/63840249/pandas-in-df-column-extract-string-after-colon-if-colon-exits-if-not-keep-text
    df['Produces'] = df['psy_out_key'].str.split(':').str[-1]
    df_out_col_list = df['psy_out_key'].tolist()
    df_in_col_list = df['psy_in_key'].tolist()
    out_list2 = get_values(df_out_col_list)
    in_list2 = get_values(df_in_col_list)
    df['Produces2'] = out_list2
    df["Produces1"] = in_list2
    df["original_string"] = input_image
    match_groups_np = np.array(match_groups)
    df["match groups"] = match_groups_np
    #: Use Produces column and match groups column to produce i
    match_groups = df["match groups"].tolist()
    produces = df["Produces2"].tolist()
    produces1 = df["Produces1"].tolist() 
    new_list = []
    for i in range(0, len(produces)):
        if produces[i] == None:
            new_list.append(match_groups[i])
        else:
            new_val = pad_it(produces[i], match_groups[i], produces1[i])
            new_list.append(new_val)
    df['newlist'] = new_list
    df["final"] = "".join(df["newlist"].tolist())
    final_str = "".join(df["newlist"].tolist())
    return final_str

def temp_to_final_filename(final_filename, digit_channame_dict):
    """
    Args:
        final_filename: img_x001_y001_starttempnum_GFP_endtempnum_regexstart_ddddd_regexend_.tif
        digit_channame_dict: {0: 'DAPI', 1: 'GFP', 2: 'TXRED'}
    Returns:
        new_filename: img_x001_y001_00001.tif
    """
    temp_dict = []
    result = re.search('starttempnum_(.*)_endtempnum_regexstart_(.*)_regexend_', final_filename)
    x = result.group(1)
    temp_dict.append(x)
    #: Use temp_dict_sorted to sort channels alphabetically
    channel_dict = {}
    re_string = 'starttempnum_(.*)_endtempnum_regexstart_(.*)_regexend_'
    result = re.search(re_string, final_filename)
    x = result.group(1)
    li = None
    for each_key, each_value in digit_channame_dict.items():
        if x == each_value:
            li = each_key
    out_format = result.group(2)
    pad_digit = "{:0" + str(len(out_format)) + str(out_format[0]) + "}"
    tmp_string = "{}".format(pad_digit)
    data2 = tmp_string.format(li)
    channel_dict[data2] = x
    new_filename = re.sub(re_string, data2, final_filename)
    return new_filename
    
def convert_filename(input_image, output_directory_name, input_pattern, output_pattern, input_images):
    """
    Args:
        input_image: img_x01_y01_GFP.tif
        output_directory_name: output_image_collection_1
        input_pattern: img_x(var_x:dd)_y(var_y:dd)_(var_name:.*).tif
        output_pattern: newdata_x(var_x:ddd)_y(var_y:ddd)_c(var_name:ddddd).tif
        input_images: [PosixPath('image_collection_1/img_x01_y01_DAPI.tif'), PosixPath('image_collection_1/img_x01_y01_TXRED.tif'), PosixPath('image_collection_1/img_x01_y01_GFP.tif')]
    Returns:
        renamed_filepath: output_image_collection_1/img_x001_y001_00001.tif
    
    """
    #: Initialize variables
    input_pattern = add_curly_brackets(input_pattern)
    output_pattern = add_curly_brackets(output_pattern)
    #: ['img_x', '{var_x:dd}', '_y', '{var_y:dd}', '_', '{var_name:c+}', '.tif']
    input_pattern_as_list = [
        ", ".join(x.split()) for x in re.split(r'[()]', input_pattern) if x.strip()] 
    #: ['newdata_x', '{var_x:ddd}', '_y', '{var_y:ddd}', '_c', '{var_name:ddd}', '.tif'] 
    output_pattern_as_list = [
        ", ".join(x.split()) for x in re.split(r'[()]', output_pattern) if x.strip()]    
    #: Build dict: key=match group id, value=regex pattern
    var_digits_dict_input = build_var_digits_dict(input_pattern_as_list) #: {'var_x': 'dd', 'var_y': 'dd', 'var_name': 'c+'}
    var_digits_dict_output = build_var_digits_dict(output_pattern_as_list) #: {'var_x': 'ddd', 'var_y': 'ddd', 'var_name': 'dddd'}
    #: Create dictionary {0: 'DAPI', 1: 'GFP', 2: 'TXRED'}
    digit_channame_dict = make_digit_channel_dict(input_images) 
    #: Build pseudoregex true regex dict
    #: Ex {'img_x': 'img_x', 'var_x:dd': '(\\d\\d)', '_y': '_y', 'var_y:dd': '(\\d\\d)', '_': '_', 'var_name:.*': '(.*)', '.tif': '.tif'}
    pseudoregex_trueregex_in_dict = build_psuedoregex_trueregex_dict(
        input_pattern_as_list, var_digits_dict_input) 
    pseudoregex_trueregex_out_dict = build_psuedoregex_trueregex_dict(
        output_pattern_as_list, var_digits_dict_output)
    #: Build match groups column 
    psy_in_vals = list(pseudoregex_trueregex_in_dict.values())
    #: Add parenthesis to each match group in list if doesnt contain parenthesis already
    #: Ex: ['(img_x)', '(\\d\\d)', '(_y)', '(\\d\\d)', '(_)', '(.*)', '(.tif)']
    match_grouping_list = [
        "(" + item + ")" if item.startswith("(") == False else item for item in psy_in_vals
        ] 
    #: regex_groups = (img_x)(\d\d)(_y)(\d\d)(_)(.*)(.tif)
    regex_groups = "".join(match_grouping_list)
    logger.debug("Regex Groups: {}".format(regex_groups))
    logger.debug("Image Name: {}".format(input_image))
    #: Search filename for regex match groups
    matches = re.search(regex_groups, input_image)
    logger.debug("Matches: {}".format(matches))
    final_filename = None  
    #: Create intermediate filename to look like img_x001_y001_starttempnum_TXRED_endtempnum_regexstart_ddddd_regexend_.tif
    try:
        match_groups = list(matches.groups())
        final_filename = build_dataframe(
            pseudoregex_trueregex_in_dict, 
            pseudoregex_trueregex_out_dict, 
            input_image, match_groups
            )
        new_filename = temp_to_final_filename(
            final_filename, digit_channame_dict
            )
        renamed_filepath = Path(Path(output_directory_name) / Path(new_filename))
        return renamed_filepath
    except AttributeError as e:
        logger.error("Ensure that your file pattern matches file names")
        logger.error(e)

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
        description="Renames files from a given image collection using file renaming pattern"
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
    #: Check for subfolders named images and switch to that subfolder
    inpDir = args.inpDir
    logger.debug("Old input directory: {}".format(inpDir))
    inpDir = Path(inpDir)
    #: outDir is the new csv collection
    outDir = args.outDir
    logger.debug("outDir = {}".format(outDir))
    #: Input pattern is the regex pattern input by user
    filePattern = args.filePattern
    logger.debug("filePattern = {}".format(filePattern))
    #: Output pattern is regex pattern expected by user
    outFilePattern = args.outFilePattern
    logger.debug("output_pattern = {}".format(outFilePattern))
    #: Check that given input directory points to images folder
    logger.info("Navigated to the image collection...")
    #: Define the path:
    logger.debug("Defining paths...")
    input_directory = inpDir
    input_images = [p for p in input_directory.iterdir() if p.is_file()]
    for input_file in input_directory.iterdir():
        logger.info("Parsing {}".format(input_file))
        output_file = convert_filename(
            str(input_file.name), Path(outDir), 
            filePattern, outFilePattern, input_images
            )
        logger.info("Output file {}".format(output_file))
        logger.info("Copying output file ", output_file, "...")
        shutil.copy2(input_file, Path(output_file))
