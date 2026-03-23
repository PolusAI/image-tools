"""Build registration groups from a filepattern-indexed image directory."""
from __future__ import annotations

import itertools

import numpy as np
from filepattern import FilePattern
from filepattern import get_regex
from filepattern import parse_directory


def parse_collection(
    directory_path: str,
    file_pattern: str,
    registration_variable: str,
    similarity_variable: str,
    _template_image: str,
) -> dict[tuple[str, str], np.ndarray]:
    r"""Parse the input directory into a registration / similar-transform map.

    Each key is ``(template_path, moving_path)``; the value lists paths that share
    the moving image's transform.

    Note:
        Tested with ``len(registration_variable) == len(similarity_variable) == 1``.
        Other lengths are not validated.

    Args:
        directory_path: Path to the input collection.
        file_pattern: Filename pattern of the input images.
        registration_variable: Variable that groups template vs moving images.
        similarity_variable: Variable for images with the same transform as moving.
        _template_image: Reserved for future use (template basename).

    Returns:
        ``result_dic`` mapping ``(template, moving)`` to an array of related paths.
    """
    # get all variables in the file pattern
    _, variables = get_regex(file_pattern)

    # get variables except the registration and similarity variable
    moving_variables = [
        var
        for var in variables
        if var not in registration_variable and var not in similarity_variable
    ]

    # uvals is dictionary with all the possible variables as key
    # corresponding to each key is a list of all values which that variable can take
    _, uvals = parse_directory(directory_path, file_pattern)

    parser_object = FilePattern(directory_path, file_pattern)

    image_set = []

    # extract the index values from uvals for each variable in moving_variables
    moving_variables_set = [uvals[var] for var in moving_variables]

    # iterate over the similar transformation variables;
    # expected output when len(registration_variable)==len(similarity_variable)==1
    for char in similarity_variable:
        # append the variable to the moving variable set
        moving_variables.append(char)

        # iterate over all possible index values of the similar transf. variable
        for ind in uvals[char]:
            registration_set = []

            # append the fixed value of the index to the moving variables set
            moving_variables_set.append([ind])

            # all combinations of index values in the moving variables set
            reg_idx_combos = list(itertools.product(*moving_variables_set))
            all_dicts = []

            # dicts like {'C': 1, 'X': 2, ...} for get_matching()
            for index_comb in reg_idx_combos:
                inter_dict = {}
                for i in range(len(moving_variables)):
                    inter_dict.update({moving_variables[i].upper(): index_comb[i]})
                # store all dictionaries
                all_dicts.append(inter_dict)

            # iterate over all dictionaries
            for reg_dict in all_dicts:
                intermediate_set = []
                files = parser_object.get_matching(**reg_dict)

                # files is a list of dictionaries
                for file_dict in files:
                    intermediate_set.append(file_dict["file"])
                registration_set.append(intermediate_set)

            # drop fixed similar-transform index before next iteration
            moving_variables_set.pop(-1)
            image_set.append(registration_set)

    # parse image set to form the result dictionary
    result_dic = {}
    old_set = np.array(image_set)
    for j in range(old_set.shape[1]):
        inter = old_set[:, j, :]
        for k in range(inter.shape[1]):
            ky = (inter[0, 0], inter[0, k])
            items = inter[1:, k]
            result_dic.update({ky: items})

    return result_dic
