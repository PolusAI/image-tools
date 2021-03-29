from __future__ import absolute_import, unicode_literals

from .classes import FilePattern, VectorPattern

from .functions import get_matching, get_regex,sw_search, \
                       parse_directory, parse_vector, parse_filename, parse_vector_line, \
                       output_name, logger, VARIABLES, \
                       infer_pattern