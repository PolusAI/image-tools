=========================================================
Filepattern: A Utility for Programmatic File Manipulation
=========================================================

The `filepattern` Python utility is designed to extract information stored in
file names. A `filepattern` is essentially a simplified regular expression with
named groups, and regular expressions are valid `filepattern` expressions
provided they do not use groups.

The utility was born from the need to manipulate and organize image data from a
variety of microscopes, all of which have a systematic but different filenaming
conventions. This made abstracting things like image stitching algorithms easier
to apply to files with disparate naming conventions by simply changing the
`filepattern` rather than generating new code to parse each new naming
convention. Although `filepattern` was born to wield against image data, it is
not limited to image data, and can handle filenames with any extension.

The `filepattern` utility was inspired by the naming convention found in the
[MIST](https://github.com/usnistgov/MIST)
stitching algorithm.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Examples
   Reference