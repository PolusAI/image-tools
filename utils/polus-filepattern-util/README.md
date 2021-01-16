# Filepattern Utility

Functions and a class to easily filename patterns for [WIPP](https://github.com/usnistgov/WIPP).

## Documentation

[Read the docs!](https://filepattern.readthedocs.io/en/latest/)

## Install

`pip install filepattern`

## FilePattern Class

A number of functions are included in `filepattern.py`, but some contain
complex output values that may be difficult to handle in an abstract way. To
simplify things, the `FilePattern` class was created. The usage of the
`FilePattern` class will be described here, but if finer control over filename
parsing is needed, detailed descriptions of each function is provided in
`filepatter.py`.

The two methods implemented by `FilePattern` are `get_matching` and `iterate`.
The `get_matching` function gets all filenames matching a specific set of
filename values, while the `iterate` function is an iterable that iterates over
every combination of filename values, which options for returning groups of
filenames according to a particular variable.

### File Pattern Format

A file pattern is a string that follows the formatting of the
[MIST plugin](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters).
It is similar to a
[regular expression](https://en.wikipedia.org/wiki/Regular_expression),
and regular expression values may be included in the file pattern. However, the
file pattern string includes variables fields that are surround by curly
brackets, `{}`, and the width of the parsed variables is indicated. For example,
if there is a set of files with names:

```bash
image_c000_z000.ome.tif
image_c000_z001.ome.tif
image_c000_z002.ome.tif
image_c001_z000.ome.tif
image_c001_z001.ome.tif
image_c001_z002.ome.tif
```

Then the filename pattern that indicates a c-variable and a z-variable would be
`image_c{ccc}_z{zzz}.ome.tif`. Note that the width of each variable is indicated
by repeating the variable (for the c-variable, a width of 3 is indicated by
`ccc`).

**NOTE**: The only possible variables are this time are `x`, `y`, `p`, `z`, `c`,
`t`, and `r`. Further, only `x` and/or `y` may be defined or `p` may be defined,
but if `p` is defined when either `x` or `y` is defined, then an error will be
thrown. This will likely change in the future as more complex data sets will
need to be processed.

### FilePattern Initialization

The `FilePattern` class is initiated using `file_path` (a folder path),
`pattern` (a file pattern as described above), and an optional `var_order` that
describes how files are sorted internally. In general, `var_order` shouldn't
need to be set since the object methods can handle most file organization
issues. If needed, `var_order` must be input as a string of variables that will
be contained in the internal file organizational strcuture. An example
`var_order` would be `xyzctr`.

### FilePattern.get_matching

This function retrieves all files that match specific variables values. Using
the example filenames presented in the **File Pattern Format** section, if `C=0`
is passed as an input argument, then `get_matching` will return all files that
contain `_c000`. However, the input values do not need to be a single value,
they can be a list. So if `C=[0,1]`, then a list of all files will be returned
such that each file would contain `_c000` or `_c001`.

The list that is returned contains dictionaries. Each dictionary contains a key
for each variable parsed from the filename and a `file` key that indicates the
name of the file.

### FilePattern.iterate

This function is an iterable that returns a list of filenames every time it is
called. Each call returns a list of files that match a unique combination of
variable values so that every image that matches a file pattern is contained in
only one of the calls to this function. Specifying a list of variables in the
`group_by` parameter will return a list of filenames with all variable values
constant except those indicated by `group_by`. Using the example filenames
presented in the **File Pattern Format** section, if `group_by='z'` then the top
three files are returned by the first call to this function and the bottom three
files are returned by the second call to this function.

In addition to the `group_by` argument, it is possible to pass arguments
matching the `get_matching` function. This will cause `iterate` to only return
files matching specific variables values.

## Examples

### Simple iterator for all tiled tiff images in an input directory

Although it is probably overkill, a simple way to iterate over all images with a
tiled tiff extension is:

```python
file_path = "/path/to/files"
extension = '.ome.tif'
pattern = ".*" + extension

files = FilePattern(file_path,pattern)

for f_list in files.iterate():
    print(f_list['file'])
```

### Stack z-slices

In some cases, a microscope will export each image in a z-stack as a separate
image file. If a microscope takes an image at three z-positions in each well,
and images 4 wells, assume the output files are:

```bash
image_x000_y000_z000.ome.tif
image_x000_y000_z001.ome.tif
image_x000_y000_z002.ome.tif
image_x000_y001_z000.ome.tif
image_x000_y001_z001.ome.tif
image_x000_y001_z002.ome.tif
image_x001_y000_z000.ome.tif
image_x001_y000_z001.ome.tif
image_x001_y000_z002.ome.tif
image_x001_y001_z000.ome.tif
image_x001_y001_z001.ome.tif
image_x001_y001_z002.ome.tif
```

To start, initialize the `FilePattern` object:
```python
file_path = "/path/to/files"
pattern = "image_x{xxx}_y{yyy}_z{zzz}.ome.tif"

fp = FilePattern(file_path,pattern)
```

To get all of the z-slices for position x=1,y=0:
```python
z_slices = fp.get_matching(X=1,Y=0)
print([f['file'] for f in z_slices]) # print the path for each file returned
```
Output:
```bash
image_x001_y000_z000.ome.tif
image_x001_y000_z001.ome.tif
image_x001_y000_z002.ome.tif
```

To loop through each position doing the same thing:
```python
for f in fp.iterate(group_by='z'):
    print('Files for (x,y): ({},{})'.format(f[0]['x'],f[0]['y']))
    print([f['file'] for f in z_slices]) # print the path for each file returned
```