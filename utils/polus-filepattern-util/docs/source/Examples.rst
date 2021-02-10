========
Examples
========

.. contents:: Table of Contents
   :local:
   :depth: 3

------------------------
What is ``filepattern``?
------------------------

To state it briefly, a ``filepattern`` is simplified notation for some types of 
`regular expressions <https://en.wikipedia.org/wiki/Regular_expression>`_.
Frequently, data from a new source (person or machine) has a unique naming
convention that is systematic, rational, and requires work/code to parse. One
tool used to help wrangle file naming conventions are regular expression, and 
working with regular expressions can be laborious, time consuming, and difficult
to explain to those unfamiliar with programming. To help abstract the section of
code that might be dedicated to file name parsing, filepattern was born. It has
a number of limitations, but frequently works well with machine generated files.

What does a ``filepattern`` look like? It is probably easiest to show by
example. Say there is a folder with the following files:

.. code-block:: bash
    
    my_data_folder/x000_y000_z001.tif
    my_data_folder/x000_y000_z002.tif
    my_data_folder/x000_y000_z003.tif

The ``filepattern`` for the above files would be ``x000_y000_z00{z}.ome.tif``.
The curly brackets indicate a file name variable, and ``{z}`` indicates that the
number will be parsed and stored as a z value. If a similar regular expression
were to be written, then it would look like ``x000_y000_z00([0-9]).ome.tif``,
which is not only longer but would require more code to parse the regular
expression.

Thus, a ``filepattern`` is just a string with sections of a file name replaced
with ``{}`` to capture and store information in the file name as a variable.
What this allows you to do is write code once that will work on new data sets
with new file naming conventions seemlessly. For example, if you want to write
a program that stacks 2d images to create a 3d image, then you can just specify
a filepattern with the ``{z}`` variable defined for each data set. The
examples below show how this works in more detail.

-----
Setup
-----

~~~~~~~~~~~
Intallation
~~~~~~~~~~~

A nice feature of ``filepattern`` is that it is pure Python. No external
dependencies required. Installation can be done by:

.. code-block:: bash

    pip install filepattern

~~~~~~~~~
Demo Data
~~~~~~~~~

Throughout the example documentation, some demo images are going to be used that
are publicly available from NIST. downloading these and following along might
help in understanding how the different parameters work, but it isn't necessary.

.. code-block:: python

    from pathlib import Path
    import requests, zipfile

    """ Get an example image """
    # Set up the directories
    PATH = Path(__file__).with_name('data')
    PATH.mkdir(parents=True, exist_ok=True)

    # Download the data if it doesn't exist
    URL = "https://github.com/USNISTGOV/MIST/wiki/testdata/"
    FILENAME = "Small_Fluorescent_Test_Dataset.zip"
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        
    with zipfile.ZipFile(PATH/FILENAME, 'r') as zip_ref:
        zip_ref.extractall(PATH)

-----------
Basic Usage
-----------

.. note::

    This example was created using ``filepattern==1.4.0``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Creating Your First ``filepattern``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing to do is import and set up the file path, and maybe take a look
at what the filenames look like.

.. code-block:: python

    import filepattern, pathlib

    filepath = pathlib.Path(__file__).parent
    filepath = filepath.joinpath('data/Small_Fluorescent_Test_Dataset/image-tiles/')

    for file in filepath.iterdir():
        print(file.name)

This should look something like this:

.. code-block:: bash

    img_r001_c004.tif
    img_r005_c005.tif
    img_r002_c001.tif
    img_r002_c005.tif
    ...

It looks like the files have row and column indexing
(img\_\ **r**\ 001\_\ **c**\ 001.tif). Thus, the following ``filepattern``
definitions would probably work:

.. code-block:: bash

    img_r00{r}_c00{c}.tif
    img_r00{y}_c00{x}.tif

.. note::

    Notice that in the second ``filepattern`` that the ``y`` variable was put
    in place of the ``r`` variable. This is because rows in matrix notation
    is roughly equivalent to the y-axis in plotting/graphing notation.

The approach of looking at the files and coming up with our own ``filepattern``
generally works for small data sets. However, if you have a large data set with
hundreds or thousands of files and want to get a first guess on a
``filepattern`` that would account for all file names, you can use the built-in
``infer_pattern`` function.

.. code-block:: python

    import filepattern, pathlib

    filepath = pathlib.Path(__file__).parent
    filepath = filepath.joinpath('data/Small_Fluorescent_Test_Dataset/image-tiles/')
    files = list(filepath.iterdir())

    pattern_guess = filepattern.infer_pattern(files)

    print('Inferred Pattern: {}'.format(pattern_guess))

.. code-block:: bash

    Inferred Pattern: img_r00{r}_c00{t}.tif

The ``infer_pattern`` function looks at all of the files you supply to it, and
then comes up with a good guess at what a filepattern should look like, but it
likely won't get the variable names correct (if the variable names matter).

The above notation with ``{r}`` indicates that the ``r`` variable can only ever
take on value of 0-9 because it is only permitted one space within the filename.
If you wanted to capture two spaces for the variables, you would do ``{rr}`` so
that the above pattern would become ``img_r0{rr}_c00{c}.tif``, or 
``img_r0{rr}_c0{cc}.tif``. Alternatively, a variable length variable definition
could be defined using the ``+`` notation, which would make the pattern
``img_r{r+}_c{c+}.tif``. This will capture all numbers following ``img_r`` but
occur before ``_c``. This is useful in cases where the file names are not all
fixed with, so that you can parse files like ``img1.tif`` and ``img10.tif``
using the same ``filepattern``.

This fixed width variable definition was created for instances where filenaming
conventions are completely numeric, as in the following case:

.. code-block:: bash

    001004.tif
    005005.tif
    002001.tif
    002005.tif
    ...

In this case, a filepattern like ``{r+}.tif`` would capture the entire sequence
of numbers. Instead, we may want to store the first three numbers in one
variable and the second three numbers in a second variables: ``{yyy}{xxx}.tif``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the FilePattern Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have a ``filepattern``, we will create a ``FilePattern`` object to
help us handle the data that gets parsed from the ``filepattern``. A
``FilePattern`` object is also an iterable that allows us to cleverly loop over
the data

.. code-block:: python

    import filepattern, pathlib, pprint

    filepath = pathlib.Path(__file__).parent
    filepath = filepath.joinpath('data/Small_Fluorescent_Test_Dataset/image-tiles/')

    pattern = 'img_r00{y}_c00{x}.tif'

    fp = filepattern.FilePattern(filepath,pattern)

    for file in fp():

        pprint.pprint(file)

.. code-block:: python

    [{'file': PosixPath('.../Small_Fluorescent_Test_Dataset/image-tiles/img_r001_c001.tif'),
    'x': 1,
    'y': 1}]
    [{'file': PosixPath('.../Small_Fluorescent_Test_Dataset/image-tiles/img_r001_c002.tif'),
    'x': 2,
    'y': 1}]
    [{'file': PosixPath('.../Small_Fluorescent_Test_Dataset/image-tiles/img_r001_c003.tif'),
    'x': 3,
    'y': 1}]
    ...

The above output only shows the first three ``file``s returned by ``fp()`` for
demonstration. Every iteration returns a list of dictionaries, where each
dictionary contains a ``file`` key whos value is an ``imglib.Path`` object to a
file location, and then the remaining key/value pairs are the variables in the
``pattern`` and the extracted value.

The reason why a list is returned is because all dictionaries in the list will
have identical values, or a range of values supplied when constructing the
iterator. As an example, let's say on each iteration we want to return all files
associated with the same column. We could modify the iterator constructor to
group the list of files returned by column as follows:

.. code-block:: python

    for file in fp(group_by='y'):

        pprint.pprint(file)

.. code-block:: bash

    [{'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r001_c001.tif'),
    'x': 1,
    'y': 1},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r002_c001.tif'),
    'x': 1,
    'y': 2},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r003_c001.tif'),
    'x': 1,
    'y': 3},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r004_c001.tif'),
    'x': 1,
    'y': 4},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r005_c001.tif'),
    'x': 1,
    'y': 5}]

The above output block is from the first iteration, and it is a list of file
dictionaries with identical ``x`` but different ``y`` values (meaning all of the
files are from the same column or have the same x value.) The ``group_by``
groups data together where all ``filepattern`` variables are identical except
for the values included in ``group_by``.

If you only desire to get data where a variable matches a value or list of
values without looping over all of the data, it you can retrieve them using the
``get_matching`` method.

.. code-block:: python

    pprint.pprint(fp.get_matching(X=[1]))

.. code-block:: bash

    [{'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r001_c001.tif'),
    'x': 1,
    'y': 1},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r002_c001.tif'),
    'x': 1,
    'y': 2},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r003_c001.tif'),
    'x': 1,
    'y': 3},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r004_c001.tif'),
    'x': 1,
    'y': 4},
    {'file': PosixPath('../Small_Fluorescent_Test_Dataset/image-tiles/img_r005_c001.tif'),
    'x': 1,
    'y': 5}]

-----------
Limitations
-----------

``filepattern`` only addresses numeric variables in a name. Some image naming
conventions will frequently use a channel descriptor, such as

.. code-block:: bash

    img_x01_y01_DAPI.tif
    img_x01_y01_TXRED.tif
    img_x01_y01_GFP.tif

In the above example, the x and y values can be extracted, but the channel names
cannot. So it is possible to loop over all DAPI images in x and y, but in order
to associate all channels with each other would require three separate
filepatterns.

Another limitation is that only the characters `rtczyxp` are permitted as
variable names at the moment.