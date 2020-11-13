================================
An Introduction to the BioReader
================================

------------
Introduction
------------

The :doc:`/Reference/BioReader` class is designed to make it easy to process
arbitrarily sized images in a fast, scalable way. The ``BioReader`` class can
use one of two different backends depending on the file type that will be read:
``backend='python'`` can only be used to read OME tiled tiff images, while
``backend='java'`` can be used to read
`any format supported by Bioformats <https://docs.openmicroscopy.org/bio-formats/6.1.0/supported-formats.html>`_.

The advantage to using the ``python`` backend is speed and scalability at the
expense of a rigid file structure, while the ``java`` backend provides broad
access to a wide array of file types but is considerably slower.


~~~~~~~~~~~~
Java Backend
~~~~~~~~~~~~

The BioReader ``java`` backend is more robust at loading images, but requires
that Java be installed and additional setup code is required to load images.
This backend is slower and more prone to memory leaks, and is usually used to
load images and convert them into the OME tiled tiff format that can be read by
the Python backend.

.. note::

    This package makes heavy usage of ``python-bioformats``, and the utility of
    ``bfio`` is that the BioReader is multi-threaded and can load larger
    sections of an image than the ``python-bioformats`` is capable of doing
    natively. Because of the way Bioformats indexes images, loading more than
    2GB of an image at a time is impossible without additional code.

~~~~~~~~~~~~~~
Python Backend
~~~~~~~~~~~~~~

The ``python`` backend can only read images that are in OME tiled tif format
with tileheight and tilewidth equal to 1024. This conforms to the
`Web Image Processing Pipeline (WIPP) <https://github.com/usnistgov/wipp>`_
standard file format. The advantage of this format is the ability to quickly 
load sections of large, compressed images without having to load the entire
image. Java is not required, and is 4-15x faster at loading images than the
``java`` backend.