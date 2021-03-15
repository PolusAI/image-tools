from __future__ import absolute_import, unicode_literals

from .bfio import BioReader, BioWriter
import javabridge,os

_jars_dir = os.path.join(os.path.dirname(__file__), 'jars')

JAR_VERSION = '6.1.0'

JARS = javabridge.JARS + [os.path.realpath(os.path.join(_jars_dir, name + '.jar'))
                          for name in ['loci_tools']]