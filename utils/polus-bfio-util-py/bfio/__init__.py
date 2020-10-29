from __future__ import absolute_import, unicode_literals
import javabridge, pathlib, os.path, logging

from .bfio import BioReader,BioWriter

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("bfio.init")

log_level = logging.WARNING

try:
    with open(pathlib.Path(__file__).parent.joinpath("VERSION"),'r') as fh:
        __version__ = fh.read()
except:
    logger.info('Could not find VERSION. This is likely due to using a local/cloned version of bfio.')
    __version__ = "0.0.0"
logger.info('VERSION = {}'.format(__version__))

if pathlib.Path(__file__).parent.joinpath('jars').exists():
    _jars_dir = os.path.join(os.path.dirname(__file__), 'jars')
    
    JAR_VERSION = '6.1.0'

    JARS = javabridge.JARS + [os.path.realpath(os.path.join(_jars_dir, name + '.jar'))
                              for name in ['loci_tools']]
    
    LOG4J = os.path.realpath(os.path.join(_jars_dir, 'log4j.properties'))
    
    logger.info('loci_tools.jar version = {}'.format(JAR_VERSION))
    
else:
    JAR_VERSION = None
    JARS = None
    logger.info('The loci_tools.jar is not present. Can only use Python backend for reading/writing images.')