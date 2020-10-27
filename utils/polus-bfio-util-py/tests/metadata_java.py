import pytest
import javabridge as jutil
import bioformats
import bfio
from pathlib import Path

''' Start the javabridge '''
log_config = Path(__file__).parent.joinpath("log4j.properties")
jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bfio.JARS)

try:
    def version():
        print('VERSION = {}'.format(bfio.__version__))
        assert bfio.__VERSION__ != '0.0.0'
    
finally:
    jutil.kill_vm()