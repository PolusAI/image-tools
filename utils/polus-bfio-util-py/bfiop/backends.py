from bfio import BioReader,BioWriter

class ReaderBackend(metaclass=abc.ABCMeta):
    
    _bioreader = None
    
    name = None
    
    def __init__(self,bioreader):
        self._bioreader = bioreader

class PythonReaderBackend():
    
    name = 'python'
    
class JavaReaderBackend():
    
    name = 'java'
    
BACKEND = PythonBackend

def set_backend(backend):
    
    backend = backend.lower()
    assert backend.lower() in ['python','java']
    
    global BACKEND
    if backend == 'python':
        BACKEND = PythonBackend
    elif backend == 'java':
        BACKEND = JavaBackend
        
    BioReader.set_backend(BACKEND)