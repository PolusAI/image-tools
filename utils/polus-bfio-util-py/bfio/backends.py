from tifffile import tifffile
from pathlib import Path
import typing,logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import bfio
from bfio.OmeXml import OMEXML
import bfio.base_classes

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("bfio.backends")

class PythonReader(bfio.base_classes.AbstractReader):
    
    logger = logging.getLogger("bfio.backends.PythonReader")

    def __init__(self, frontend):
        super().__init__(frontend)

        self.logger.debug('__init__(): Initializing _rdr (tifffile.TiffFile)...')
        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        
    def read_metadata(self):
        self.logger.debug('read_metadata(): Reading metadata...')
        return OMEXML(self._rdr.ome_metadata)
    
    def _chunk_indices(self,X,Y,Z):
        
        self.logger.debug('_chunk_indices(): (X,Y,Z) -> ({},{},{})'.format(X,Y,Z))
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2
        
        offsets = []
        bytecounts = []
        
        ts = self.frontend._TILE_SIZE
        
        x_tiles = np.arange(X[0]//ts,np.ceil(X[1]/ts),dtype=int)
        y_tile_stride = np.ceil(self.frontend.x/ts).astype(int)
        
        self.logger.debug('_chunk_indices(): x_tiles = {}'.format(x_tiles))
        self.logger.debug('_chunk_indices(): y_tile_stride = {}'.format(y_tile_stride))
        
        for z in range(Z[0],Z[1]):
            for y in range(Y[0]//ts,int(np.ceil(Y[1]/ts))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()
                
                o = [self._rdr.pages[z].dataoffsets[i] for i in ind]
                b = [self._rdr.pages[z].databytecounts[i] for i in ind]
                
                self.logger.debug('_chunk_indices(): offsets = {}'.format(o))
                self.logger.debug('_chunk_indices(): bytecounts = {}'.format(b))
                
                offsets.extend(o)
                bytecounts.extend(b)
        
        return offsets,bytecounts
    
    def _read_tile(self, args):
        
        keyframe = self._keyframe
        out = self._out
        
        w,l,d = self._tile_indices[args[1]]
        
        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)
        
        if segment is None:
            segment = keyframe.nodata
            
        self.logger.debug('_read_tile(): shape = {}'.format(shape))
        self.logger.debug('_read_tile(): (w,l,d) = {},{},{}'.format(w,l,d))
        
        out[l: l + shape[1],
            w: w + shape[2],
            d,0,0] = segment.squeeze()
    
    def read_image(self,X,Y,Z,C,T,output):
        if (len(C)>1 and C[0]!=0) or (len(T)>0 and T[0]!=0):
            raise Warning('More than channel 0 was specified for either channel or timepoint data.' + \
                          'For the Python backend, only the first channel/timepoint will be loaded.')
        
        # Define tile bounds
        ts = self.frontend._TILE_SIZE
        X_tile_shape = X[1] - X[0]
        Y_tile_shape = Y[1] - Y[0]
        Z_tile_shape = Z[1] - Z[0]
        
        # Get keyframe and filehandle objects
        self._keyframe = self._rdr.pages[0].keyframe
        fh = self._rdr.pages[0].parent.filehandle
        
        # Set the output for asynchronous reading
        self._out = output

        # Do the work
        offsets,bytecounts = self._chunk_indices(X,Y,Z)
        self._tile_indices = []
        for z in range(0,Z_tile_shape):
            for y in range(0,Y_tile_shape,ts):
                for x in range(0,X_tile_shape,ts):
                    self._tile_indices.append((x,y,z))
        
        self.logger.debug('read_image(): _tile_indices = {}'.format(self._tile_indices))
        
        with ThreadPoolExecutor(self.frontend.max_workers) as executor:
            executor.map(self._read_tile,fh.read_segments(offsets,bytecounts))

try:
    import bioformats,javabridge
    
    class JavaReader(bfio.base_classes.AbstractReader):
        
        logger = logging.getLogger("bfio.backends.JavaReader")
        
        def __init__(self, frontend):
            super().__init__(frontend)
            
            # Test to see if the loci_tools.jar is present
            if bfio.JARS == None:
                raise FileNotFoundError('The loci_tools.jar could not be found.')
            
        def read_metadata(self, update=False):
            # Wrap the ImageReader to get access to additional class methods
            rdr = javabridge.JClassWrapper('loci.formats.ImageReader')()
            
            rdr.setOriginalMetadataPopulated(True)
            
            # Access the OMEXML Service
            clsOMEXMLService = javabridge.JClassWrapper(
                'loci.formats.services.OMEXMLService')
            serviceFactory = javabridge.JClassWrapper(
                'loci.common.services.ServiceFactory')()
            service = serviceFactory.getInstance(clsOMEXMLService.klass)
            omexml = service.createOMEXMLMetadata()
            
            # Read the metadata
            rdr.setMetadataStore(omexml)
            rdr.setId(str(self.frontend._file_path))
            
            return OMEXML(omexml.dumpXML())
        
        def _read_tile(self, dims):
            
            # self.attach()
            
            out = self._out
            
            X,Y,Z,C,T = dims
            
            self.logger.debug('_read_tile(): dims = {}'.format(dims))
            x_range = min([self.frontend.X, X[1]+1024]) - X[1]
            y_range = min([self.frontend.Y, Y[1]+1024]) - Y[1]
            self.logger.debug('_read_tile(): x_range, y_range = {}, {}'.format(x_range,y_range))
            
            print(str(self.frontend._file_path))
            print('X,Y,Z,C,T = {},{},{},{},{}'.format(X[1],Y[1],Z[1],C[1],T[1]))
            with bioformats.ImageReader(str(self.frontend._file_path)) as reader:
                image = reader.read(c=C[1], z=Z[1], t=T[1],
                                    rescale=False,
                                    XYWH=(X[1], Y[1], x_range, y_range))
                print('image.shape = {}'.format(image.shape))
            
            out[Y[0]: Y[0]+image.shape[0],
                X[0]: X[0]+image.shape[1],
                Z[0],
                C[0],
                T[0]] = image

            # self.detach()
        
        def read_image(self,X,Y,Z,C,T,output):
            
            # Define tile bounds
            ts = self.frontend._TILE_SIZE
            X_tile_shape = X[1] - X[0]
            Y_tile_shape = Y[1] - Y[0]
            Z_tile_shape = Z[1] - Z[0]
            
            # Set the output for asynchronous reading
            self._out = output

            # Do the work
            self._tile_indices = []
            for t in range(len(T)):
                for c in range(len(C)):
                    for z in range(Z_tile_shape):
                        for y in range(0,Y_tile_shape,ts):
                            for x in range(0,X_tile_shape,ts):
                                self._tile_indices.append(((x,X[0]+x),
                                                        (y,Y[0]+y),
                                                        (z,Z[0]+z),
                                                        (c,C[c]),
                                                        (t,T[t])))
            
            self.logger.debug('read_image(): _tile_indices = {}'.format(self._tile_indices))
            
            with ThreadPoolExecutor(self.frontend.max_workers) as executor:
                executor.map(self._read_tile,self._tile_indices)
        
        def attach(self):
            javabridge.attach()
            
        def detach(self):
            javabridge.detach()

except ModuleNotFoundError:
    
    logger.warning('Java backend is not available. This could be due to a ' +
                   'missing dependency (javabridge or bioformats), or ' + 
                   'javabridge could not find the java runtime.')
    
    class JavaReader(bfio.base_classes.AbstractReader):
        
        def __init__(self, frontend):
            
            raise ImportError('JavaReader class unavailable. Could not import' +
                              ' javabridge and/or bioformats.')