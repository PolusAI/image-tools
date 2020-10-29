from tifffile import tifffile
from pathlib import Path
import bioformats,javabridge,typing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from bfio.OmeXml import OMEXML
import bfio.base_classes

class PythonReader(bfio.base_classes.AbstractReader):

    def __init__(self, frontend):
        super().__init__(frontend)

        self._rdr = tifffile.TiffFile(self.frontend._file_path)
        
    def read_metadata(self):
        return OMEXML(self._rdr.ome_metadata)
    
    def _chunk_indices(self,X,Y,Z):
        assert len(X) == 2
        assert len(Y) == 2
        assert len(Z) == 2
        
        offsets = []
        bytecounts = []
        
        ts = self.frontend._TILE_SIZE
        
        x_tiles = np.arange(X[0]//ts,np.ceil(X[1]/ts),dtype=int)
        y_tile_stride = np.ceil(self.frontend.x/ts).astype(int)
        
        for z in range(Z[0],Z[1]):
            for y in range(Y[0]//ts,int(np.ceil(Y[1]/ts))):
                y_offset = int(y * y_tile_stride)
                ind = (x_tiles + y_offset).tolist()
                
                offsets.extend([self._rdr.pages[z].dataoffsets[i] for i in ind])
                bytecounts.extend([self._rdr.pages[z].databytecounts[i] for i in ind])
        
        return offsets,bytecounts
    
    def _read_tile(self, args):
        
        keyframe = self._keyframe
        out = self._out
        
        w,l,d = self._tile_indices[args[1]]
        
        # copy decoded segments to output array
        segment, _, shape = keyframe.decode(*args)
        
        if segment is None:
            segment = keyframe.nodata
        
        out[l: l + shape[1],
            w: w + shape[2],
            d,0,0] = segment.squeeze()
    
    def read_image(self,X,Y,Z,C,T,output):
        if (len(C)>1 and C[0]!=0) or (len(T)>0 and T[0]!=0):
            raise Warning('More than channel 0 was specified for either channel or timepoint data.' + \
                          'For the Python backend, only the first channel/timepoint will be loaded.')
        
        # Define tile bounds
        ts = self.frontend._TILE_SIZE
        x_range = X[1]-X[0]
        y_range = Y[1]-Y[0]
        X_tile_start = (X[0]//ts) * ts
        Y_tile_start = (Y[0]//ts) * ts
        X_tile_end = np.ceil(X[1]/ts).astype(int) * ts
        Y_tile_end = np.ceil(Y[1]/ts).astype(int) * ts
        X_tile_shape = X_tile_end - X_tile_start
        Y_tile_shape = Y_tile_end - Y_tile_start
        Z_tile_shape = Z[1]-Z[0]
        
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
        
        with ThreadPoolExecutor(self.frontend.max_workers) as executor:
            executor.map(self._read_tile,fh.read_segments(offsets,bytecounts))
            
        xi = X[0] - ts*(X[0]//ts)
        yi = Y[0] - ts*(Y[0]//ts)

        return self._out[yi:yi+y_range,xi:xi+x_range,...]

class JavaReader(bfio.base_classes.AbstractReader):
    
    def __init__(self, frontend):
        super().__init__(frontend)
        
        # Test to see if the loci_tools.jar is present
        if bfio.JARS == None:
            raise FileNotFoundError('The loci_tools.jar could not be found.')
        
        # Get the reader
        self._rdr = bioformats.ImageReader(str(self.frontend._file_path.absolute()))
        
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
    
    def read_image(self,X,Y,Z,C,T):
        pass
    
    def attach(self):
        javabridge.attach()
        
    def detach(self):
        javabridge.detach()