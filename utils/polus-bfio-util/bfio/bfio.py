"""
The two primary classes this code uses are bfio.BioReader and bfio.BioWriter:
bfio.BioReader will read any image that the Bioformats tool can read.
bfio.BioWriter will only save images as an ome tiled tiff.

Example usage is provided in the comments to each class.

Required packages:
javabridge (also requires jdk > 8)
python-bioformats
numpy

Note: Prior to reading or writing using these classes, the javabridge session
      must be started. This may be automated in the future.
"""
import bioformats
import numpy as np
import os
import javabridge as jutil

def make_ome_tiff_writer_class():
    '''Return a class that wraps loci.formats.out.OMETiffWriter'''
    class_name = 'loci/formats/out/OMETiffWriter'
    IFormatWriter = bioformats.formatwriter.make_iformat_writer_class(class_name)

    class OMETiffWriter(IFormatWriter):

        new_fn = jutil.make_new('loci/formats/out/OMETiffWriter', '()V')
        
        def __init__(self):

            self.new_fn()

    return OMETiffWriter

class BioReader():
    # Note: the javabridge connection must be started before initializing a 
    # BioReader object. The reason for this is that the java VM can only be
    # initialized once. Doing it within the code would cause it to run possibly
    # without shutting down, so the javabridge connection must be handled
    # outside of the class.
    _file_path = None
    _metadata = None
    _xyzct = None
    _pix = None
    
    # Set constants for opening images
    _MAX_BYTES = 2**30
    _BPP = {'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float': 4,
            'double': 8}
    _TILE_SIZE = 2**10
    
    ''' Initialization '''
    def __init__(self,file_path):
        self._file_path = file_path
        self._metadata = self.read_metadata()
        
        # Information about image dimensions
        self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                       'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                       'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                       'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                       'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints
        
        # Information about data type and loading
        self._pix = {'type':self._metadata.image().Pixels.get_PixelType(),            # string indicating pixel type
                     'bpp':self._BPP[self._metadata.image().Pixels.get_PixelType()],  # bytes per pixel
                     'spp':self._metadata.image().Pixels.Channel().SamplesPerPixel}   # samples per pixel
        self._pix['chunk'] = self._MAX_BYTES/(self._pix['spp']*self._pix['bpp'])      # number of pixels to load at a time
        self._pix['interleaved'] = self._pix['spp']>1                                 # determine if channels are interleaved
        
    ''' Utility functions to get basic metadata'''
    def channel_names(self):
        image = self._metadata.image()
        return [image.Pixels.Channel(i).Name for i in range(0,self._xyzct['C'])]
    
    def num_x(self):
        return self._xyzct['X']
    def num_y(self):
        return self._xyzct['Y']
    def num_z(self):
        return self._xyzct['Z']
    def num_c(self):
        return self._xyzct['C']
    def num_t(self):
        return self._xyzct['T']
            
    def physical_size_x(self, psize=None, units=None):
        return (self._metadata.image(0).Pixels.PhysicalSizeX,self._metadata.image(0).Pixels.PhysicalSizeXUnit)
        
    def physical_size_y(self, psize=None, units=None):
        return (self._metadata.image(0).Pixels.PhysicalSizeY,self._metadata.image(0).Pixels.PhysicalSizeYUnit)
        
    def physical_size_z(self, psize=None, units=None):
        return (self._metadata.image(0).Pixels.PhysicalSizeZ,self._metadata.image(0).Pixels.PhysicalSizeZUnit)
    
    ''' Get raw metadata '''
    def read_metadata(self,update=False):
        if self._metadata and not update:
            return self._metadata
        # For some reason, tif files need to use the generic ImageReader while everything else
        # can use the OMETiffReader.
        if self._file_path.endswith('.ome.tif'):
            rdr = jutil.JClassWrapper('loci.formats.in.OMETiffReader')()
        else:
            rdr = jutil.JClassWrapper('loci.formats.ImageReader')()
        rdr.setOriginalMetadataPopulated(True)
        clsOMEXMLService = jutil.JClassWrapper('loci.formats.services.OMEXMLService')
        serviceFactory = jutil.JClassWrapper('loci.common.services.ServiceFactory')()
        service = serviceFactory.getInstance(clsOMEXMLService.klass)
        omexml = service.createOMEXMLMetadata()
        rdr.setMetadataStore(omexml)
        rdr.setId(self._file_path)
        
        self._metadata = bioformats.OMEXML(omexml.dumpXML())
        return self._metadata
    
    ''' Utility functions to validate image dimensions '''
    def _val_xyz(self,xyz,axis):
        if not xyz:
            xyz = [0,self._xyzct[axis]]
        else:
            if len(xyz)!=2:
                ValueError('{} must be a list or tuple of length 2.'.format(axis))
            elif xyz[0]<0:
                ValueError('{}[0] must be greater than or equal to 0.'.format(axis))
            elif xyz[1]>self._xyzct[axis]:
                ValueError('{}[1] cannot be greater than the maximum of the dimension ({}).'.format(axis,self._xyzct[axis]))
        return xyz
    
    def _val_ct(self,ct,axis):
        if not ct:
            ct = [t for t in range(0,self._xyzct[axis])] # number of timepoints
        else:
            if np.any(np.greater_equal(self._xyzct[axis],ct)):
                ValueError('At least one of the {}-indices was larger than largest index ({}).'.format(axis,self._xyzct[axis]-1))
            elif np.any(np.less(0,ct)):
                ValueError('At least one of the {}-indices was less than 0.'.format(axis))
            elif len(ct)==0:
                ValueError('At least one {}-index must be selected.'.format(axis))
        return ct
    
    def read_image(self,X=None,Y=None,Z=None,C=None,T=None,series=None):
        '''
        image = BioReader.open()
        
        -Inputs-
        X,Y,Z -     If None, then load all values in the given dimension. If a
                    2-tuple, load values within the range.
        C,T -       If None, then load all channels/timepoints, respectively. If a
                    list or tuple, load indicated channels/timepoints.
        series -    Placeholder, in the future will open multi-series images
        
        -Outputs-
        image - a 5D numpy array with dimensions corresponding to (y,x,z,channel,timepoint)
        
        -Example-
        # Import javabridge and start the vm
        import javabridge
        javabridge.start_vm(class_path=bioformats.JARS)
        
        # Path to bioformats supported image
        image_path = 'path/to/image'
        
        # Create the BioReader object
        bf = BioReader(image_path)
        
        # Load the full image
        image = bf.read_image()
        
        # Only load the first 256x256 pixels, will still load all Z,C,T dimensions
        image = bf.read_image(X=(0,256),Y=(0,256))
        
        # Only load the second channel
        image = bf.read_image(C=[1])
        
        # Done executing program, so kill the vm. If the program needs to be run
        # again, a new interpreter will need to be spawned to start the vm.
        javabridge.kill_vm()
        '''
        
        # Validate inputs
        X = self._val_xyz(X,'X')
        Y = self._val_xyz(Y,'Y')
        Z = self._val_xyz(Z,'Z')
        C = self._val_ct(C,'C')
        T = self._val_ct(T,'T')
        
        # Set options for file loading based on metadata
        open_in_parts = (X[1]-X[0])*(Y[1]-Y[0])>self._pix['chunk']  # open in parts if more than max_bytes
        
        # Initialize the output
        I = np.zeros([Y[1]-Y[0],X[1]-X[0],Z[1]-Z[0],len(C),len(T)],self._pix['type'])
        
        # Do the work
        with bioformats.ImageReader(self._file_path) as reader:
            for ti,t in zip(range(0,len(T)),T):
                for zi,z in zip(range(0,Z[1]-Z[0]),range(Z[0],Z[1])):
                    if not open_in_parts:
                        if self._pix['interleaved']:
                            I_temp = reader.read(c=None,z=z,t=t,rescale=False,XYWH=(X[0],Y[0],X[1]-X[0],Y[1]-Y[0]))
                            for ci,c in zip(range(0,len(C)),C):
                                I[:,:,zi,ci,ti] = I_temp[:,:,c]
                        else:
                            for ci,c in zip(range(0,len(C)),C):
                                I[:,:,zi,ci,ti] = reader.read(c=c,z=z,t=t,rescale=False,XYWH=(X[0],Y[0],X[1]-X[0],Y[1]-Y[0]))
                    
                    else:
                        if self._pix['interleaved']:
                            for x in range(X[0],X[1],self._TILE_SIZE):
                                x_max = np.min([x+self._TILE_SIZE,X[1]])
                                x_range = x_max - x
                                for y in range(Y[0],Y[1],self._TILE_SIZE):
                                    y_max = np.min([y+self._TILE_SIZE,Y[1]])
                                    y_range = y_max - y
                                    I[y-Y[0]:y_max-Y[0],x-X[0]:x_max-X[0],zi,:,ti] = reader.read(c=None,z=z,t=t,rescale=False,XYWH=(x,y,x_range,y_range))
                        else:
                            for ci,c in zip(range(0,len(C)),C):
                                for x in range(X[0],X[1],self._TILE_SIZE):
                                    x_max = np.min([x+self._TILE_SIZE,X[1]])
                                    x_range = x_max - x
                                    for y in range(Y[0],Y[1],self._TILE_SIZE):
                                        y_max = np.min([y+self._TILE_SIZE,Y[1]])
                                        y_range = y_max - y
                                        I[y-Y[0]:y_max-Y[0],x-X[0]:x_max-X[0],zi,ci,ti] = reader.read(c=c,z=z,t=t,rescale=False,XYWH=(x,y,x_range,y_range))
            
        return I
    
class BioWriter():
    # Note: the javabridge connection must be started before initializing a 
    # BioReader object. The reason for this is that the java VM can only be
    # initialized once. Doing it within the code would cause it to run possibly
    # without shutting down, so the javabridge connection must be handled
    # outside of the class.
    _file_path = None
    _metadata = None
    _xyzct = None
    _pix = None
    
    # Set constants for opening images
    _MAX_BYTES = 2**30
    _BPP = {'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'uint32': 4,
            'int32': 4,
            'float': 4,
            'double': 8}
    _TILE_SIZE = 2**10
    
    __writer = None
    
    def __init__(self,file_path,image=None,
                 X=None,Y=None,Z=None,C=None,T=None,
                 metadata=None):
        
        self._file_path = file_path
        
        if metadata:
            assert isinstance(metadata, bioformats.omexml.OMEXML)
            self._metadata = metadata
            self._xyzct = {'X': self._metadata.image().Pixels.get_SizeX(),  # image width
                           'Y': self._metadata.image().Pixels.get_SizeY(),  # image height
                           'Z': self._metadata.image().Pixels.get_SizeZ(),  # image depth
                           'C': self._metadata.image().Pixels.get_SizeC(),  # number of channels
                           'T': self._metadata.image().Pixels.get_SizeT()}  # number of timepoints
            self._pix = {'type':self._metadata.image().Pixels.get_PixelType(),            # string indicating pixel type
                         'bpp':self._BPP[self._metadata.image().Pixels.get_PixelType()],  # bytes per pixel
                         'spp':1}   # samples per pixel
            self._pix['chunk'] = self._MAX_BYTES/(self._pix['spp']*self._pix['bpp'])      # number of pixels to load at a time
            self._pix['interleaved'] = False
            self._metadata.image(0).Name = file_path
            self._metadata.image().Pixels.channel_count = self._xyzct['C']
            self._metadata.image().Pixels.DimensionOrder = bioformats.omexml.DO_XYZCT
        elif np.any(image):
            assert len(image.shape)==5, "Image must be 5-dimensional (x,y,z,c,t)."
            x = X if X else image.shape[1]
            y = Y if Y else image.shape[0]
            z = Z if Z else image.shape[2]
            c = C if C else image.shape[3]
            t = T if T else image.shape[4]
            self._xyzct = {'X': x,  # image width
                           'Y': y,  # image height
                           'Z': z,  # image depth
                           'C': c,  # number of channels
                           'T': t}  # number of timepoints
            self._pix = {'type':str(image.dtype),            # string indicating pixel type
                         'bpp':self._BPP[str(image.dtype)],  # bytes per pixel
                         'spp':1}                            # samples per pixel
            self._pix['chunk'] = self._MAX_BYTES/(self._pix['spp']*self._pix['bpp'])      # number of pixels to load at a time
            self._pix['interleaved'] = False
            self._metadata = self._minimal_xml()
        else:
            x = X if X else 1
            y = Y if Y else 1
            z = Z if Z else 1
            c = C if C else 1
            t = T if T else 1
            self._xyzct = {'X': x,  # image width
                           'Y': y,  # image height
                           'Z': z,  # image depth
                           'C': c,  # number of channels
                           'T': t}  # number of timepoints
            self._pix = {'type':'uint8',            # string indicating pixel type
                         'bpp':self._BPP['uint8'],  # bytes per pixel
                         'spp':1}                   # samples per pixel
            self._pix['chunk'] = self._MAX_BYTES/(self._pix['spp']*self._pix['bpp'])      # number of pixels to load at a time
            self._pix['interleaved'] = False
            self._metadata = self._minimal_xml()
        
        if file_path.endswith('.ome.tif'):
            ValueError("The file name that will be saved to must have extension .ome.tif")
    
    def _minimal_xml(self):
        assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
        omexml = bioformats.omexml.OMEXML()
        omexml.image(0).Name = os.path.split(self._file_path)[1]
        p = omexml.image(0).Pixels
        assert isinstance(p, bioformats.omexml.OMEXML.Pixels)
        p.SizeX = self._xyzct['X']
        p.SizeY = self._xyzct['Y']
        p.SizeZ = self._xyzct['Z']
        p.SizeC = self._xyzct['C']
        p.SizeT = self._xyzct['T']
        p.DimensionOrder = bioformats.omexml.DO_XYZCT
        p.PixelType = self._pix['type']
        if self._xyzct['C'] > 1:
            p.channel_count = self._xyzct['C']
        return omexml
            
    def _init_writer(self):
        # If file already exists, delete it before initializing again.
        # If this is not done, then the file will be appended.
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        
        w_klass = make_ome_tiff_writer_class()
        w_klass.setId = jutil.make_method('setId', '(Ljava/lang/String;)V',
                                        'Sets the current file name.')
        w_klass.saveBytesXYWH = jutil.make_method('saveBytes', '(I[BIIII)V',
                                                'Saves the given byte array to the current file')
        w_klass.close = jutil.make_method('close','()V',
                                        'Closes currently open file(s) and frees allocated memory.')
        w_klass.setTileSizeX = jutil.make_method('setTileSizeX','(I)I',
                                                'Set tile size width in pixels.')
        w_klass.setTileSizeY = jutil.make_method('setTileSizeY','(I)I',
                                                'Set tile size height in pixels.')
        w_klass.getTileSizeX = jutil.make_method('getTileSizeX','()I',
                                                'Set tile size width in pixels.')
        w_klass.getTileSizeY = jutil.make_method('getTileSizeY','()I',
                                                'Set tile size height in pixels.')
        w_klass.setBigTiff = jutil.make_method('setBigTiff','(Z)V',
                                               'Set the BigTiff flag.')
        writer = w_klass()
        
        # Set the BigTiff flag if needed, must be done before anything else
        if self.num_x() * self.num_y() * self._pix['spp'] * self._pix['bpp'] > self._MAX_BYTES:
            writer.setBigTiff(True)
        
        script = """
        importClass(Packages.loci.formats.services.OMEXMLService,
                    Packages.loci.common.services.ServiceFactory);
        var service = new ServiceFactory().getInstance(OMEXMLService);
        var metadata = service.createOMEXMLMetadata(xml);
        var writer = writer
        writer.setMetadataRetrieve(metadata);
        """
        jutil.run_script(script,
                         dict(path=self._file_path,
                              xml=self._metadata.to_xml(),
                              writer=writer))
        writer.setId(self._file_path)
        writer.setInterleaved(False)
        writer.setCompression("LZW")
        x = writer.setTileSizeX(self._TILE_SIZE)
        y = writer.setTileSizeY(self._TILE_SIZE)
        
        self._pix['chunk'] = self._MAX_BYTES/(self._pix['spp']*self._pix['bpp'])      # number of pixels to load at a time
            
        self.__writer = writer
        
    def pixel_type(self,dtype=None):
        if dtype:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert dtype in self._BPP.keys(), "Invalid data type."
            self._metadata.image(0).Pixels.PixelType = dtype
            self._pix['type'] = dtype
        else:
            return self._metadata.image(0).Pixels.PixelType
        
    def channel_names(self,cnames=None):
        if cnames:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert len(cnames)==self._xyzct['C'], "Number of names does not match number of channels."
            for i in range(0,len(cnames)):
                self._metadata.image(0).Pixels.Channel(i).Name = cnames[i]
        else:
            image = self._metadata.image()
            return [image.Pixels.Channel(i).Name for i in range(0,self._xyzct['C'])]
    
    def num_x(self,X=None):
        if X:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert X>=1
            self._metadata.image(0).Pixels.SizeX = X
            self._xyzct['X'] = X
        else:
            return self._xyzct['X']
    def num_y(self,Y=None):
        if Y:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert Y>=1
            self._metadata.image(0).Pixels.SizeY = Y
            self._xyzct['Y'] = Y
        else:
            return self._xyzct['Y']
    def num_z(self,Z=None):
        if Z:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert Z>=1
            self._metadata.image(0).Pixels.SizeZ = Z
            self._xyzct['Z'] = Z
        else:
            return self._xyzct['Z']
    def num_c(self,C=None):
        if C:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert C>=1
            self._metadata.image(0).Pixels.SizeC = C
            self._xyzct['C'] = C
        else:
            return self._xyzct['C']
    def num_t(self,T=None):
        if T:
            assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
            assert T>=1
            self._metadata.image(0).Pixels.SizeT = T
            self._xyzct['T'] = T
        else:
            return self._xyzct['T']
        
    def physical_size_x(self, psize=None, units=None):
        if not psize and not units:
            return (self._metadata.image(0).Pixels.PhysicalSizeX,self._metadata.image(0).Pixels.PhysicalSizeXUnit)
        elif not psize or not units:
            ValueError("Both psize and units must be input.")
        assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
        self._metadata.image(0).Pixels.PhysicalSizeX = psize
        self._metadata.image(0).Pixels.PhysicalSizeXUnit = units
        
    def physical_size_y(self, psize=None, units=None):
        if not psize and not units:
            return (self._metadata.image(0).Pixels.PhysicalSizeY,self._metadata.image(0).Pixels.PhysicalSizeYUnit)
        elif not psize or not units:
            ValueError("Both psize and units must be input.")
        assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
        self._metadata.image(0).Pixels.PhysicalSizeY = psize
        self._metadata.image(0).Pixels.PhysicalSizeYUnit = units
        
    def physical_size_z(self, psize=None, units=None):
        if not psize and not units:
            return (self._metadata.image(0).Pixels.PhysicalSizeZ,self._metadata.image(0).Pixels.PhysicalSizeZUnit)
        elif not psize or not units:
            ValueError("Both psize and units must be input.")
        assert not self.__writer, "The image has started to be written. To modify the xml again, reinitialize."
        self._metadata.image(0).Pixels.PhysicalSizeZ = psize
        self._metadata.image(0).Pixels.PhysicalSizeZUnit = units
    ''' Utility functions to validate image dimensions '''
    def _val_xyz(self,xyz,axis):
        if len(xyz)!=2:
            ValueError('{} must be a scalar.'.format(axis))
        elif xyz[0]<0:
            ValueError('{}[0] must be greater than or equal to 0.'.format(axis))
        elif xyz[1]>self._xyzct[axis]:
            ValueError('{}[1] cannot be greater than the maximum of the dimension ({}).'.format(axis,self._xyzct[axis]))
    
    def _val_ct(self,ct,axis):
        if np.any(np.greater_equal(self._xyzct[axis],ct)):
            ValueError('At least one of the {}-indices was larger than largest index ({}).'.format(axis,self._xyzct[axis]-1))
        elif np.any(np.less(0,ct)):
            ValueError('At least one of the {}-indices was less than 0.'.format(axis))
        elif len(ct)==0:
            ValueError('At least one {}-index must be selected.'.format(axis))
        elif isinstance(ct,list):
            TypeError("The values for {} must be a list.".format(axis))
    
    def write_image(self, image, X=None, Y=None, Z=None, C=None, T=None):
        '''
        image = BioReader.open()
        
        -Inputs-
        X,Y,Z -     If None, then load all values in the given dimension. If a
                    2-tuple, load values within the range.
        C,T -       If None, then load all channels/timepoints, respectively. If a
                    list or tuple, load indicated channels/timepoints.
        series -    Placeholder, in the future will open multi-series images
        
        -Outputs-
        image - a 5D numpy array with dimensions corresponding to (y,x,z,channel,timepoint)
        
        -Example-
        # Import javabridge and start the vm
        jutil.start_vm(class_path=bioformats.JARS)
        
        # Path to bioformats supported image
        image_path = 'AMD-CD_1_Maximumintensityprojection_c2.ome.tif'
        
        # Create the BioReader object
        br = BioReader(image_path)
        
        # Load the full image
        image = br.read_image()
        
        # Save the image, rename the channels
        bw = BioWriter("New_" + image_path,image=image)
        bw.channel_names(["Empty","ZO1","Empty"])
        bw.write_image(image)
        bw.close_image()
        
        # Only save one channel
        bw = BioWriter("ZO1_" + image_path,image=image)
        bw.num_c(1)
        bw.write_image(image[:,:,0,1,0].reshape((image.shape[0],image.shape[1],1,1,1)))
        bw.close_image()
        
        # List the channel names
        print(bw.channel_names())
        
        # Done executing program, so kill the vm. If the program needs to be run
        # again, a new interpreter will need to be spawned to start the vm.
        jutil.kill_vm()
        '''
        assert len(image.shape)==5, "Image must be 5-dimensional (x,y,z,c,t)."
        
        # Set pixel bounds
        if not X:
            X = [0]
        if not Y:
            Y = [0]
        if not Z:
            Z = [0]
        X.append(image.shape[1]+X[0])
        Y.append(image.shape[0]+Y[0])
        Z.append(image.shape[2]+Z[0])
        
        C = C if C else [c for c in range(0,self._xyzct['C'])]
        T = T if T else [t for t in range(0,self._xyzct['T'])]
        
        # Validate inputs
        self._val_xyz(X,'X')
        self._val_xyz(Y,'Y')
        self._val_xyz(Z,'Z')
        self._val_ct(C,'C')
        self._val_ct(T,'T')
        
        # Set options for file loading based on metadata
        save_in_parts = (X[1]-X[0])*(Y[1]-Y[0])>self._pix['chunk']  # open in parts if more than max_bytes
        
        # Initialize the writer if it hasn't already been initialized
        if not self.__writer:
            self._init_writer()
        
        # Do the work\
        for ti,t in zip(range(0,len(T)),T):
            for zi,z in zip(range(0,Z[1]-Z[0]),range(Z[0],Z[1])):
                if not save_in_parts:
                    for ci,c in zip(range(0,len(C)),C):
                        index = z + self._xyzct['Z'] * c + self._xyzct['Z'] * self._xyzct['C'] * t
                        pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(image[:,:,zi,ci,ti], self._pix['type'])
                        self.__writer.saveBytesXYWH(index, pixel_buffer,X[0],Y[0],X[1]-X[0],Y[1]-Y[0])
                else:
                    for ci,c in zip(range(0,len(C)),C):
                        index = z + self._xyzct['Z'] * c + self._xyzct['Z'] * self._xyzct['C'] * t
                        for x in range(X[0],X[1],self._TILE_SIZE):
                            x_max = np.min([x+self._TILE_SIZE,X[1]])
                            x_range = x_max - x
                            for y in range(Y[0],Y[1],self._TILE_SIZE):
                                y_max = np.min([y+self._TILE_SIZE,Y[1]])
                                y_range = y_max - y
                                
                                pixel_buffer = bioformats.formatwriter.convert_pixels_to_buffer(image[y-Y[0]:y_max-Y[0],x-X[0]:x_max-X[0],zi,ci,ti], self._pix['type'])
                                self.__writer.saveBytesXYWH(index, pixel_buffer,x,y,x_range,y_range)
                                
    def close_image(self):
        self.__writer.close()