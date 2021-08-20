
'''
A conversion utility built to convert abstract to primitive
This works for Threshold Apply and Gaussian Filter functions
Note change from jnius to jpype for handling conversions in ij_converter.py

'''

import imagej
import logging
import imglyb
import jpype
import numpy as np
import scyjava

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("ij_converter")
logger.setLevel(logging.INFO)

ij = None

## fill in types to convert
ABSTRACT_ITERABLES = [
    'IterableInterval',
    'Iterable',
]

IMG_ARRAYS = [
    'ArrayImg'
]

ABSTRACT_SCALARS = [
    'RealType',
]

SCALARS = [
    'double',
    'float',
    'long', #long type (int64) not supported by bfio
    'int',
    'short',
    'char',
    'byte',
    'boolean'
]

## recognize array objects as scalar objects + '[]'
ARRAYS = [s+'[]' for s in SCALARS]

def _java_setup():
    global IMGLYB_PRIMITIVES, PRIMITIVES, PRIMITIVE_ARRAYS
    IMGLYB_PRIMITIVES = {
        'float32'    : imglyb.types.FloatType,
        'float64'    : imglyb.types.DoubleType,
        'int8'       : imglyb.types.ByteType,
        'int16'      : imglyb.types.ShortType,
        'int32'      : imglyb.types.IntType,
        'int64'      : imglyb.types.LongType,
        'uint8'      : imglyb.types.UnsignedByteType,
        'uint16'     : imglyb.types.UnsignedShortType,
        'uint32'     : imglyb.types.UnsignedIntType,
        'uint64'     : imglyb.types.UnsignedLongType
    }
    # PRIMITIVES = {
    #     'double'     : jpype.JDouble,
    #     'float'      : jpype.JFloat,
    #     'long'       : jpype.JLong,
    #     'int'        : jpype.JInt,
    #     'short'      : jpype.JShort,
    #     'char'       : jpype.JChar,
    #     'byte'       : jpype.JByte,
    #     'boolean'    : jpype.JBoolean
    # }
    # PRIMITIVES = {
    #     'double'     : jpype.JDouble,
    #     'float'      : jpype.JFloat,
    #     'long'       : jpype.JLong,
    #     'int'        : jpype.JInt,
    #     'short'      : jpype.JShort,
    #     'char'       : jpype.JChar,
    #     'byte'       : jpype.JByte,
    #     'boolean'    : jpype.JBoolean
    # }
    PRIMITIVE_ARRAYS = {
        'double[]'     : jpype.JDouble[:],
        'float[]'      : jpype.JFloat[:],
        'long[]'       : jpype.JLong[:],
        'int[]'        : jpype.JInt[:],
        'short[]'      : jpype.JShort[:],
        'char[]'       : jpype.JChar[:],
        'byte[]'       : jpype.JByte[:],
        'boolean[]'    : jpype.JBoolean[:]
    }
    
scyjava.when_jvm_starts(_java_setup)

JAVA_CONVERT = {}
# JAVA_CONVERT.update({
#     t:lambda s,t,st: IMGLYB_PRIMITIVES[str(st)](st.type(s)) for t in ABSTRACT_SCALARS
# })
# JAVA_CONVERT.update({
#     t:lambda s,t,st: PRIMITIVES[t](float(s)) for t in SCALARS
# })
# JAVA_CONVERT.update({
#     t:lambda s,t,st: PRIMITIVE_ARRAYS[t]([float(si) for si in s.split(',')]) for t in ARRAYS
# })
JAVA_CONVERT.update({
    t: lambda s,t,st: IMGLYB_PRIMITIVES[str(st)](s) for t in SCALARS
})

JAVA_CONVERT.update({
    t:lambda s: imglyb.util.Views.iterable(ij.py.to_java(s)) for t in ABSTRACT_ITERABLES
})
JAVA_CONVERT.update({
    t:lambda s: imglyb.util._to_imglib(s) for t in IMG_ARRAYS
})

def to_java(np_array,java_type,java_dtype=None):

    if ij == None:
        raise ValueError('No imagej instance found.')

    if isinstance(np_array,type(None)):
        return None

    if java_type in JAVA_CONVERT.keys():
        if str(java_dtype) != 'None':
            out_array = JAVA_CONVERT[java_type](np_array,java_type,java_dtype)
        else:
            out_array = JAVA_CONVERT[java_type](np_array)
    else:
        logger.warning('Did not recognize type, {}, will pass default.'.format(java_type))
        out_array = ij.py.to_java(np_array)
        

    return out_array

def from_java(java_array,java_type):

    if ij == None:
        raise ValueError('No imagej instance found.')

    if ij.py.dtype(java_array) == bool:
        java_array = ij.op().convert().uint8(java_array)

    return ij.py.from_java(java_array)

if __name__ == '__main__':
    
    import traceback
    from bfio import BioReader, BioWriter
    from pathlib import Path
    
    # Bioformats throws a debug message, disable the loci debugger to mute it
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")
    scyjava.when_jvm_starts(disable_loci_logs)
    
    print('Starting JVM...')
    
    # This is the version of ImageJ pre-downloaded into the docker container
    ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4", headless=True)
    
    # ArrayImgs = scyjava.jimport('net.imglib2.img.array.ArrayImgs')
    # UnsafeUtil = scyjava.jimport('net.imglib2.img.basictypelongaccess.unsafe.UnsafeUtil')
    # Arrays = scyjava.jimport('java.util.Arrays')
    # OwningFloatUnsafe = scyjava.jimport('net.imglib2.img.basictypelongaccess.unsafe.owning.OwningFloatUnsafe')
    # Fraction = scyjava.jimport('net.imglib2.util.Fraction')
    # LongStream = scyjava.jimport('java.util.stream.LongStream')
    
    NUMPY_TYPES = {
        'uint8': (np.uint8, imglyb.types.UnsignedByteType),
        'byte': (np.int8, imglyb.types.ByteType),
        'uint16': (np.uint16, imglyb.types.UnsignedShortType),
        'short': (np.int16, imglyb.types.ShortType),
        'uint32': (np.uint32, imglyb.types.UnsignedIntType),
        'int': (np.int32, imglyb.types.IntType),
        'float': (np.float32, imglyb.types.DoubleType),
        'double': (np.float64, imglyb.types.UnsignedLongType),
        'long' : (np.int64, imglyb.types.LongType)
    }
    
    TEST_CASES = [
        'double',
        'float',
        'long',
        'int',
        'short',
        'char',
        'byte',
        'boolean'
    ]
    
    def tester(t, ij):
        try:
                print('Testing {} data type...'.format(t))
                shape = (2048,2048)
                print('Creating Array...')
                array = np.random.randint(0,255,size=shape,dtype=np.uint16)
                print('Converting Array...')
                array = NUMPY_TYPES[t][0](array)
                dtype0 = ij.py.dtype(array)
                print('The initial data type is {}'. format(dtype0))
                temp_path = Path(__file__).with_name('data-convert-temp')
                print('Writing image array to file...')
                with BioWriter(temp_path) as writer:
                    writer.X = shape[0]
                    writer.Y = shape[1]
                    writer.dtype = array.dtype
                    writer[:] = array[:]
                print('Reading image from file...')
                arr = BioReader(temp_path)
                print('Getting data type after reading image...')
                dtype1 = ij.py.dtype(arr[:,:,0:1,0,0])
                print('Data type after reading image is {}'.format(dtype1))
                #print('Trying to convert to PlanarImg')
                #planarimg = ij.planar(arr)
                if dtype0 != dtype1:
                    print('Manully forcing data type back to {}'.format(dtype0))
                    arr = NUMPY_TYPES[t][0](arr[:,:,0:1,0,0])
                    print('Converting to Java object...')
                    arr = to_java(np.squeeze(arr),'ArrayImg')
                    print('Getting data type after manually forcing...')
                    dtype2 = ij.py.dtype(arr)
                    print('Data type after manual forcing is {}'.format(dtype2))
                    val_dtype = dtype2
                else:
                    arr = to_java(np.squeeze(arr[:,:,0:1,0,0]),'ArrayImg')
                    val_dtype = dtype1
                    
                value = 5
                print('Converting input (value) to Java primitive type {}...'.format(val_dtype))
                val = to_java(value, t, val_dtype)
                print('Calling ImageJ op...')
                out = ij.op().math().add(arr,val)
                print('The op was SUCCESSFUL with data type {}'.format(t))
        
        except:
            print('Testing data type {} was NOT SUCCESSFUL'.format(t))
            #print(traceback.format_exc())
        
        finally:
            print('Shutting down JVM...')
            del ij
            jpype.shutdownJVM()
    
    tester(TEST_CASES[0], ij)