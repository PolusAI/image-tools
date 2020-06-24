# distutils: language=c++
cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport fill_n

ctypedef fused uint_ind:
    unsigned int [:]
    unsigned long [:]

""" Define processor intrinsics

This section of code makes specific processor instructions available
to Cython. These processor instructions permit fast looping through
pixels when compressing data using the run_length_encoding method.

These intrinsics work for the majority of non-arm processors released
after 2015.

"""
cdef extern from "x86intrin.h":
    ctypedef int __m128i
    ctypedef int __m256i

    long long _mm_popcnt_u64(unsigned long long a) nogil

    long long _pext_u64(unsigned long long a,unsigned long long b) nogil

    int _mm_movemask_epi8(__m128i a) nogil

    long long _mm_cvtsi128_si64(__m128i a) nogil

    __m128i _mm_setzero_si128() nogil

    __m128i _mm_slli_si128(__m128i a,int b) nogil

    __m128i _mm_loadu_si128(__m128i* d) nogil

    __m128i _mm_cmpgt_epi8(__m128i a,__m128i b) nogil

    __m128i _mm_xor_si128(__m128i a,__m128i b) nogil

    __m128i _mm_srli_epi64(__m128i a,int i) nogil

    __m128i _mm_shuffle_epi8(__m128i a,__m128i b) nogil

    __m256i _mm256_add_epi16(__m256i a, __m256i b) nogil
    
    __m128i _mm_set_epi8(char e15,char e14,char e13,char e12,
                         char e11,char e10,char e9, char e8,
                         char e7, char e6, char e5, char e4,
                         char e3, char e2, char e1, char e0) nogil

    __m256i _mm256_set_epi8(char e31,char e30,char e29,char e28,
                            char e27,char e26,char e25,char e24,
                            char e23,char e22,char e21,char e20,
                            char e19,char e18,char e17,char e16,
                            char e15,char e14,char e13,char e12,
                            char e11,char e10,char e9, char e8,
                            char e7, char e6, char e5, char e4,
                            char e3, char e2, char e1, char e0) nogil
    
    __m256i _mm256_set_epi64x(long long e3,long long e2,
                              long long e1,long long e0) nogil

    __m256i _mm256_set1_epi8(char a) nogil

    __m256i _mm256_set1_epi16(short a) nogil

    __m256i _mm256_set1_epi32(int a) nogil

    __m256i _mm256_set1_epi64x(long long a) nogil

    __m256i _mm256_srlv_epi64(__m256i a, __m256i b) nogil

    __m256i _mm256_and_si256(__m256i a, __m256i b) nogil

    __m256i _mm256_shuffle_epi8(__m256i a, __m256i b) nogil

    __m256i _mm256_add_epi16(__m256i a, __m256i b) nogil

    __m256i _mm256_add_epi32(__m256i a, __m256i b) nogil

    void _mm256_storeu_si256(__m256i* a,__m256i b) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef np.ndarray run_length_encode_16(unsigned char [:] image,tuple shape):
    """run_length_encode Create run length encoded pixels

    This function will run on a matrix with arbitrary number of dimensions.
    It is best to use the appropriate run length encoding algorithm for a
    specified number of dimensions if it exists.

    This function finds consecutive pixels along the final dimension
    that have non-zero values. Then, the first and last dimension index
    along with the other coordinates are stored in a numpy array. This
    results in a 2d numpy array with ndim+1 columns for each column object:
    [...,last_dim_index_start,last_dim_index_end]

    The input image must be a linear array (i.e. use reshape(-1)).
    The input image must be np.uint8 or bool data type.

    Note: This function is the main bottleneck in the labeling algorithm.
        This method generally takes up 50-90% or more of the time required
        image labeling depending on the size and number of objects in the
        the image. This function can be made to run faster by specializing
        for a specific number of dimensions, particularly in the section
        where values are stored. It may be a good idea to do this for common
        matrix dimensions (2d and 3d), since the performance improvement
        would be noticeable.

    Args:
        image (numpy.ndarray): An n-dimensional image reshaped to a linear
            array of pixels.
        shape (tuple): The shape of the image
        
    Outputs:
        numpy.ndarray: Indices of pixel objects
    """

    ''' Set up AVX vectors '''
    # Vector to hold pixel values
    cdef __m128i v
    # Vector to hold 
    cdef __m128i mask
    # Vector for pixel indices
    cdef __m256i edges
    # Byte shuffling constants
    cdef __m128i pshufbcnst = _mm_set_epi8(0x80,  0x80,  0x80,  0x80,
                                           0x80,  0x80,  0x80,  0x80,
                                           0x0E,  0x0C,  0x0A,  0x08,
                                           0x06,  0x04,  0x02,  0x00)
    cdef __m256i shf        = _mm256_set_epi8(0x80,0x0F,  0x80,0x07,  0x80,0x0E,  0x80,0x06,
                                              0x80,0x0D,  0x80,0x05,  0x80,0x0C,  0x80,0x04,
                                              0x80,0x0B,  0x80,0x03,  0x80,0x0A,  0x80,0x02,
                                              0x80,0x09,  0x80,0x01,  0x80,0x08,  0x80,0x00)
    # Bit shifts to decode 4-bit packing
    cdef __m256i shft       = _mm256_set_epi64x(0x04,0x00,0x04,0x00)
    # 4-bit mask
    cdef __m256i vmsk       = _mm256_set1_epi8(0x0F)
    # Indexing
    cdef __m256i cnst15     = _mm256_set1_epi16(0x000F)
    cdef __m256i i_vec      = _mm256_set1_epi16(0)

    ''' Constants for bit operations '''
    # Casts 4-bit boolean to position index
    cdef unsigned long long cast_index = 0xFEDCBA9876543210
    # Used to determine if pixel runs were detected
    cdef unsigned int detect_edge = 0xFFFFFFFE
    # Clear the lowest value
    cdef unsigned long long clear_lowest = 0xFFFFFFFFFFFFFFF0
    # Holds (16) 4-bit indices
    cdef unsigned long long ind

    ''' Temporary and output vectors '''
    cdef vector[np.uint16_t] temp
    temp.resize(16,0)
    cdef vector[np.uint32_t] output
    
    ''' Looping variables '''
    cdef unsigned long p,i,j,n,r
    cdef unsigned long position = 0
    # Whether the last found point was a leading edge of pixels
    cdef bint on_obj = False

    ''' Dimension and point variables '''
    cdef unsigned long ndim = len(shape)
    cdef unsigned long positions
    cdef unsigned long dim
    cdef unsigned int [:] coords = np.zeros(ndim-1,dtype=np.uint32)
    # The number of edges discovered
    cdef unsigned long count

    ''' Stride information '''
    cdef unsigned int [:] strides = np.ones(ndim,dtype=np.uint32)
    for i in range(1,ndim):
        for j in range(ndim-i):
            strides[j] = strides[j]*np.uint32(shape[ndim - i])
    cdef Py_ssize_t last_stride = shape[ndim-1]
    positions = strides[0]*shape[0]//last_stride
        
    '''
    
    Loop through all points and find the start and stopping edges of
    consecutive nonzero values. The way this works is that pixels are laid
    out linearly in memory according to the last dimension of the matrix.
    So, in a 2-d matrix pixels in the same row are immediately next to
    each other in memory. In a 3-d matrix, pixels in the z-dimension are
    next to each other in memory. Since the input is required to be 8-bit
    data (boolean or uint8), then mapping data directly to a vector allows
    8x more data to be processed per clock cycle. Otherwise each clock
    cycle is used to process one 8-bit piece of data.

    The outtermost loop is a linear index of the upper dimensions. So, if
    the matrix is 3-d with dimensions (128x128x64), then there are 128x128
    positions to evaluate. Each position is starting point for a new line
    of pixels along the last dimension.
    
    '''
    for p in range(positions):

        # Calculate the upper dimension values
        dim = position
        for i in range(ndim-1):
            dim = dim//strides[i]
            coords[i] = dim
            dim = position - dim*strides[i]

        # The following code ignores the first pixel, so check to see if
        # the first pixel has a value.
        if image[position] > 0:
            on_obj=True
            for i in range(ndim-1):
                output.push_back(coords[i])
            output.push_back(0)
        
        '''
        
        The following loop is designed for speed. It analyzes 15 pixels at
        a time, and if all 15 pixels have the same value then it quickly
        escapes to the next iteration of the loop. It stops loops when it
        gets less than 15 pixels from the end of the line of pixels so that
        it doesn't run into the next line of pixels.
        
        '''
        r = 0 # manualy register the pixel index, can be optimized with modulo outside the loop
        # print('Starting fast loop position: {}'.format(position))
        for n in range(0,last_stride-15,15):
            # Load 16 pixels and compare against the neighbors (technically only checks 15 pixels)
            v      = _mm_loadu_si128(<__m128i*>&image[position])   # Load 16 pixels.
            mask   = _mm_cmpgt_epi8(v,_mm_setzero_si128())         # Generate 16x8 bit mask.
            mask   = _mm_xor_si128(mask,_mm_slli_si128(mask,1))    # Shift bits to find leading and trailing edges

            # advance the memory position
            position += 15
            r += 15

            # If no edges are found, advance loop
            if (_mm_movemask_epi8(mask)&detect_edge)==0:
                # advance the vector tracking the row index
                i_vec = _mm256_add_epi16(i_vec,cnst15)
                continue

            # Determine how many edges were found
            mask   = _mm_srli_epi64(mask,4)                        # Pack 16x8 bit mask to 32x4 bit mask.
            mask   = _mm_shuffle_epi8(mask,pshufbcnst)             # Align the 16x4 bit mask.
            ind    = _mm_cvtsi128_si64(mask)                       # Extract the 16x4 bit mask.
            ind    = ind & clear_lowest                            # Clear the lowest 4 bit address
            count  = _mm_popcnt_u64(ind)>>2                        # Number of edges detected
            ind    = _pext_u64(cast_index,ind)                     # Get 1-15 index offset of edges

            # The next 5 intrinsics unpack the 4 bit values
            edges    = _mm256_set1_epi64x(ind)
            edges    = _mm256_srlv_epi64(edges,shft)
            edges    = _mm256_and_si256(edges,vmsk)
            edges    = _mm256_shuffle_epi8(edges,shf)
            edges    = _mm256_add_epi16(i_vec,edges)

            # store the results into a vector with appropriate data type for accessibility
            _mm256_storeu_si256(<__m256i*>&temp[0],edges)

            # store the run length encoded values
            for i in range(count):
                if not on_obj:
                    on_obj=True
                    for j in range(ndim-1):
                        output.push_back(coords[j])
                    output.push_back(temp[i])
                else:
                    on_obj=False
                    output.push_back(temp[i])
            
            # Advance the vector index
            i_vec      = _mm256_add_epi16(i_vec,cnst15)

        # Evaluate the final n number of pixels, where n<15
        r += 1
        position += 1
        for n in range(r,last_stride):
            # print('Starting slow loop position: {}'.format(position))
            if image[position] == image[position-1]:
                position += 1
                continue
            if on_obj:
                on_obj = False
                output.push_back(n)
            else:
                on_obj=True
                dim = position
                for i in range(ndim-1):
                    output.push_back(coords[i])
                output.push_back(n)
            position += 1

        # If the very end of the line is reached and no final edge was detected
        if on_obj:
            on_obj = False
            output.push_back(last_stride)

        # Reset the vector index
        i_vec = _mm256_set1_epi16(0)

    # Turn the uint16 vector into a numpy.ndarray of appropriate size
    row_objects = np.asarray(output,dtype=np.uint32).reshape(-1,ndim+1)
    
    return row_objects

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef np.ndarray run_length_encode_32(unsigned char [:] image,tuple shape):
    """run_length_encode Create run length encoded pixels

    This function works identically to run_length_encode_16, except it
    encodes data where one of the image dimensions is larger than 2**16.
    The reason this function is needed is that the index unpacking is
    slightly different.

    For more details, see run_length_encode_16

    Args:
        image (numpy.ndarray): An n-dimensional image reshaped to a linear
            array of pixels.
        shape (tuple): The shape of the image
        
    Outputs:
        numpy.ndarray: Indices of pixel objects
    """

    ''' Set up AVX vectors '''
    # Vector to hold pixel values
    cdef __m128i v
    # Vector to hold 
    cdef __m128i mask
    # Vector for pixel indices
    cdef __m256i edges,edges_shf
    # Byte shuffling constants
    cdef __m128i pshufbcnst = _mm_set_epi8(0x80,  0x80,  0x80,  0x80,
                                           0x80,  0x80,  0x80,  0x80,
                                           0x0E,  0x0C,  0x0A,  0x08,
                                           0x06,  0x04,  0x02,  0x00)
    cdef __m256i shf_lo     = _mm256_set_epi8(0x80,0x80,0x80,0x0B,  0x80,0x80,0x80,0x03,
                                              0x80,0x80,0x80,0x0A,  0x80,0x80,0x80,0x02,
                                              0x80,0x80,0x80,0x09,  0x80,0x80,0x80,0x01,
                                              0x80,0x80,0x80,0x08,  0x80,0x80,0x80,0x00)
    cdef __m256i shf_hi     = _mm256_set_epi8(0x80,0x80,0x80,0x0F,  0x80,0x80,0x80,0x07,
                                              0x80,0x80,0x80,0x0E,  0x80,0x80,0x80,0x06,
                                              0x80,0x80,0x80,0x0D,  0x80,0x80,0x80,0x05,
                                              0x80,0x80,0x80,0x0C,  0x80,0x80,0x80,0x04)
    # Bit shifts to decode 4-bit packing
    cdef __m256i shft       = _mm256_set_epi64x(0x04,0x00,0x04,0x00)
    # 4-bit mask
    cdef __m256i vmsk       = _mm256_set1_epi8(0x0F)
    # Indexing
    cdef __m256i cnst15     = _mm256_set1_epi32(0x000F)
    cdef __m256i i_vec      = _mm256_set1_epi32(0)

    ''' Constants for bit operations '''
    # Casts 4-bit boolean to position index
    cdef unsigned long long cast_index = 0xFEDCBA9876543210
    # Used to determine if pixel runs were detected
    cdef unsigned int detect_edge = 0xFFFFFFFE
    # Clear the lowest value
    cdef unsigned long long clear_lowest = 0xFFFFFFFFFFFFFFF0
    # Holds (16) 4-bit indices
    cdef unsigned long long ind

    ''' Temporary and output vectors '''
    cdef vector[np.uint32_t] temp
    temp.resize(16,0)
    cdef vector[np.uint32_t] output
    
    ''' Looping variables '''
    cdef unsigned long long p,i,j,n,r
    cdef unsigned long long position = 0
    # Whether the last found point was a leading edge of pixels
    cdef bint on_obj = False

    ''' Dimension and point variables '''
    cdef unsigned long long ndim = len(shape)
    cdef unsigned long long positions
    cdef unsigned long long dim
    cdef unsigned int [:] coords = np.zeros(ndim-1,dtype=np.uint32)
    # The number of edges discovered
    cdef unsigned long long count

    ''' Stride information '''
    cdef unsigned long [:] strides = np.ones(ndim,dtype=np.uint64)
    for i in range(1,ndim):
        for j in range(ndim-i):
            strides[j] = strides[j]*np.uint64(shape[ndim - i])
    cdef Py_ssize_t last_stride = shape[ndim-1]
    positions = strides[0]*shape[0]//last_stride
        
    '''
    
    Loop through all points and find the start and stopping edges of
    consecutive nonzero values. The way this works is that pixels are laid
    out linearly in memory according to the last dimension of the matrix.
    So, in a 2-d matrix pixels in the same row are immediately next to
    each other in memory. In a 3-d matrix, pixels in the z-dimension are
    next to each other in memory. Since the input is required to be 8-bit
    data (boolean or uint8), then mapping data directly to a vector allows
    8x more data to be processed per clock cycle. Otherwise each clock
    cycle is used to process one 8-bit piece of data.

    The outtermost loop is a linear index of the upper dimensions. So, if
    the matrix is 3-d with dimensions (128x128x64), then there are 128x128
    positions to evaluate. Each position is starting point for a new line
    of pixels along the last dimension.
    
    '''
    for p in range(positions):

        # Calculate the upper dimension values
        dim = position
        for i in range(ndim-1):
            dim = dim//strides[i]
            coords[i] = dim
            dim = position - dim*strides[i]

        # The following code ignores the first pixel, so check to see if
        # the first pixel has a value.
        if image[position] > 0:
            on_obj=True
            for i in range(ndim-1):
                output.push_back(coords[i])
            output.push_back(0)
        
        '''
        
        The following loop is designed for speed. It analyzes 15 pixels at
        a time, and if all 15 pixels have the same value then it quickly
        escapes to the next iteration of the loop. It stops loops when it
        gets less than 15 pixels from the end of the line of pixels so that
        it doesn't run into the next line of pixels.
        
        '''
        r = 0 # manualy register the pixel index, can be optimized with modulo outside the loop
        for n in range(0,last_stride-15,15):
            # Load 16 pixels and compare against the neighbors (technically only checks 15 pixels)
            v      = _mm_loadu_si128(<__m128i*>&image[position])   # Load 16 pixels.
            mask   = _mm_cmpgt_epi8(v,_mm_setzero_si128())         # Generate 16x8 bit mask.
            mask   = _mm_xor_si128(mask,_mm_slli_si128(mask,1))    # Shift bits to find leading and trailing edges

            # advance the memory position
            position += 15
            r += 15

            # If no edges are found, advance loop
            if (_mm_movemask_epi8(mask)&detect_edge)==0:
                # advance the vector tracking the row index
                i_vec = _mm256_add_epi32(i_vec,cnst15)
                continue

            # Determine how many edges were found
            mask   = _mm_srli_epi64(mask,4)                        # Pack 16x8 bit mask to 32x4 bit mask.
            mask   = _mm_shuffle_epi8(mask,pshufbcnst)             # Align the 16x4 bit mask.
            ind    = _mm_cvtsi128_si64(mask)                       # Extract the 16x4 bit mask.
            ind    = ind & clear_lowest                            # Clear the lowest 4 bit address
            count  = _mm_popcnt_u64(ind)>>2                        # Number of edges detected
            ind    = _pext_u64(cast_index,ind)                     # Get 1-15 index offset of edges

            # The next 5 intrinsics unpack the first (8) 4-bit values
            edges     = _mm256_set1_epi64x(ind)
            edges     = _mm256_srlv_epi64(edges,shft)
            edges     = _mm256_and_si256(edges,vmsk)
            edges_shf = _mm256_shuffle_epi8(edges,shf_lo)
            edges_shf = _mm256_add_epi32(i_vec,edges_shf)

            # store the results into a vector with appropriate data type for accessibility
            _mm256_storeu_si256(<__m256i*>&temp[0],edges_shf)

            # If there are more than 8 values, unpack the rest
            if count > 8:
                    edges_shf = _mm256_shuffle_epi8(edges,shf_hi)
                    edges_shf = _mm256_add_epi32(i_vec,edges_shf)
                    _mm256_storeu_si256(<__m256i*>&temp[8],edges_shf)

            # store the run length encoded values
            for i in range(count):
                if not on_obj:
                    on_obj=True
                    for j in range(ndim-1):
                        output.push_back(coords[j])
                    output.push_back(temp[i])
                else:
                    on_obj=False
                    output.push_back(temp[i])

            
            # Advance the vector index
            i_vec      = _mm256_add_epi32(i_vec,cnst15)

        # Evaluate the final n number of pixels, where n<15
        r += 1
        position += 1
        for n in range(r,last_stride):
            if image[position] == image[position-1]:
                position += 1
                continue
            if on_obj:
                on_obj = False
                output.push_back(n)
            else:
                on_obj=True
                dim = position
                for i in range(ndim-1):
                    output.push_back(coords[i])
                output.push_back(n)
            position += 1

        # If the very end of the line is reached and no final edge was detected
        if on_obj:
            on_obj = False
            output.push_back(last_stride)

        # Reset the vector index
        i_vec = _mm256_set1_epi16(0)

    # Turn the uint16 vector into a numpy.ndarray of appropriate size
    row_objects = np.asarray(output,dtype=np.uint32).reshape(-1,ndim+1)
    
    return row_objects

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef rle_index(tuple image_shape,
               np.ndarray[np.uint32_t, ndim=2] rle_objects):
    """ rle_index Determine indices where higher dimension values change

    This method identifies the rows of rle_objects that contain a change in
    the superficial dimensions of an n-dimensional image. It returns two
    numpy matrices, one that indicates all the superficial dimensions that
    must be evaluated (rle_sparse) and another that indicates the row in
    rle_objects that each point begins.

    Args:
        shape (tuple): The shape of the image
        rle_objects (numpy.ndarray): An n-dimensional image reshaped to a linear
            array of pixels.
        
    Outputs:
        numpy.ndarray: Indices of pixel objects
    """
    
    # Get indices of lower dimension transitions
    cdef Py_ssize_t shape0 = rle_objects.shape[0]
    cdef Py_ssize_t shape1 = rle_objects.shape[1]
    cdef np.ndarray ld_change = np.argwhere(np.any((rle_objects[1:,:shape1-2] - rle_objects[:shape0-1,:shape1-2]) != 0,axis=1)) + 1
    cdef Py_ssize_t ld_shape0 = ld_change.shape[0]
    cdef Py_ssize_t ld_shape1 = ld_change.shape[1]
    
    ld_change = np.vstack((np.array(0,dtype=np),
                           ld_change,
                           np.array(rle_objects.shape[0]))).astype(np.int)
    
    # Initialize the index matrix
    shape = 2
    for i in range(len(image_shape)-1):
        shape *= (image_shape[i] + 2)
    cdef np.ndarray rle_indices = np.full(shape,np.iinfo(np.uint64).max,dtype=np.uint64)
    
    # Assign values to the index matrix
    cdef np.ndarray rle_sparse = np.zeros(ld_change.shape[0]-1,dtype=np.uint32)
    for i in range(rle_objects.shape[1]-3):
        rle_sparse = rle_sparse + rle_objects[ld_change[:ld_change.shape[0]-1],i].squeeze() + 1
        rle_sparse = rle_sparse * (image_shape[i+1] + 2)
    rle_sparse += rle_objects[ld_change[:ld_change.shape[0]-1],shape1-3].squeeze() + 1
    rle_sparse *= 2
    
    # Set the indices
    rle_indices[rle_sparse] = ld_change[:ld_change.shape[0]-1].squeeze()
    rle_indices[rle_sparse + 1] = ld_change[1:].squeeze()
    
    return rle_sparse,rle_indices

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compare_objects(unsigned long [:] range1,
                          unsigned long [:] range2,
                          unsigned int [:,:] rle_objects,
                          unsigned long [:] labels) nogil:
    """ compare_objects Finds and relabels overlapping objects

    This function looks for run length encoded objects that overlap
    and relabels overlapping objects with the smallest index. The
    range1 and range2 variables indicate the linear indices (min and
    max) of the rle_objects to compare.

    Args:
        range1 (np.ndarray): First and last index of objects
        range2 (np.ndarray): First and last index of objects
        rle_objects (np.ndarray): 2d array of run length encoded objects
        labels (np.ndarray): 1d array of labels for each object
    """
    cdef unsigned long long current_row = range1[0]
    
    cdef unsigned long long next_row = range2[0]

    cdef unsigned long long ind_start = rle_objects.shape[1] - 2

    cdef unsigned long long ind_end = rle_objects.shape[1] - 1
    
    # Loop through all row objects in the current and next rows
    while current_row < range1[1] and next_row < range2[1]:
        # if the current objects do not overlap, move to the next one
        if rle_objects[current_row,ind_end] < rle_objects[next_row,ind_start]:
            current_row += 1
            continue
        elif rle_objects[current_row,ind_start] > rle_objects[next_row,ind_end]:
            next_row += 1
            continue
        
        # relabel the overlapping object in the next row
        if labels[labels[labels[next_row]]] < labels[labels[labels[current_row]]]:
            labels[labels[labels[current_row]]] = labels[labels[labels[next_row]]]
            labels[current_row] = labels[labels[labels[next_row]]]
        else:
            labels[labels[labels[next_row]]] = labels[labels[labels[current_row]]]
            labels[next_row] = labels[labels[current_row]]
        next_row += 1
        
        # relabel additional objects in the next row
        while next_row < range2[1] and rle_objects[current_row,ind_end] >= rle_objects[next_row,ind_start]:
            if labels[labels[labels[next_row]]] < labels[labels[labels[current_row]]]:
                labels[labels[labels[current_row]]] = labels[labels[labels[next_row]]]
                labels[current_row] = labels[labels[labels[next_row]]]
            else:
                labels[labels[labels[next_row]]] = labels[labels[labels[current_row]]]
                labels[next_row] = labels[labels[labels[current_row]]]
            next_row += 1
            
        # relabel collisions
        while current_row+1 < range1[1] and rle_objects[current_row+1,ind_start] <= rle_objects[next_row-1,ind_end]:
            current_row += 1
            if labels[labels[labels[next_row-1]]] < labels[labels[labels[current_row]]]:
                labels[labels[labels[current_row]]] = labels[labels[labels[next_row-1]]]
                labels[current_row] = labels[labels[labels[next_row-1]]]
            else:
                labels[labels[labels[next_row-1]]] = labels[labels[labels[current_row]]]
                labels[next_row-1] = labels[labels[labels[current_row]]]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void reconcile_labels(uint_ind labels) nogil:
    """ reconcile_labels Realign labels for rle objects

    As rle objects are compared, they are assigned the smallest
    label of the two objects that overlap. While objects are
    being relabeled, only the root label is generally changed,
    reducing the number of the memory calls by not relabeling all
    rle objects when a new label is assigned. At the end of a 
    comparison in a given dimension, the labels need to be
    re-assigned based on their root label, where the root label
    is the label that is an index to itself.

    Args:
        labels (np.ndarray): 1d array of labels for each object
    
    """
    
    cdef Py_ssize_t i
    cdef Py_ssize_t size = labels.shape[0]

    for i in range(size):
        while labels[i] != labels[labels[i]]:
            labels[i] = labels[labels[i]]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray generate_output_8(unsigned int [:,:] rle_objects,
                                  unsigned long [:] labels,
                                  tuple image_shape):
    """ generate_output_8 Generate 8-bit output image

    This method takes an rle compressed set of data and generates
    an image with up to 255 unique labels.

    Args:
        rle_objects (np.ndarray): 2d array containg rle indices
        labels (np.ndarray): 1d array of labels for each object
        image_shape (tuple): Shape of the output image

    Outputs:
        np.ndarray: 8-bit labeled image
    
    """
    
    # Initialize iteration counter
    cdef long long i

    # Initialize the output
    cdef np.ndarray label_image = np.zeros(image_shape,dtype=np.uint8)
    cdef unsigned char [:] linear_image = label_image.reshape(-1)

    # Get the indices
    cdef long long ndims = len(image_shape) - 1
    cdef long long obj_start = rle_objects.shape[1] - 2
    cdef long long obj_end = rle_objects.shape[1] - 1
    cdef np.ndarray start_ind = np.zeros(rle_objects.shape[0],dtype=np.uint64)
    
    for i in range(ndims):
        start_ind += rle_objects[:,i] 
        start_ind *= image_shape[i+1]
    start_ind += rle_objects[:,obj_start]

    cdef unsigned long [:] start_ind_memview = start_ind
    for i in range(labels.shape[0]):
        fill_n(&linear_image[start_ind_memview[i]],
                rle_objects[i,obj_end] - rle_objects[i,obj_start],
                labels[i])
    
    return label_image

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray generate_output_16(unsigned int [:,:] rle_objects,
                                  unsigned long [:] labels,
                                  tuple image_shape):
    """ generate_output_16 Generate 16-bit output image

    This method takes an rle compressed set of data and generates
    an image with up to 2**16-1 unique labels.

    Args:
        rle_objects (np.ndarray): 2d array containg rle indices
        labels (np.ndarray): 1d array of labels for each object
        image_shape (tuple): Shape of the output image

    Outputs:
        np.ndarray: 16-bit labeled image
    
    """
    
    # Initialize iteration counter
    cdef long i

    # Initialize the output
    # print('Initializing the output...')
    cdef np.ndarray label_image = np.zeros(image_shape,dtype=np.uint16)
    cdef unsigned short [:] linear_image = label_image.reshape(-1)
    # print('label_image.shape={}'.format(image_shape))

    # Get the indices
    # print('Initializing the indices...')
    cdef long long ndims = len(image_shape) - 1
    cdef long long obj_start = rle_objects.shape[1] - 2
    cdef long long obj_end = rle_objects.shape[1] - 1
    cdef np.ndarray start_ind = np.zeros(rle_objects.shape[0],dtype=np.uint64)
    
    # print('Calculating the indices...')
    for i in range(ndims):
        start_ind += rle_objects[:,i] 
        start_ind *= image_shape[i+1]
    start_ind += rle_objects[:,obj_start]
    # print(start_ind[:10])

    # print('Generating the output...')
    cdef unsigned long [:] start_ind_memview = start_ind
    for i in range(labels.shape[0]):
        # print('i={}'.format(i))
        # print('start_ind_memview={}'.format(start_ind_memview[i]))
        # print('size={}'.format(rle_objects[i,obj_end] - rle_objects[i,obj_start]))
        # print('value={}'.format(labels[i]))
        fill_n(&linear_image[start_ind_memview[i]],
                rle_objects[i,obj_end] - rle_objects[i,obj_start],
                labels[i])
    
    return label_image

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray generate_output_32(unsigned int [:,:] rle_objects,
                                   unsigned long [:] labels,
                                   tuple image_shape):
    """ generate_output_32 Generate 32-bit output image

    This method takes an rle compressed set of data and generates
    an image with up to 2**32-1 unique labels.

    Args:
        rle_objects (np.ndarray): 2d array containg rle indices
        labels (np.ndarray): 1d array of labels for each object
        image_shape (tuple): Shape of the output image

    Outputs:
        np.ndarray: 32-bit labeled image
    
    """
    
    # Initialize iteration counter
    cdef long long i

    # Initialize the output
    cdef np.ndarray label_image = np.zeros(image_shape,dtype=np.uint32)
    cdef unsigned int [:] linear_image = label_image.reshape(-1)

    # Get the indices
    cdef long long ndims = len(image_shape) - 1
    cdef long long obj_start = rle_objects.shape[1] - 2
    cdef long long obj_end = rle_objects.shape[1] - 1
    cdef np.ndarray start_ind = np.zeros(rle_objects.shape[0],dtype=np.uint64)
    
    for i in range(ndims):
        start_ind += rle_objects[:,i] 
        start_ind *= image_shape[i+1]
    start_ind += rle_objects[:,obj_start]

    cdef unsigned long [:] start_ind_memview = start_ind
    for i in range(labels.shape[0]):
        fill_n(&linear_image[start_ind_memview[i]],
                rle_objects[i,obj_end] - rle_objects[i,obj_start],
                labels[i])
    
    return label_image

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef unsigned int human_labels(unsigned long [:] labels) nogil:
    """ human_labels Assign labels for humans

    The labels generated by the algorithm are not necessarily
    sequential. For example, if an image has 10 labels, the actual
    labels that have been assigned may be any 10 positive integers.
    This function reassigns labels so that if there are 10 labels,
    then the actual labels are 1 to 10.

    Args:
        labels (np.ndarray): Labels for an image

    Outputs:
        np.uint32: Number of labels
    
    """
    cdef Py_ssize_t size = labels.shape[0]
    cdef unsigned int num = 0
    cdef long long i
    cdef long long ind
    for i in range(size):
        if labels[i]==i:
            num += 1
            labels[i] = num
            continue
        labels[i] = labels[labels[i]]
    
    return num

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray label(unsigned char [:] image,
                      tuple shape,
                      long long connectivity):

    cpdef long long i,d,index,num_points
    cpdef unsigned long size = image.size
    cpdef unsigned long dim_max = np.max(shape)

    if dim_max < 2**16-1:
        rle_objects = run_length_encode_16(image,shape)
    else:
        rle_objects = run_length_encode_32(image,shape)
        
    # Get evaluation coordinates
    ndims = rle_objects.shape[1] - 2
    if ndims == 1:
        offsets = np.ones((1,1),dtype=np.uint8)
    else:
        ind_mat = np.zeros(ndims*(3,))
        ind_d = ind_mat
        for i in range(ndims):
            ind_d[2] = 1
            ind_d = ind_d[1]
        offsets = np.argwhere(ind_mat>0) - 1
        offsets = offsets[np.argwhere(np.sum(np.absolute(offsets),axis=1)<=connectivity).squeeze(),:]
    
    # Adjust pixel coordinates to account for connectivity
    rle_objects_less_one = rle_objects.copy()
    rle_objects_less_one[...,rle_objects.shape[1]-1] -= 1
    rle_objects_mats = []

    for i in range(offsets.shape[0]):
        if connectivity - np.sum(np.absolute(offsets[i])) < 1:
            rle_objects_mats.append(rle_objects_less_one)
        else:
            rle_objects_mats.append(rle_objects)
        
    # Get indices of higher coordinate changes
    rle_sparse,rle_indices = rle_index(shape,rle_objects)
    cdef unsigned long [:] rle_indices_memview = rle_indices
    cdef unsigned int [:] rle_sparse_memview = rle_sparse
    num_points = rle_sparse.shape[0] - 1
    
    # Initalize the output
    cdef unsigned long[:] labels = np.arange(rle_objects.shape[0],dtype=np.uint64)
    
    # null value
    cdef unsigned long null_val = np.iinfo(np.uint64).max
    cdef unsigned long [:] offset_index,current_index
    cdef unsigned int [:,:] rle_objects_mat
    cdef unsigned int [:] rle_sparse_offset_memview 

    # Loop over the dimensions
    compare_time = 0
    reconcile_time = 0
    for d in range(offsets.shape[0]):
    
        rle_sparse_offset = rle_sparse.copy()
        o = 0
        for i in range(offsets.shape[1]-1):
            o += offsets[d][i]
            o *= (shape[i+1] + 2)
        rle_sparse_offset += 2*(offsets[d][offsets.shape[1]-1] + o)
        rle_sparse_offset_memview = rle_sparse_offset

        rle_objects_mat = rle_objects_mats[d]
        
        # Loop over points
        for index in range(num_points):
            offset_index = rle_indices_memview[rle_sparse_offset_memview[index]:rle_sparse_offset_memview[index+1]]
            current_index = rle_indices_memview[rle_sparse_memview[index]:rle_sparse_memview[index+1]]

            if offset_index[0] == null_val:
                continue
                
            compare_objects(current_index,
                            offset_index,
                            rle_objects_mat,
                            labels)
        
        # Reconcile object labels after each offset is analyzed
        reconcile_labels(labels)
    
    # Make labels for humans
    num_objects = human_labels(labels)
    
    # Generate the output with smallest data type
    if num_objects < 2**8-1:
        label_image = generate_output_8(rle_objects,labels,shape)
    elif num_objects < 2**16-1:
        label_image = generate_output_16(rle_objects,labels,shape)
    else:
        label_image = generate_output_32(rle_objects,labels,shape)
    
    return label_image

def label_nd(image,connectivity):
    if connectivity == None:
        connectivity = image.ndim
    
    # Error checking
    assert connectivity<=image.ndim,\
        "connectivity must be less than or equal to the number of image dimensions"
    return label(image.reshape(-1),image.shape,connectivity)