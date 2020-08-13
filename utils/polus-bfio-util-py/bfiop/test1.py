# Import javabridge and start the vm
import bioformats
import javabridge
import matplotlib.pyplot as plt
import numpy as np
from bfiop.Bio_Reader import BioReader, BioWriter

import bfio

javabridge.start_vm(class_path=bioformats.JARS)

# Path to bioformats supported image
# image_path = './r001_c001_z000.ome.tif'           # large image
# image_path = './r01_x01_y01_z<01-30>.ome.tif'     # 3d image
#image_path = './r001_z000_y006_x011_c000.ome.tif' # small image
image_path='/home/sudharsan/Desktop/work/images/r001_z000_y000_x000_c001.ome.tif'
out_path = '/home/sudharsan/Desktop/work/images/bfiop.ome.tif'

# Number of times to repeat a process for speed testing
replicates = 1

# Show results of each read, if False will only display results when different
show_results = False

try:
    ''' Compare BioReaders with different backends '''
    # Read once to get log4j warnings out of the way
    br = bfio.BioReader(image_path)
    image = br.read_image()

    print('')
    print('-- Comparing results of reading images with each backend --')

    # java backend read
    print('Reading {} using java backend...'.format(image_path))
    br = bfio.BioReader(image_path)
    image = br.read_image().squeeze()

    # python backend read
    print('Reading {} using Python backend...'.format(image_path))
    brp = BioReader(image_path)
    imagep = brp.read_image().squeeze()

    # check if results are equal
    array_equal = np.array_equal(image,imagep)
    print('Result of bfio and bfiop are equal: {}'.format(np.array_equal(image,imagep)))

    # If results are not equal, show a plot
    if not array_equal or show_results:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(image)
        ax1.set_title('Java Backend Result')
        ax2.imshow(imagep)
        ax2.set_title('Python Backend Result')
        plt.show()

    # ''' Compare speed of loading with each backend '''
    # print('')
    # print('-- Comparing speed of loading an lzw compressed image using each backend (n={}) --'.format(replicates))
    #
    # print('Reading with java backend...')
    # start = time.time()
    # for _ in range(replicates):
    #     br = bfio.BioReader(image_path)
    #     image = br.read_image()
    # print('Time to read with java backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    # print('Reading with python backend...')
    # start = time.time()
    # for _ in range(replicawriting result in an empty imagetes):
    #     brp = BioReader(image_path)
    #     imagep = brp.read_image()
    # print('Time to read with python backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    ''' Determine if saved image is the same as the original '''
    print('')
    print('-- Comparing saved image to original image --')

    #brp = BioReader(out_path)
    #imagep = brp.read_image().squeeze()
    #
    # print('Reading {} using Python backend...'.format(image_path))
    # brp = bfiop.BioReader(image_path)brp = bfiop.BioReader(out_path)
    #imagep = brp.read_image().squeeze()

    print('Reading {} using Python backend...'.format(image_path))
    brp = BioReader(image_path)
    imagep = brp.read_image()

    print('Saving image to file {} using Python backend...'.format(out_path))
    bwp = BioWriter(out_path,metadata=brp.read_metadata())
    bwp.write_image(imagep)
    bwp.close_image()

    print('Reading {} using Python backend...'.format(out_path))
    brpw = BioReader(out_path)

    imagepw = brpw.read_image()

    print('Original and saved images are equal: {}'.format(np.array_equal(imagep,imagepw)))

    # If results are not equal, show a plot
    if not array_equal or show_results:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(imagep)
        ax1.set_title('Original Image')
        ax2.imagep(imagepw)
        ax2.set_title('Saved Image')
        plt.show()
    #
    # ''' Compare speed of loading deflate image with each backend '''
    # print('')
    # print('-- Comparing speed of loading a deflate compressed image using each backend (n={}) --'.format(replicates))
    #
    # print('Reading with java backend...')
    # start = time.time()
    # for _ in range(replicates):
    #     br = bfio.BioReader(out_path)
    #     image = br.read_image()
    # print('Time to read with java backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    # print('Reading with python backend...')
    # start = time.time()
    # for _ in range(replicates):
    #     brp = BioReader(out_path)
    #     imagep = brp.read_image()
    # print('Time to read with python backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    # ''' Load a small section in the center of an image '''
    # print('')
    # print('-- Loading small section of image --'.format(replicates))
    #
    # print('Reading with java backend...')
    # br = bfio.BioReader(image_path)
    # X = [br.num_x()//2-100,br.num_x()//2+100]
    # Y = [br.num_y()//2-100,br.num_y()//2+100]
    # image = br.read_image(X=X,Y=Y)
    #
    # print('Reading with python backend...')
    # brp = BioReader(image_path)
    # X = [brp.num_x()//2-100,br.num_x()//2+100]
    # Y = [brp.num_y()//2-100,br.num_y()//2+100]
    # imagep = br.read_image(X=X,Y=Y)
    #
    # print('Original and saved images are equal: {}'.format(np.array_equal(image,imagep)))
    #
    # # If results are not equal, show a plot
    # if not array_equal or show_results:
    #     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #     ax1.imshow(image)
    #     ax1.set_title('Original Image')
    #     ax2.imagep(imagep)
    #     ax2.set_title('Saved Image')
    #     plt.show()
    # '''BUg: Testing Ome tags in ome xml'''
    # print('Reading {} using Python backend...'.format(image_path))
    # brp = BioReader(image_path)
    # imagep = brp.read_image()
    # metadata_read=brp.read_metadata()
    #
    # print('Saving image to file {} using Python backend...'.format(out_path))
    # bwp = BioWriter(out_path,metadata=brp.read_metadata())
    # bwp.write_image(imagep)
    # bwp.close_image()
    #
    # print('Reading {} using Python backend...'.format(out_path))
    # brpw = BioReader(out_path)
    # imagepw = brpw.read_image()
    # Imagepw_metadata=brpw.read_metadata()
    #
    # print(Imagepw_metadata)
    # print('Original and saved images Metadata are equal: {}'.format((metadata_read == Imagepw_metadata)))
    #


    ''' BUG: Repeatedly saving an image creates a blank image '''
    print('')
    print('-- Testing repeated saves on an image --'.format(replicates))

    brp = BioReader(out_path)
    imagep = brp.read_image().squeeze()

    print('Reading {} using Python backend...'.format(image_path))
    brp = BioReader(image_path)
    imagep = brp.read_image()
    #print(imagep.squeeze().siz)
    print('Saving image to file {} using Python backend...'.format(out_path))
    for _ in range(2):
        bwp = BioWriter(out_path,metadata=brp.read_metadata())
        bwp.write_image(imagep)
        bwp.close_image()
    print('Reading {} using java backend...'.format(image_path))
    br = bfio.BioReader(image_path)
    imagepw = br.read_image()
    # print('Reading {} using Python backend...'.format(out_path))print('Reading {} using Python backend...'.format(out_path))
    brpw = BioReader(out_path)
    imagepw = brpw.read_image()
    print(imagepw.shape)
    # brpw = BioReader(out_path)
    # imagepw = brpw.read_image()
    #test1=imagepw.squeeze()
    #print(test1.shape)
    print('Original and saved images are equal: {}'.format(np.array_equal(imagep,imagepw)))

    # If results are not equal, show a plot
    if not array_equal or  show_results:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(imagep.squeeze())
        ax1.set_title('Original Image')
        ax2.imshow(imagepw.squeeze())
        ax2.set_title('Saved Image')
        plt.show()
    # imagep = brp.read_image()
    #
    # print('Saving image to file {} using Python backend...'.format(out_path))
    # bwp = bfiop.BioWriter(out_path,metadata=brp.read_metadata())
    # bwp.write_image(brp.read_image())
    # bwp.close_image()
    #
    # print('Reading {} using Python backend...'.format(out_path))
    # brpw = bfiop.BioReader(out_path)
    # imagepw = brpw.read_image()
    #
    # print('Original and saved images are equal: {}'.format(np.array_equal(imagep,imagepw)))
    #
    # # If results are not equal, show a plot
    # if not array_equal or show_results:
    #     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #     ax1.imshow(imagep)
    #     ax1.set_title('Original Image')
    #     ax2.imagep(imagepw)
    #     ax2.set_title('Saved Image')
    #     plt.show()
    #
    # ''' Compare speed of loading deflate image with each backend '''
    # print('')
    # print('-- Comparing speed of loading a deflate compressed image using each backend (n={}) --'.format(replicates))
    #
    # print('Reading with java backend...')
    # start = time.time()
    # for _ in range(replicates):
    #     br = bfio.BioReader(out_path)
    #     image = br.read_image()
    # print('Time to read with java backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    # print('Reading with python backend...')
    # start = time.time()
    # for _ in range(replicates):
    #     brp = bfiop.BioReader(out_path)
    #     imagep = brp.read_image()
    # print('Time to read with python backend: {:.4f}s'.format((time.time() - start)/replicates))
    #
    # ''' Load a small section in the center of an image '''
    # print('')
    # print('-- Loading small section of image --'.format(replicates))
    #
    # print('Reading with java backend...')
    # br = bfio.BioReader(image_path)
    # X = [br.num_x()//2-100,br.num_x()//2+100]
    # Y = [br.num_y()//2-100,br.num_y()//2+100]
    # image = br.read_image(X=X,Y=Y)
    #
    # print('Reading with python backend...')
    # brp = bfiop.BioReader(image_path)
    # X = [brp.num_x()//2-100,br.num_x()//2+100]
    # Y = [brp.num_y()//2-100,br.num_y()//2+100]
    # imagep = br.read_image(X=X,Y=Y)
    #
    # print('Original and saved images are equal: {}'.format(np.array_equal(image,imagep)))
    #
    # # If results are not equal, show a plot
    # if not array_equal or show_results:
    #     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #     ax1.imshow(image)brp.read_image()
    #     ax1.set_title('Original Image')
    #     ax2.imagep(imagep)
    #     ax2.set_title('Saved Image')
    #     plt.show()
    #
    # ''' BUG: Repeatedly saving an image creates a blank image '''
    # print('')
    # print('-- Testing repeated saves on an image --'.format(replicates))
    #
    # brp = bfiop.BioReader(out_path)
    # imagep = brp.read_image().squeeze()
    #
    # print('Reading {} using Python backend...'.format(image_path))
    # brp = bfiop.BioReader(image_path)
    # imagep = brp.read_image()
    #
    # print('Saving image to file {} using Python backend...'.format(out_path))
    # for _ in range(2):
    #     bwp = bfiop.BioWriter(out_path,metadata=brp.read_metadata())
    #     bwp.write_image(brp.read_image())
    #     bwp.close_image()
    #
    # print('Reading {} using Python backend...'.format(out_path))
    # brpw = bfiop.BioReader(out_path)
    # imagepw = brpw.read_image()
    #
    # print('Original and saved images are equal: {}'.format(np.array_equal(imagep,imagepw)))
    #
    # # If results are not equal, show a plot
    # if not array_equal or show_results:
    #     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #     ax1.imshow(imagep)
    #     ax1.set_title('Original Image')
    #     ax2.imagep(imagepw)
    #     ax2.set_title('Saved Image')
    #     plt.show()

finally:
    # Done executing program, so kill the vm. If the program needs to be run
    # again, a new interpreter will need to be spawned to start the vm.
    javabridge.kill_vm()