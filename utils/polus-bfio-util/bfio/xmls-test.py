from bfio import  BioReader,BioWriter
import bioformats
import javabridge

javabridge.start_vm(class_path=bioformats.JARS)
image_path='/home/sudharsan/Desktop/work/images/r001_z000_y000_x000_c001.ome.tif'

br = BioReader(image_path)
image = br.read_image()
print(br.read_metadata())