import logging
import os, sys
import requests, zipfile, io
import unittest

import numpy as np

import bfio
from bfio import BioReader, LOG4J, JARS
import tempfile

from splinedist.utils import phi_generator, grid_generator
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist import fill_label_holes
from csbdeep.utils import normalize

from sklearn.metrics import jaccard_score

logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("test_splinedist_model")
logger.setLevel(logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

model_basedir = "/home/ec2-user/workdir/splinedist/splinedist_scalability/models/"



def get_jaccard_index(prediction : np.ndarray,
					  ground_truth : np.ndarray):
	""" This function gets the jaccard index between the 
	predicted image and its ground truth.

	Args:
		prediction - the predicted output from trained neural network
		ground_truth - ground truth given by inputs
	Returns:
		jaccard - The jaccard index between the two inputs
					https://en.wikipedia.org/wiki/Jaccard_index
	Raises:
		None
	"""
	imageshape = prediction.shape

	prediction = prediction.ravel()
	ground_truth = ground_truth.ravel()

	prediction[(prediction > 0)] = 1.0
	ground_truth[(ground_truth > 0)] = 1.0

	jaccard = jaccard_score(prediction, ground_truth, average='macro')

	return jaccard

class TestEncodingDecoding(unittest.TestCase):

	def test_model(self):
		# Generate a temporary directory to store the outputs
		with tempfile.TemporaryDirectory() as temp_dir:

			os.chdir(temp_dir)

			hyperlink = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"
			request = requests.get(hyperlink)
			zip = zipfile.ZipFile(io.BytesIO(request.content))
			zip.extractall(temp_dir)
			
			print(os.listdir(temp_dir))
			image_dir = os.path.join(temp_dir, "dsb2018/test/images/")
			label_dir = os.path.join(temp_dir, "dsb2018/test/masks/")

			model_dir_name = '.'
			model_dir_path = os.path.join(model_basedir, model_dir_name)
			assert os.path.exists(model_dir_path), \
				"{} does not exist".format(model_dir_path)

			model = SplineDist2D(None, name=model_dir_name, basedir=model_basedir)
			logger.info("\n Done Loading Model ...")

			# make sure phi and grid exist in current directory, otherwise create.
			logger.info("\n Getting extra files ...")
			conf = model.config
			M = int(conf.n_params/2)

			if not os.path.exists("./phi_{}.npy".format(M)):
				contoursize_max = conf.contoursize_max
				logger.info("Contoursize Max for phi_{}.npy: {}".format(M, contoursize_max))
				phi_generator(M, contoursize_max, '.')
				logger.info("Generated phi")
			if not os.path.exists("./grid_{}.npy".format(M)):
				training_patch_size = conf.train_patch_size
				logger.info("Training Patch Size for grid_{}.npy: {}".format(training_patch_size, M))
				grid_generator(M, training_patch_size, conf.grid, '.')
				logger.info("Generated grid")

			weights_best = os.path.join(model_dir_path, "weights_best.h5")
			model.keras_model.load_weights(weights_best)
			logger.info("\n Done Loading Best Weights ...")

			logger.info("\n Parameters in Config File ...")
			config_dict = model.config.__dict__
			for ky,val in config_dict.items():
				logger.info("{}: {}".format(ky, val))

			M = int(config_dict['n_params']/2)
			X_val = sorted(os.listdir(image_dir))
			Y_val = sorted(os.listdir(label_dir))
			num_images = len(X_val)
			num_labels = len(Y_val)

			assert num_images > 0, "Input Directory is empty"
			assert num_images == num_labels, "The number of images do not match the number of ground truths"

			array_images_tested = []
			array_labels_tested = []

			# Neural network parameters
			axis_norm = (0,1)
			n_channel = 1 # this is based on the input data


			# Read the input images and labels used for testing
			jaccards = []
			for im in range(num_images):

				image = os.path.join(image_dir, X_val[im])
				label = os.path.join(label_dir, Y_val[im])

				with bfio.BioReader(image) as br_image:
					with bfio.BioReader(label) as br_label:

						im_array = br_image[:,:,0:1,0:1,0:1].reshape(br_image.shape[:2]) 
						im_array = normalize(im_array,pmin=1,pmax=99.8,axis=axis_norm)
						im_array, details = model.predict_instances(im_array)
						im_array = np.asarray(im_array, dtype=br_image.dtype)

						lab_array = br_label[:,:,0:1,0:1,0:1].reshape(br_label.shape[:2])
						lab_array = np.asarray(fill_label_holes(lab_array),
											 dtype=br_label.dtype)

						jaccard = get_jaccard_index(im_array, lab_array)
						assert jaccard >= .70
						jaccards.append(jaccard)

			avg_jaccard = sum(jaccards)/num_images
			assert avg_jaccard > .85



if __name__ == '__main__':
    unittest.main()
