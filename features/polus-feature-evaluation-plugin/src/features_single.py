import numpy as np
from sklearn.metrics import mean_absolute_error
import scipy.stats
import pandas as pd
import cv2
from scipy.spatial import distance
import pyarrow.feather as feather
from metrics import evaluate_all
from pathlib import Path
import logging 
import filepattern
import math

logger = logging.getLogger("evaluation")
logger.setLevel(logging.INFO)

def comparison(expected_array, actual_array, combine, bin_count):
	'''Calculate the metrics for predicted and ground truth histograms
	Args:
	   expected_array: numpy array of original values
	   actual_array: numpy array of predicted values
	Returns:
	   All metrics
	'''

	count1, bins_count1 = np.histogram(expected_array, bins=bin_count)
	pdf1 = count1 / sum(count1)
	cdf1 = np.cumsum(pdf1)

	for i in range(0, len(actual_array)):
		if actual_array[i] < expected_array.min():
			actual_array[i] = expected_array.min()
		if actual_array[i] > expected_array.max():
			actual_array[i] = expected_array.max()

	count2, bins_count2 = np.histogram(actual_array, bins=bin_count)
	pdf2 = count2 / sum(count2)
	cdf2 = np.cumsum(pdf2)

	expected_percents = pdf1
	actual_percents = pdf2

	### PDF input
	def sub_psi(e_perc, a_perc):
		if a_perc == 0:
		    a_perc = 0.0001
		if e_perc == 0:
		    e_perc = 0.0001

		value = (e_perc - a_perc) * np.log(e_perc / a_perc)
		return(value)


	def sub_kld(e_perc, a_perc):
		if a_perc == 0:
		    a_perc = 0.0001
		if e_perc == 0:
		    e_perc = 0.0001

		value = (e_perc) * np.log(e_perc / a_perc)
		return(value)

	def sub_jsd(expected_percents, actual_percents):
		p = np.array(expected_percents)
		q = np.array(actual_percents)
		m = (p + q) / 2
		# compute Jensen Shannon Divergence
		divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
		# compute the Jensen Shannon Distance
		value = np.sqrt(divergence)
		return value

	def l1(pdf1, pdf2):
		return np.sum(abs(pdf1 - pdf2))

	def l2(pdf1, pdf2):
		return np.sqrt(sum((pdf1-pdf2)**2))

	def linfinity(pdf1, pdf2):
		return np.max(abs(pdf1-pdf2))

	def hist_intersect(pdf1, pdf2):
		pdf1 = pdf1.astype(np.float32)
		pdf2 = pdf2.astype(np.float32)
		return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_INTERSECT)

	def cosine_d(pdf1, pdf2):
		return distance.cosine(pdf1, pdf2)

	def canberra(pdf1, pdf2):
		return distance.canberra(pdf1, pdf2)

	def correlation(pdf1, pdf2):
		pdf1 = pdf1.astype(np.float32)
		pdf2 = pdf2.astype(np.float32)
		return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_CORREL)

	def chi_square(pdf1, pdf2):
		pdf1 = pdf1.astype(np.float32)
		pdf2 = pdf2.astype(np.float32)
		return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_CHISQR)

	def bhattacharya(pdf1, pdf2):
		pdf1 = pdf1.astype(np.float32)
		pdf2 = pdf2.astype(np.float32)
		return cv2.compareHist(pdf1, pdf2, cv2.HISTCMP_BHATTACHARYYA)

	###### CDF input

	def ks_divergence(cdf1, cdf2):
		return np.max(abs(cdf1 - cdf2))

	def match(cdf1, cdf2):
		return np.sum(abs(cdf1 - cdf2))

	def cvm(cdf1, cdf2):
		return np.sum((cdf1-cdf2)**2)

	def ws_d(cdf1, cdf2):
		return scipy.stats.wasserstein_distance(cdf1, cdf2)


	###### INCORRECT DEFINITION - IGNORE THIS METRIC ########
	def ks_test(cdf1, cdf2):
		return scipy.stats.ks_2samp(cdf1, cdf2)

	### metrics that take pdf input
	psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

	kld_value = np.sum(sub_kld(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

	jsd_value = sub_jsd(expected_percents, actual_percents)

	errors = evaluate_all(expected_percents, actual_percents)

	### metrics that take cdf input

	wd_value = ws_d(cdf1, cdf2)

	p_value = ks_test(cdf1, cdf2)  ### Ignore this value.

	return(hist_intersect(pdf1, pdf2),correlation(pdf1, pdf2), chi_square(pdf1, pdf2), bhattacharya(pdf1, pdf2), \
		l1(pdf1, pdf2),l2(pdf1, pdf2), linfinity(pdf1, pdf2),cosine_d(pdf1, pdf2),canberra(pdf1, pdf2),ks_divergence(cdf1, cdf2),match(cdf1, cdf2),cvm(cdf1, cdf2),\
		psi_value, kld_value, jsd_value, wd_value, p_value, errors)


def runMain(gt, pred, outFileFormat, combineLabels, filePattern, singleCSV, outDir):
	
	fp = filepattern.FilePattern(gt,filePattern)

	if singleCSV:
		lst = []
	
	header = ['Image','features','histogram intersection','correlation', 'chi square', 'bhattacharya distance','L1 score','L2 score', 'L infinity score','cosine distance','canberra distance','ks divergence','match distance','cvm distance',\
				'psi value','kl divergence','js divergence', 'wasserstein distance', 'p-value', 'Mean square error', 'Root mean square error', 'Normalized Root Mean Squared Error', 'Mean Error', 'Mean Absolute Error',\
				'Geometric Mean Absolute Error', 'Median Absolute Error', 'Mean Percentage Error', 'Mean Absolute Percentage Error', 'Median Absolute Percentage Error', 'Symmetric Mean Absolute Percentage Error', 'Symmetric Median Absolute Percentage Error',\
				'Mean Arctangent Absolute Percentage Error', 'Normalized Absolute Error', 'Normalized Absolute Percentage Error', 'Root Mean Squared Percentage Error', 'Root Median Squared Percentage Error', 'Integral Normalized Root Squared Error',\
				'Root Relative Squared Error', 'Relative Absolute Error (aka Approximation Error)', 'Mean Directional Accuracy']
	
	for fP in fp():
		for PATH in fP:
			file_name = PATH.get("file")
			df_gt = pd.read_csv(file_name)

			from os.path import exists
			file_exists = exists(Path(str(pred+"/"+file_name.name)))
			if not file_exists:
				continue
			
			df_pred = pd.read_csv(str(pred+"/"+file_name.name))

			if not singleCSV:
				lst = []

			for feature in df_gt.columns[1:]:
				z_gt = np.array(df_gt[feature])
				if feature not in df_pred.columns:
					continue
				z_pred = np.array(df_pred[feature])
				
				if feature not in ["intensity_image", "label", "touching_border"] and z_pred.size > 3 and z_gt.size > 3:
					z_gt = np.array(z_gt, dtype=float)
					z_pred = np.array(z_pred, dtype=float)
					z_gt = z_gt[~np.isnan(z_gt)]
					z_pred = z_pred[~np.isnan(z_pred)]
					if z_pred.size > 3 and z_gt.size > 3:
						logger.info('evaluating feature {} for {}'.format(feature, file_name.name))

						expected_array = z_gt 
						actual_array = z_pred

						if combineLabels:
							combined = np.concatenate((actual_array, expected_array))
							q1 = np.quantile(combined, 0.25)
							q3 = np.quantile(combined, 0.75)
							iqr = q3 - q1
							bin_width = (2 * iqr) / (len(combined) ** (1 / 3))
							if bin_width == float(0.0) or np.isnan(bin_width):
								continue
							bin_count = np.ceil((combined.max() - combined.min()) / (bin_width))
						else:
							q1 = np.quantile(expected_array, 0.25)
							q3 = np.quantile(expected_array, 0.75)
							iqr = q3 - q1
							bin_width = (2 * iqr) / (len(expected_array) ** (1 / 3))
							if bin_width == float(0.0) or np.isnan(bin_width):
								continue
							bin_count = np.ceil((expected_array.max() - expected_array.min()) / (bin_width))
						if bin_count > 2**16 or np.isnan(bin_count) or bin_count == 0:
							continue
						else:
							bin_count = int(bin_count)

						hist_intersect,correlation, chi_square, bhattacharya,l1,l2, linfinity,cosine_d,canberra,ks_divergence,match,cvm,\
						psi_value, kld_value, jsd_value, wd_value, p_value, errors = comparison(z_gt, z_pred, combineLabels, bin_count)

						data_result = [file_name.name, feature, hist_intersect,correlation, chi_square, bhattacharya,l1,l2, linfinity,cosine_d,canberra,ks_divergence,match,cvm,\
						psi_value, kld_value, jsd_value, wd_value, p_value[1], errors.get('mse'),errors.get('rmse'), errors.get('nrmse'), errors.get('me'), errors.get('mae'),\
						errors.get('gmae'), errors.get('mdae'), errors.get('mpe'), errors.get('mape'), errors.get('mdape'), errors.get('smape'), errors.get('smdape'),\
						errors.get('maape'), errors.get('std_ae'), errors.get('std_ape') , errors.get('rmspe'), errors.get('rmdspe'), errors.get('inrse'),\
						errors.get('rrse'), errors.get('rae'),errors.get('mda')]

						lst.append(data_result)

			if not singleCSV:
				df = pd.DataFrame(lst,columns=header)
				if outFileFormat:
					df.to_csv(str(Path(outDir)/(file_name.name+".csv")))
				else:
					feather.write_feather(df, str(Path(outDir)/(file_name.name+".lz4")))

	if singleCSV:
		df = pd.DataFrame(lst,columns=header)
		if outFileFormat:
			df.to_csv(str(Path(outDir)/("result.csv")))
		else:
			feather.write_feather(df, str(Path(outDir)/("result.lz4")))

	logger.info("evaluation complete.")