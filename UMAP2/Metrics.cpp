/*
 * This section covers the computation of various distance metrics as implemented in the following link
 * https://github.com/lmcinnes/umap/blob/master/umap/distances.py
 */

#include <string>
#include <iostream>
#include <math.h>
#include <fstream>
#include "KNN_Serial_Code.h"

using namespace std;

/**
 * Estimation to pi() value
 */
double pi() { return atan(1)*4;}
/**
 *  zero approximation in float precision for Metric computations
 */
double epsilon=1e-6;

template<typename T>
double approx_log_Gamma(T x){
	if (x - 1 < epsilon) return 0;
	return x*log(x) - x + 0.5*log(2.0*pi()/x) + 1.0/(x*12.0);
}

double log_beta(double x, double y){
	double a = min(x, y);
	double b = max(x, y);

	if (b < 5){
		double value = -log(b);
		for (int i = 1; i < int(a); ++i) value += log(i) - log(b + i);
		return value;
	}    
	else return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y);
}

double log_single_beta(double x){
	return log(2.0) * (-2.0 * x + 0.5) + 0.5 * log(2.0 * pi() / x) + 0.125 / x;
}

/**
 * computes spatial distance between two points based on the chosen Metric
 * @param distanceMetric the metric to compute the distance between the points in high-D space
 * @param dataPoints represents input dataPoints read from filePath
 * @param *it and *it2 indices of the desired points in input dataset
 * @param Dim is #columns (or features) in input dataset
 * @param distanceV1 is the first optional variable needed for computing distance in some metrics
 * @param distanceV2 is the second optional variable needed for computing distance in some metrics	
 * @param filePathOptionalArray The full path to optional array for the distance metric computation 
 * @param logFile The errors and informational messages are outputted to the log file 
 * @return spatial distance between points two points 
 */
double computeDistance (string distanceKeyword, double** dataPoints, int it, int it2, int Dim, float distanceV1, float distanceV2, string filePathOptionalArray, ofstream & logFile){

	/**
	 * We first focus on computing the distance from a few Metrics that depend on input array from the user  
	 */
	if (filePathOptionalArray !=""){
		/**
		 * Compute the size of input array without the header (i.e.(#Rows in dataset)-1).
		 */
		string cmd="wc -l "+filePathOptionalArray;
		string outputCmd = exec(cmd.c_str());
		int RecordCounts=stoi(outputCmd.substr(0, outputCmd.find(" ")))-1;
		if (RecordCounts != Dim) {
		logFile<<"ALERT: The Optional Vector has different length than the number of features in the input data set"<<endl;
		cout<<"ALERT: The Optional Vector has different length than the number of features in the input data set"<<endl;		
		}
		/**
		 * Compute the dimension of input array (#Columns)
		 */
		int featureCounts;
		string cmd2="head -n 1 "+ filePathOptionalArray + " |tr '\\,' '\\n' |wc -l ";
		featureCounts = stoi(exec(cmd2.c_str())); 

		logFile << "The lenght of the Optional Vector used in Metric computation is "<< RecordCounts <<" and its dimension is " << featureCounts << endl;
		cout << "The lenght of the Optional Vector used in Metric computation is "<< RecordCounts <<" and its dimension is " << featureCounts << endl;
		/**
		 * Make the 2D array containing the input array
		 */
		int** inputArray = new int*[RecordCounts];
		for (int i = 0; i < RecordCounts; ++i) { inputArray[i] = new int[featureCounts]; }

		ifstream infile;
		infile.open(filePathOptionalArray);
		if (infile.fail())
		{
			logFile << "Error in Opening Input File" << endl;
			cout << "Error in Opening Input File" << endl;
			return 0.0;
		}
		/**
		 * Remove the header info
		 */
		string dummyLine;
		getline(infile, dummyLine);
		/**
		 * Reading the Entire array and storing it in inputArray
		 */
		for (int i = 0; i < RecordCounts; ++i) {
			string temp, temp2;
			getline(infile, temp);
			for (int j = 0; j < featureCounts; ++j) {
				temp2 = temp.substr(0, temp.find(","));
				inputArray[i][j] = atof(temp2.c_str());
				temp.erase(0, temp.find(",") + 1);
			}
		}
		/**
		 * Now, we are able to compute the distance
		 */
		if (distanceKeyword =="standardised_euclidean") {
			double result = 0.0, tmp;
			for (int i = 0; i < Dim; ++i) {
				tmp = dataPoints[it][i] - dataPoints[it2][i];
				result += (tmp * tmp)/inputArray[i][0];  
			}
			return sqrt(result);
		}

		else if (distanceKeyword =="weighted_minkowski") {
			double result = 0.0;
			for (int i = 0; i < Dim; ++i) {
				result += pow(inputArray[i][0] * abs(dataPoints[it][i] - dataPoints[it2][i]), distanceV1);  
			}
			return pow(result, 1.0/distanceV1);
		}

		else if (distanceKeyword =="mahalanobis") {
			double result = 0.0, tmp;
			double diff[Dim];

			for (int i = 0; i < Dim; ++i) {
				diff[i] = dataPoints[it][i] - dataPoints[it2][i];
			}
			for (int i = 0; i < Dim; ++i) {
				tmp=0;
				for (int j = 0; j < Dim; ++j) {
					tmp += inputArray[i][j] * diff[j];
				}
				result += tmp * diff[i];
			}
			return sqrt(result);
		}
	}

	/**
	 * All other distance metrics that do not depend on input array from the user will be computed below 
	 */
	if (distanceKeyword =="euclidean") {
		double tmp = 0;
		for (int i = 0; i < Dim; ++i) {
			tmp += pow((dataPoints[it][i] - dataPoints[it2][i]), 2);
		}
		return sqrt(tmp);
	}

	else if (distanceKeyword =="manhattan") {
		double tmp = 0;
		for (int i = 0; i < Dim; ++i) {
			tmp += abs(dataPoints[it][i] - dataPoints[it2][i]);
		}
		return tmp;
	}

	else if (distanceKeyword =="minkowski") {
		double tmp = 0;
		for (int i = 0; i < Dim; ++i) {
			tmp += pow(abs(dataPoints[it][i] - dataPoints[it2][i]), distanceV1);
		}   
		return pow(tmp, (1.0 / distanceV1) );  
	}

	else if (distanceKeyword =="cosine") {
		double result = 0.0, norm_x = 0.0, norm_y = 0.0;
		for (int i = 0; i < Dim; ++i) {
			result += dataPoints[it][i] * dataPoints[it2][i];
			norm_x += dataPoints[it][i] * dataPoints[it][i];
			norm_y += dataPoints[it2][i]* dataPoints[it2][i];
		}
		if (norm_x < epsilon && norm_y < epsilon) return 0.0;
		else if (norm_x < epsilon || norm_y < epsilon) return 1.0;
		else return 1.0 - (result / sqrt(norm_x * norm_y)); 
	}

	else if (distanceKeyword =="correlation") {  
		double mu_x=0.0, norm_x=0.0;
		double mu_y=0.0, norm_y=0.0;
		double  dot_product = 0.0;

		for (int i = 0; i < Dim; ++i) {
			mu_x += dataPoints[it][i];
			mu_y += dataPoints[it2][i];	  
		}
		mu_x /=Dim;
		mu_y /=Dim;

		double shifted_x,shifted_y; 
		for (int i = 0; i < Dim; ++i) {
			shifted_x = dataPoints[it][i] - mu_x;
			shifted_y = dataPoints[it2][i]- mu_y;
			norm_x += shifted_x * shifted_x;
			norm_y += shifted_y * shifted_y;
			dot_product += shifted_x * shifted_y;	  	  	  
		}

		if (norm_x < epsilon && norm_y < epsilon) return 0.0;
		else if (dot_product < epsilon)  return 1.0;
		else  return 1.0 - (dot_product / sqrt(norm_x * norm_y));    
	}

	else if (distanceKeyword =="bray_curtis") {
		double numerator = 0.0, denominator = 0.0;

		for (int i = 0; i < Dim; ++i) {
			numerator += abs(dataPoints[it][i] - dataPoints[it2][i]);
			denominator += abs(dataPoints[it][i] + dataPoints[it2][i]);
		}

		if (denominator > 0.0) return numerator/denominator;
		else return 0.0;
	}

	else if (distanceKeyword =="ll_dirichlet") {
		double n1,n2;
		for (int i = 0; i < Dim; ++i) {
			n1 +=dataPoints[it][i];
			n2 +=dataPoints[it2][i];	
		}
		double log_b = 0.0, self_denom1 = 0.0, self_denom2 = 0.0;

		for (int i = 0; i < Dim; ++i) {
			if (dataPoints[it][i] * dataPoints[it2][i] > 0.9){
				log_b += log_beta(dataPoints[it][i], dataPoints[it2][i]);
				self_denom1 += log_single_beta(dataPoints[it][i]);
				self_denom2 += log_single_beta(dataPoints[it2][i]);
			}
			else {
				if (dataPoints[it][i] > 0.9) self_denom1 += log_single_beta(dataPoints[it][i]);
				if (dataPoints[it2][i] > 0.9) self_denom2 += log_single_beta(dataPoints[it2][i]);	  
			}  	    
		}

		return sqrt(1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
				+ 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1))) );
	}

	else if (distanceKeyword =="jaccard") {
		int x_true, y_true, num_non_zero=0, num_equal=0; 

		for (int i = 0; i < Dim; ++i) {
			if ( dataPoints[it][i] < epsilon) x_true=0;
			else x_true=1;

			if ( dataPoints[it2][i] < epsilon) y_true=0;
			else y_true=1;

			if (x_true==1 || y_true==1) ++num_non_zero;
			if (x_true==1 && y_true==1) ++num_equal;    
		}

		if (num_non_zero == 0) return 0.0;
		else return double (num_non_zero - num_equal) / num_non_zero;
	}

	else if (distanceKeyword =="dice") {
		int num_true_true=0, num_not_equal=0,x_true, y_true;
		for (int i = 0; i < Dim; ++i) {
			if ( dataPoints[it][i] < epsilon) x_true=0;
			else x_true=1;

			if ( dataPoints[it2][i] < epsilon) y_true=0;
			else y_true=1;

			if (x_true==1 && y_true==1) ++num_true_true;
			if (x_true != y_true) ++num_not_equal;    
		}

		if (num_not_equal==0) return 0.0;
		else return double(num_not_equal) / (2.0 * num_true_true + num_not_equal);
	}

	else if (distanceKeyword =="categorical_distance") {
		if (dataPoints[it][0] == dataPoints[it2][0]) return 0.0;
		else return 1.0;
	}

	else if (distanceKeyword =="ordinal_distance") {
		return abs(dataPoints[it][0] - dataPoints[it2][0]) / distanceV1;
	}

	else if (distanceKeyword =="count_distance") {
		double poisson_lambda=distanceV1; //default 1.0
		double normalisation=distanceV2; //default 1.0
		double log_k_factorial;

		double lo=int(min(dataPoints[it][0], dataPoints[it2][0]));
		double hi=int(max(dataPoints[it][0], dataPoints[it2][0]));

		double log_lambda = log(poisson_lambda);

		if (lo < 2) log_k_factorial = 0.0;
		else if (lo < 10) {
			log_k_factorial = 0.0;
			for (int k=2; k<lo; ++k) log_k_factorial += log(k);
		}
		else log_k_factorial = approx_log_Gamma(lo + 1);

		double result = 0.0;
		for (int k = lo; k < hi; ++k) {
			result += k * log_lambda - poisson_lambda - log_k_factorial;
			log_k_factorial += log(k);
		}    
		return result/normalisation;
	}

	else if (distanceKeyword =="levenshtein") {
		float normalisation = distanceV1;  //default 1.0
		int max_distance = int(distanceV2); //default 20
		int x_len=Dim, y_len=Dim;
		float v0[y_len + 1], v1[y_len + 1];

		if (abs(x_len - y_len) > max_distance) return float(abs(x_len - y_len))/normalisation;

		for (int i=0; i<y_len+1; ++i){
			v0[i]=i;
			v1[i]=0;
		}

		int substitution_cost;
		float minVal,deletion_cost,insertion_cost;
		for (int i=0; i<x_len; ++i){
			v1[i] = i + 1;
			for (int j=0; j<y_len; ++j){      
				deletion_cost = v0[j + 1] + 1;
				insertion_cost = v1[j] + 1;

				if (dataPoints[it][i] == dataPoints[it2][i]) substitution_cost=1;
				else  substitution_cost=0;

				v1[j + 1] = min(deletion_cost, min(insertion_cost, float(substitution_cost))  );
			}

			for (int k=0; k<y_len+1; ++k) {
				v0[k]=v1[k];

				if (k==0) minVal= v0[k];
				else { if (v0[k] < minVal) minVal= v0[k];}
			}

			if (minVal> max_distance) return float(max_distance)/normalisation;

		}
		return v0[y_len] / normalisation; 
	}

}
