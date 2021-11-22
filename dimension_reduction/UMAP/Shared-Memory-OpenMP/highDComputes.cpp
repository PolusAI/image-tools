/**
 * Extra Functions needed for computations 
 */

#include <math.h> 
#include <float.h>
#include <iostream>

using namespace std;


/**
 * Compute B_Index and B_Dist for the closest points (K-NNs) 
 */
void findMin(int** B_Index,double** B_Dist, int N,int K,int* B_Index_Min,double* B_Dist_Min){

	for (int i=0;i<N;++i){
		double minValue=B_Dist[i][0];
		int minID=B_Index[i][0];

		for (int j=1; j<K; ++j){

			if (B_Dist[i][j]<minValue){
				minValue=B_Dist[i][j];
				minID=B_Index[i][j];
			}
		}
		B_Index_Min[i]=minID;
		B_Dist_Min[i]=minValue;
	}
}

/**
 * Smooth approximator to K-NN distance
 */
void findSigma(double ** B_Dist, double * B_Dist_Min, double * SigmaValues, int N, int K){
	/**
	 * Right-side of equation to compute SigmaValues
	 */
	double target=log2(K);
	/**
	 * Design Parameters to estimate SigmaValues
	 */	
	const int iterations=640;
	const double Error=1e-5;

	for (int i=0; i<N; ++i){
		double sigma=1;
		double low=0;
		double high=DBL_MAX;

		for (int iter=0; iter<iterations; ++iter){
			double sum=0;
			for (int j=0; j<K; ++j){
				sum += exp((B_Dist_Min[i]-B_Dist[i][j])/sigma);
			}

			if ( abs(sum-target) < Error) break;

			if (sum > target){
				high=sigma;
				sigma=(low+high)*0.5;
			}
			else{
				low=sigma;
				if (high == DBL_MAX) {
					sigma *=2;
				}
				else{
					sigma=(low+high)*0.5;
				}
			}
		}
		SigmaValues[i] = sigma;
	}
}




