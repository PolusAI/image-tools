#include <math.h> 
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;
/**
 * Default number of GPU Threads Per Block
 */
#define TPB 32

/**
 * negative_sample_rate is a Constant variable on GPU device
 * negative_sample_rate is the rate at which we sample from the non-connected surrounding points as compared to the connected edges. 
 * Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
 */
#define negative_sample_rate_Value 5

__constant__ int negative_sample_rate=negative_sample_rate_Value;

/**
 * GPU function to compute euclidean distance in Low-D space
 */
__device__ double rdist(double * embedding, int Dim, int index1, int index2){
	double dist_squared=0;

	for (int j = 0; j < Dim; ++j) { 
		double tmp= embedding[index1*Dim+j]-embedding[index2*Dim+j];
		dist_squared += tmp *tmp;
	} 
	return dist_squared;			
}

/**
 * GPU Kernel to initialize curand state used for random number generation
 */
__global__ void setup_curand(curandState *state){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int seed=threadIdx.x; 
	curand_init(seed, idx, 0, &state[idx]);
}

/**
 * GPU function to clip the gradient
 */
__device__ double clip(double value){
	const double clipLowVal=-4.0;
	const double clipHighVal=4.0;
	double returnValue;

	if (value < clipLowVal) {returnValue=clipLowVal;}
	else if ( value > clipHighVal) {returnValue=clipHighVal;}
	else {returnValue=value;}

	return returnValue;
}

/**
 * GPU kernel to initialize some arrays on GPU memory
 * device_epoch_of_next_sample is an index of the epoch state of the edges. If it is less than epoch index, we will use the edge in the computation
 * device_epoch_of_next_negative_sample is an index of the epoch state of the edges for sampling from non-connected surrounding points. 
 */
__global__ void initializeEpochs(int edgeCounts, float* device_epochs_per_sample, float *device_epoch_of_next_sample, float *device_epochs_per_negative_sample, float *device_epoch_of_next_negative_sample){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < edgeCounts){
		device_epoch_of_next_sample[tid]=device_epochs_per_sample[tid];
		device_epochs_per_negative_sample[tid]=device_epochs_per_sample[tid]/negative_sample_rate;
		device_epoch_of_next_negative_sample[tid]=device_epochs_per_negative_sample[tid];	
	}
}		

/**
 * Main GPU kernel to solve Stochastic Gradient Descent (SGD) problem 
 */
__global__ void SGDEngine(double * embedding, int * head, int * tail, float alpha, int N,int DimLowSpace, float aValue, float bValue, int edgeCounts,float* device_epochs_per_sample, float *device_epoch_of_next_sample, float *device_epochs_per_negative_sample, float *device_epoch_of_next_negative_sample ,int n, int move_other,curandState *state){

	/**
	 *  dEpsilon is zero approximation in double precision
	 */	
	double dEpsilon=1e-14;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < edgeCounts){
		if (device_epoch_of_next_sample[i] <= n){ 	
			int headIndex = head[i];   
			int tailIndex = tail[i];  
			double dist_squared = rdist(embedding, DimLowSpace, headIndex, tailIndex); 

			double grad_coeff;
			if (dist_squared<dEpsilon) grad_coeff=0;  
			else {grad_coeff= -2.0*aValue*bValue*pow(dist_squared,bValue-1.0)/(1.0+aValue*pow(dist_squared,bValue)); } 
			double grad_d;

			for (int jj = 0; jj < DimLowSpace; ++jj) { 
				grad_d=alpha*clip(grad_coeff*(embedding[headIndex*DimLowSpace+jj]-embedding[tailIndex*DimLowSpace+jj]));
				atomicAdd(&embedding[headIndex*DimLowSpace+jj],grad_d);    
				if (move_other==1) 	{
					atomicAdd(&embedding[tailIndex*DimLowSpace+jj], -grad_d);				
				}
			}

			device_epoch_of_next_sample[i] += device_epochs_per_sample[i];
			int n_neg_samples = int(float(n - device_epoch_of_next_negative_sample[i])/ device_epochs_per_negative_sample[i]);     	      

			for (int ll = 0; ll < n_neg_samples; ++ll) { 
				float random= curand_uniform(&state[i]); 
				int randomIndex = random * N;				
				if (randomIndex==headIndex) continue;

				dist_squared= rdist(embedding, DimLowSpace, headIndex, randomIndex);  

				if (dist_squared < dEpsilon) grad_coeff=0; 
				else{ grad_coeff = 2.0*bValue/((0.001+dist_squared)*(1.0+aValue*pow(dist_squared,bValue))); } 

				for (int jj = 0; jj < DimLowSpace; ++jj) {  
					if  (grad_coeff > 0) {
						grad_d = alpha*clip(grad_coeff*(embedding[headIndex*DimLowSpace+jj]-embedding[randomIndex*DimLowSpace+jj]));
					} else  {						
						grad_d = alpha*4.0; 
					}
					atomicAdd(&embedding[headIndex*DimLowSpace+jj], grad_d);
				}            	    
			}
			device_epoch_of_next_negative_sample[i] += (n_neg_samples * device_epochs_per_negative_sample[i]);  
		}
	}
}


