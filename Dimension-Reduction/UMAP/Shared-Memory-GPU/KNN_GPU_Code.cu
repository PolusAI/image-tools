/**
 * Computing KNN in high-D space. 
 * This section of program is CUDA-enabled version of the algorithm developed by Dong et al., 2012,
 * titled "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures".
 */
#include <vector>
#include <iostream>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h>      
#include <list>
#include <string>
#include <math.h>
#include <fstream>
#include <float.h>
#include <boost/iostreams/stream.hpp> 
#include "KNN_GPU_Code.cuh"
#include "Metrics.h"  
#include "Metrics.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include "highDComputes.h"

using namespace std;

/**
 * The Max number of Threads per Block. This is one of GPU hardware characteristics.
 */
#define MAXTPB 1024

/**
 * The Minimum number of (pair distance) computations needed to switch to GPU device (Otherwise it is more efficient to stay in host)
 */
#define MinimumThreads 10

/**
 * GPU Kernel definition
 */
__global__ void ComputeDistancesKernel(int * device_New_Final_List_1D, int * device_New_Final_List_Index, int Dim, double * device_New_Final_List_Dist_1D, double * device_dataPointsGPU, int * device_New_Final_List_Dist_Index, int metricID, float distanceV1, float distanceV2, float * v0, float * v1){

	int localDim=Dim;
	double localvalue=0;
	int Cnts=device_New_Final_List_Index[blockIdx.x+1]-device_New_Final_List_Index[blockIdx.x];
	int Cnts_Dist=device_New_Final_List_Dist_Index[blockIdx.x+1]-device_New_Final_List_Dist_Index[blockIdx.x];	
	int par1, par2;
	int cnt=0;
	int flag=0;

	if (threadIdx.x < Cnts_Dist){
		for (int i=0; i < Cnts; ++i){
			if (flag ==1) break;
			for (int j=i+1; j < Cnts; ++j){
				if (threadIdx.x == cnt) {
					par1 = device_New_Final_List_1D[i + device_New_Final_List_Index[blockIdx.x]];
					par2 = device_New_Final_List_1D[j + device_New_Final_List_Index[blockIdx.x]]; 
					flag=1;
					break;         
				}
				++cnt;
			}
		}

        localvalue= distanceCompute (localDim, device_dataPointsGPU, par1, par2, metricID, distanceV1, distanceV2, v0, v1);
        
		int IndexIDWrite= device_New_Final_List_Dist_Index[blockIdx.x]+threadIdx.x;
		device_New_Final_List_Dist_1D[IndexIDWrite] = localvalue;	
	}
	return;
}

/**
 * Replace the farthest point in B_Index (for u1) with u2 if u2 is closer
 * <p>
 * This method corresponds to UPDATENN(B[u1],<u2,l,true>) in the paper
 * </p>
 * @param  Dist  represents B_Dist
 * @param  Index represents B_Index
 * @param  IsNew represents B_IsNew
 * @param  u1    the indice of point that we want to potentially update its K-NN with the point u2
 * @param  u2    the indice of potential K-NN fpr point u1
 * @param  distance the spatial distance between u1 and u2
 * @param  flag updates B_IsNew
 * @return 1 if B_Index[u1][.] is updated, 0 otherwise
 */
int UpdateNN (int** B_Index, double ** B_Dist, short** B_IsNew, short* allEntriesFilled, int K, int u1, int u2, double distance, int flag = 1) {

	if(allEntriesFilled[u1]==0){		
		for (int j = 0; j < K; j++) {	
			if (B_Dist[u1][j] < 0) {

				for (int jj = 0; jj < j; jj++) {if (B_Index[u1][jj] == u2) return 0;}

				B_Dist[u1][j] = distance;
				B_Index[u1][j] = u2;
				B_IsNew[u1][j] = flag;
				if (j==K-1) allEntriesFilled[u1]=1;
				return 1;}
		}
	}

	else{
		for (int j = 0; j < K; j++) {
			if (B_Index[u1][j] == u2) return 0;
		}

		double max = DBL_MIN;
		int index = -1;
		for (int j = 0; j < K; j++) {
			if (B_Dist[u1][j] > max) {
				max = B_Dist[u1][j];
				index = j;
			}
		}
		if (index == -1) { cout << "Error"<<endl; } 
		if (distance < max) {
			B_Dist[u1][index] = distance;
			B_Index[u1][index] = u2;
			B_IsNew[u1][index] = flag;
			return 1;
		}
		else { return 0; }
	}
	return 0;  
}

/**
 * Compute K-NN following the algorithm for shared-memory K-NN
 * @param filePath The full path to the input file containig the dataset.
 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1).	 
 * @param Dim Dimension of Dataset (#Columns) 
 * @param K the desired number of Nearest Neighbours to be computed
 * @param sampleRate The rate at which we do sampling
 * @param convThreshold Convergance Threshold
 * @param logFile The errors and informational messages are outputted to the log file 
 * @param distanceMetric is the metric to compute the distance between the points in high-D space, by deafult should be euclidean
 * @param distanceV1 is the first optional variable needed for computing distance in some metrics
 * @param distanceV2 is the second optional variable needed for computing distance in some metrics	
 * @param filePathOptionalArray The full path to optional array for the distance metric computation 	 
 * @return B_Index indices of K-NN for each data point 	 
 * @return B_Dist corresponding distance for K-NN indices stored in B_Index	 
 */
void computeKNNs(string filePath, const int N, const int Dim, const int K, float sampleRate, const int convThreshold,int** B_Index,double** B_Dist, ofstream& logFile, string distanceMetric, float distanceV1, float distanceV2, string filePathOptionalArray){ 

	logFile<<"------------Starting K-NN Solution------------"<<endl;
	cout<<"------------Starting K-NN Solution------------"<<endl;
	/**
	 * A 2D Array containing the entire input dataset (read from filePath).
	 */
	double** dataPoints = new double*[N];
	for (int i = 0; i < N; ++i) { dataPoints[i] = new double[Dim]; }

	double* dataPointsGPU = new double[N*Dim];
	/**
	 * corresponding flag for K-NN indices stored in B_Index
	 */
	short** B_IsNew = new short*[N];
	for (int i = 0; i < N; ++i) { B_IsNew[i] = new short[K]; }
	/**
	 * Data structure for new[v]
	 */
	vector<int> *New_Index = new std::vector<int>[N];
	/**
	 * Data structure for REVERSE(new[v]) or new'
	 */
	vector<int> *Reverse_New_Index = new vector<int>[N];
	/**
	 * Data Structure for SAMPLE(new'[v],pk)
	 */
	vector<int> *Sampled_Reverse_New_Index = new vector<int>[N];
	/**
	 * Data Structure for new[v] U SAMPLE(new'[v],pk)
	 */
	vector<int> *New_Final_List = new vector<int>[N];
	/**
	 * An approximation of zero in computing distances. Two points with the distance
	 * smaller than epsilon are considered as one point.
	 */
	double epsilon = 1e-10; 
	short* allEntriesFilled = new short[N];
	/**
	 * At first, let's Read Dataset from Input File
	 */
	ifstream infile;
	infile.open(filePath);
	if (infile.fail())
	{
		logFile << "error in Opening Input File" << endl;
		cout << "error in Opening Input File" << endl;
		return ;
	}
	/**
	 * Remove the header info
	 */
	string dummyLine;
	getline(infile, dummyLine);
	/**
	 * Reading the Entire Dataset
	 */
	for (int i = 0; i < N; ++i) {
		string temp, temp2;
		getline(infile, temp);
		for (int j = 0; j < Dim; ++j) {
			temp2 = temp.substr(0, temp.find(","));
			double tempV=atof(temp2.c_str());
			dataPoints[i][j] = tempV;
			dataPointsGPU[i*Dim+j] = tempV;
			temp.erase(0, temp.find(",") + 1);
		}
	}
	infile.close();
    
	/**
	 * Numeric of the metric
	 */    
    int metricID = classification(distanceMetric);
	/**
	 * Converting Pagged Memory to Pinned Memory for better performance in cudaMemcpyAsync
	 */
	cudaHostRegister(dataPointsGPU,N*Dim*sizeof(double),0);
	/**
	 * Copy the GPU version of input data (dataPointsGPU) to GPU memory (device_dataPointsGPU)
	 */
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	double * device_dataPointsGPU;
	cudaMalloc ((void **) &device_dataPointsGPU, N*Dim*sizeof(double));            
	cudaMemcpyAsync (device_dataPointsGPU, dataPointsGPU, N*Dim*sizeof(double),cudaMemcpyHostToDevice, stream); 	
	gpuErrchk(cudaPeekAtLastError());		

	/**
	 * define a seed for random generator. Using a constant value produces
	 * the same set of random numbers and is good for debugging. Alternatively,
	 * we can select the seed number randomly as srand(time(NULL))
	 */
	srand(17);
	/**
	 * Initialization of Arrays B_IsNew and B_Dist
	 */
	for (int i = 0; i < N; ++i) {
		allEntriesFilled[i]=0;
		for (int j = 0; j < K; ++j) {
			B_IsNew[i][j] = 1;
			B_Dist[i][j] = -1.0;
		}
	}
	/**
	 * Random Initialization of B_Index
	 */
	int randomIndex, iter;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {
			iter = 1;
			while (iter) {
				randomIndex = rand() % N;
				if (randomIndex != i) {
					B_Index[i][j] = randomIndex;
					iter = 0;
				}
			}
		}
	}

	/**
	 * Main Loop of the Algorithm
	 */
	bool iterate = true;
	while (iterate) {
		int c_criteria = 0;
		int abort=0;
		/**
		 * Create "New" for each Datapoint
		 */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < K; ++j) {
				if (float(rand() % 100) < sampleRate*100) {
					if (B_IsNew[i][j] == 1) {
						New_Index[i].push_back(B_Index[i][j]);
						B_IsNew[i][j] = 0;
					}
				}
			}
		}
		/**
		 * Create "New'"(or REVERSE("New")) for each Datapoint
		 */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < New_Index[i].size(); ++j) {
				Reverse_New_Index[New_Index[i][j]].push_back(i);
			}
		}
		/**
		 * Random Sampling from "New'"
		 */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < Reverse_New_Index[i].size(); ++j) {
				if (float(rand() % 100) < sampleRate*100) {
					Sampled_Reverse_New_Index[i].push_back(Reverse_New_Index[i][j]);
				}
			}
		}
		/**
		 * "New"= "New" U SAMPLE("New'", pK)
		 */
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < New_Index[i].size(); ++j) {
				New_Final_List[i].push_back(New_Index[i][j]);
			}
			for (int j = 0; j < Sampled_Reverse_New_Index[i].size(); ++j) {
				New_Final_List[i].push_back(Sampled_Reverse_New_Index[i][j]);
			}
		}
		/**
		 * Remove duplicates from New_Final_List
		 */
		for (int i = 0; i < N; ++i) {	
			sort(New_Final_List[i].begin(), New_Final_List[i].end());
			auto last = std::unique(New_Final_List[i].begin(), New_Final_List[i].end());
			New_Final_List[i].erase(last, New_Final_List[i].end());
		}

		/**
		 * Max_New_Final_List_Length is the maximum length of New_Final_List array
		 */
		int Max_New_Final_List_Length=0;

		for (int i = 0; i < N; ++i) {       
			if (New_Final_List[i].size()> Max_New_Final_List_Length) Max_New_Final_List_Length=New_Final_List[i].size();
		}
		/**
		 * ThreadsPerBlockNeeded is the required number of threads per block to compute the longest array of New_Final_List
		 */
		int ThreadsPerBlockNeeded=0;	
		for (int i = 0; i < Max_New_Final_List_Length; ++i) {              
			for (int j = i+1; j < Max_New_Final_List_Length; ++j) {				        
				++ThreadsPerBlockNeeded;	
			}
		}

		/**
		 * Switch to GPU computations if the following conditions met. Otherwise proceed to CPU computations. 
		 * For now, exclude the metrics depend on filePathOptionalArray from GPU computations
		 */		 
		if (ThreadsPerBlockNeeded < MAXTPB  && ThreadsPerBlockNeeded > MinimumThreads && filePathOptionalArray=="") { 
			/**
			 * TotalCounts is the total number of elements in New_Final_List
			 */		
			int TotalCounts=0;		
			for (int i = 0; i < N; ++i) {       
				TotalCounts += New_Final_List[i].size();
			}	
			/**
			 * New_Final_List_1D is the 1D representation of New_Final_List for transferring to GPU
			 */										
			int * New_Final_List_1D = new int [TotalCounts]; 
			int cnt=0;

			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < New_Final_List[i].size(); ++j) {	
					New_Final_List_1D[cnt] = New_Final_List[i][j];
					++cnt;
				}
			}	
			/**
			 * device_New_Final_List_1D is on the GPU memory and contains New_Final_List_1D
			 */	
			int *device_New_Final_List_1D;	
			cudaMalloc ((void **) &device_New_Final_List_1D, TotalCounts*sizeof(int)); 
			gpuErrchk(cudaMemcpy (device_New_Final_List_1D, New_Final_List_1D, TotalCounts* sizeof(int),cudaMemcpyHostToDevice)); 
			/**
			 * New_Final_List_Index is the index of New_Final_List[i] data. It is needed as New_Final_List has variable size in each row of data.
			 */									 
			int * New_Final_List_Index = new int [N+1];
			New_Final_List_Index[0] = 0;
			for (int i = 1; i < N+1; ++i) {	
				New_Final_List_Index[i] = New_Final_List[i-1].size()+New_Final_List_Index[i-1];
			}
			/**
			 * device_New_Final_List_Index is on the GPU memory and contains New_Final_List_Index
			 */	
			int *device_New_Final_List_Index;		
			cudaMalloc ((void **) &device_New_Final_List_Index, (N+1)*sizeof(int)); 
			gpuErrchk(cudaMemcpy (device_New_Final_List_Index, New_Final_List_Index, (N+1)* sizeof(int),cudaMemcpyHostToDevice));
			/**
			 * New_Final_List_Dist_Index is the index of pairs of distances computed in GPU. 
			 */							     	       	
			int * New_Final_List_Dist_Index = new int [N+1];         
			int TotalCounts_Dist=0;

			for (int i = 0; i < N; ++i) {
				New_Final_List_Dist_Index[i]=TotalCounts_Dist;
				for (int j = 0; j < New_Final_List[i].size(); ++j) {	
					for (int k = j+1; k < New_Final_List[i].size(); ++k) {	
						++TotalCounts_Dist;
					}
				}				
			}
			New_Final_List_Dist_Index[N]=TotalCounts_Dist;
			/**
			 * device_New_Final_List_Dist_Index is on the GPU memory and contains New_Final_List_Dist_Index
			 */	
			int * device_New_Final_List_Dist_Index;	            
			cudaMalloc ((void **) &device_New_Final_List_Dist_Index, (N+1)*sizeof(int)); 		                
			gpuErrchk(cudaMemcpy (device_New_Final_List_Dist_Index, New_Final_List_Dist_Index, (N+1) * sizeof(int),cudaMemcpyHostToDevice));
			/**
			 * device_New_Final_List_Dist_1D is on the GPU memory and contains 1D array of pairs of distances computed in GPU.
			 */						        
			double *device_New_Final_List_Dist_1D;  
			cudaMalloc ((void **) &device_New_Final_List_Dist_1D, TotalCounts_Dist*sizeof(double)); 
			
			/**
			 * Creating 2 arrays on device for the case that the Metric is levenshtein
			 */	
            float *device_v0, *device_v1;
			cudaMalloc ((void **) &device_v0, (Dim+1)*sizeof(float)); 
			cudaMalloc ((void **) &device_v1, (Dim+1)*sizeof(float)); 			
			/**
			 * Launch the Kernel to compute the distance computations for all pairs of the points.
			 * cudaDeviceSynchronize is required to ensure data transfer to GPU memory is already finished.
			 */				        
			gpuErrchk(cudaDeviceSynchronize());	

			logFile<< "Number of Blocks = "<<N<< " and Number of Threads Per Block = "<<ThreadsPerBlockNeeded<<endl;
			cout<< "Number of Blocks = "<<N<< " and Number of Threads Per Block = "<<ThreadsPerBlockNeeded<<endl;
			
			ComputeDistancesKernel<<<N, ThreadsPerBlockNeeded>>>(device_New_Final_List_1D,device_New_Final_List_Index, Dim,device_New_Final_List_Dist_1D, device_dataPointsGPU,device_New_Final_List_Dist_Index, metricID, distanceV1, distanceV2,device_v0,device_v1);
			gpuErrchk(cudaDeviceSynchronize());	
			/**
			 * New_Final_List_Dist_1D is on the host containing device_New_Final_List_Dist_1D
			 */				
			double * New_Final_List_Dist_1D = new double [TotalCounts_Dist]; 
			gpuErrchk(cudaMemcpy (New_Final_List_Dist_1D, device_New_Final_List_Dist_1D, TotalCounts_Dist* sizeof(double),cudaMemcpyDeviceToHost)); 

			/**
			 * Now that we have computed all the distance pairs on GPU, we update the appropriate arrays on host 
			 * c=c+UPDATENN(B[u1],<u2,l,true>)
			 */

			for (int i = 0; i < N; ++i) {
				if (abort != 0) break;
				int tmpcnt=0;

				for (int it = 0; it < New_Final_List[i].size(); ++it) {
					int par1= New_Final_List[i][it];

					for (int it2 = it+1; it2 < New_Final_List[i].size(); ++it2) {
						int par2= New_Final_List[i][it2];

						if (abort ==0) {
							double dista= New_Final_List_Dist_1D[New_Final_List_Dist_Index[i]+tmpcnt];
							++tmpcnt;

							if (dista < epsilon) {
								logFile << "Found Duplicate Data for Points "<< par1 << " and " << par2 <<endl;
								cout << "Found Duplicate Data for Points "<< par1 << " and " << par2 <<endl; 
								abort=1; iterate = false; 
							}

							c_criteria += UpdateNN(B_Index, B_Dist, B_IsNew, allEntriesFilled, K, par1, par2, dista, 1);
							c_criteria += UpdateNN(B_Index, B_Dist, B_IsNew, allEntriesFilled, K, par2, par1, dista, 1);

						}
					}
				}
			}

			/**
			 * Free the pointers' memory allocations on host and device
			 */

			cudaFree(device_New_Final_List_1D); 		   
			cudaFree(device_New_Final_List_Index);
			cudaFree(device_New_Final_List_Dist_Index);			
			cudaFree(device_New_Final_List_Dist_1D);

			delete [] New_Final_List_Dist_1D, New_Final_List_Index, New_Final_List_Dist_Index;
			delete [] New_Final_List_1D;

		} else {
			for (int i = 0; i < N; ++i) {
				if (abort != 0) break;

				for (int it = 0; it < New_Final_List[i].size(); ++it) {
					int par1= New_Final_List[i][it];

					for (int it2 = it+1; it2 < New_Final_List[i].size(); ++it2) {
						int par2= New_Final_List[i][it2];
						if (abort ==0) {
		
							double dista = computeDistance (distanceMetric, dataPoints, par1, par2, Dim, distanceV1, distanceV2, filePathOptionalArray, logFile);
							

							if (dista < epsilon) {
								logFile << "Found Duplicate Data for Points "<< par1 << " and " << par2 <<endl;; 
								cout << "Found Duplicate Data for Points "<< par1 << " and " << par2 <<endl; 
								abort=1;iterate = false; 
							}						
							c_criteria += UpdateNN(B_Index, B_Dist, B_IsNew, allEntriesFilled, K, par1, par2, dista, 1);
							c_criteria += UpdateNN(B_Index, B_Dist, B_IsNew, allEntriesFilled, K, par2, par1, dista, 1);
						}
					}
				}
			}
		}

		logFile << "c_criteria = " << c_criteria << " With Threshold Convergence of " << convThreshold << endl;
		cout << "c_criteria = " << c_criteria << " With Threshold Convergence of " << convThreshold << endl;
		if (c_criteria < convThreshold) { iterate = false; }
		/**
		 * Clear the contents of the used data structures
		 */
		for (int i = 0; i < N; ++i) {
			New_Index[i].clear();
			Reverse_New_Index[i].clear();
			Sampled_Reverse_New_Index[i].clear();
			New_Final_List[i].clear();
		}
	}
	// Free the memory
	cudaHostUnregister(dataPointsGPU);
	cudaFree(device_dataPointsGPU);	
	delete[] dataPoints, dataPointsGPU,allEntriesFilled,B_IsNew;

	logFile<<"------------Ending K-NN Solution------------"<<endl;
	cout<<"------------Ending K-NN Solution------------"<<endl;
	return;
}
