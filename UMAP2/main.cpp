/**
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * This code is an implementation of UMAP algorithm for dimension reduction. 
 * The reference paper is “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction“, by McInnes et al., 2018 (https://arxiv.org/abs/1802.03426)
 * Jan 2020
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
#include <boost/filesystem.hpp>
#include "KNN_Serial_Code.h"
#include "highDComputes.h"
#include "Initialization.h"
#include "LMOptimization.h"
#include "SGD.h"
#include <exception>
#include <sstream>
#include <omp.h>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

int main(int argc, char ** argv) {
	/**
	 * The errors and informational messages are outputted to the log file 
	 */
	ofstream logFile;
	string logFileName="Setting.txt";
	logFile.open(logFileName);
	/**
	 * The input parameters are read from command line which are as follow.
	 * filePath: The full path to the input file containig the dataset.
	 * outputPath: The full path to the output csv file containing the coordinates of data in the embedding space.
	 * K: K in K-NN that means the desired number of Nearest Neighbours to be computed.
	 * sampleRate: The rate at which we do sampling. This parameter plays a key role in the performance.
	 * This parameter is a trades-off between the performance and the accuracy of the results.
	 * Values closer to 1 provides more accurate results but the execution takes longer.
	 * DimLowSpace: Dimension of Low-D or embedding space (usually 1,2,or 3).
	 * randomInitializing: Defining the Method for Initialization of data in low-D space
	 * n_epochs: is the number of training epochs to be used in optimizing. Larger values result in more accurate embeddings
	 * min_dist defines how tight the points are from each other in Low-D space
	 * distanceMetric is the metric to compute the distance between the points in high-D space, by deafult should be euclidean
	 * distanceV1 is the first optional variable needed for computing distance in some metrics
	 * distanceV2 is the second optional variable needed for computing distance in some metrics
	 * inputPathOptionalArray is the full path to the directory that contains a csv file of the optional array needed for computing distance in some metrics. 
	 */
	string filePath, filePathOptionalArray="", outputPath, LogoutputPath, inputPath;
	int K,DimLowSpace,n_epochs;
	float sampleRate,min_dist,distanceV1=0,distanceV2=0;
	bool randomInitializing;
	string distanceMetric="euclidean";

	for (int i=1; i<argc;++i){
		if (string(argv[i])=="--inputPath") {
			inputPath=argv[i+1];

			if(!boost::filesystem::exists(inputPath) || !boost::filesystem::is_directory(inputPath))
			{
				logFile << "Incorrect input path";
				cout << "Incorrect input path";
				return 1;
			}

			const std::string ext = ".csv";
			boost::filesystem::recursive_directory_iterator it(inputPath);
			boost::filesystem::recursive_directory_iterator endit;

			bool fileFound = false;
			while(it != endit) {
				if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ext){
					fileFound = true;
					filePath = it->path().string();
					break;
				}
				++it;
			}
			if (!fileFound){
				logFile << "CSV file is not found in the input path";
				cout << "CSV file is not found in the input path";
				return 1;
			}
		}
		else if (string(argv[i])=="--K") K=atoi(argv[i+1]);
		else if (string(argv[i])=="--sampleRate") sampleRate=stof(argv[i+1]);
		else if (string(argv[i])=="--min_dist") min_dist=stof(argv[i+1]);
		else if (string(argv[i])=="--DimLowSpace") DimLowSpace=atoi(argv[i+1]);
		else if (string(argv[i])=="--randomInitializing") {
			std::stringstream ss(argv[i+1]);
			ss >> std::boolalpha >> randomInitializing;
		}
		else if (string(argv[i])=="--outputPath"){
			boost::filesystem::path p(argv[i+1]);

			if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
			{
				logFile << "Incorrect output path";
				cout << "Incorrect output path";
				return 1;
			}

			LogoutputPath=argv[i+1];
			boost::filesystem::path joinedPath = p / boost::filesystem::path("ProjectedData_EmbeddedSpace.csv");
			outputPath = joinedPath.string();

		}
		else if (string(argv[i])=="--n_epochs") n_epochs=atoi(argv[i+1]);
		else if (string(argv[i])=="--distanceMetric") distanceMetric=argv[i+1];
		else if (string(argv[i])=="--distanceV1") distanceV1=stof(argv[i+1]);
		else if (string(argv[i])=="--distanceV2") distanceV2=stof(argv[i+1]);
		else if (string(argv[i])=="--inputPathOptionalArray") {
			string inputPathOptionalArray=argv[i+1];

			if(!boost::filesystem::exists(inputPathOptionalArray) || !boost::filesystem::is_directory(inputPathOptionalArray))
			{
				logFile << "Incorrect input path";
				cout << "Incorrect input path";
				return 1;
			}

			const std::string ext = ".csv";
			boost::filesystem::recursive_directory_iterator it(inputPathOptionalArray);
			boost::filesystem::recursive_directory_iterator endit;

			bool fileFound = false;
			while(it != endit) {
				if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ext){
					fileFound = true;
					filePathOptionalArray = it->path().string();
					break;
				}
				++it;
			}
			if (!fileFound){
				logFile << "CSV file is not found in the input path";
				cout << "CSV file is not found in the input path";
				return 1;
			}
		}

	}
	logFile<<"------------The following Input Arguments were read------------"<<endl;
	logFile<<"The full path to the input file: "<< filePath<<endl;
	logFile<<"The full path to the output file: "<< outputPath<<endl;
	logFile<<"The desired number of NN to be computed: "<< K <<endl;
	logFile<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl;  
	logFile<<"The Dimension of Low-D Space: "<< DimLowSpace <<endl; 
	logFile << std::boolalpha;
	logFile<<"Random Initialization of Points in Low-D Space: "<< randomInitializing <<endl; 
	logFile<<"The number of training epochs: "<< n_epochs <<endl; 
	logFile<<"The chosen min_dist parameter: "<< min_dist <<endl; 	
	logFile<<"The metric to compute the distance between the points in high-D space: "<< distanceMetric <<endl; 
	logFile<<"The optional variable 1 for the distance: "<< distanceV1 <<endl; 
	logFile<<"The optional variable 2 for the distance: "<< distanceV2 <<endl;
	logFile<<"The full path to optional array for the distance metric computation: "<< filePathOptionalArray <<endl;	

	cout<<"------------The following Input Arguments were read------------"<<endl;
	cout<<"The full path to the input file: "<< filePath<<endl;
	cout<<"The full path to the output file: "<< outputPath<<endl;
	cout<<"The desired number of NN to be computed: "<< K <<endl;
	cout<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl;   
	cout<<"The Dimension of Low-D Space: "<< DimLowSpace <<endl; 
	cout << std::boolalpha;
	cout<<"Random Initialization of Points in Low-D Space: "<< randomInitializing <<endl; 
	cout<<"The number of training epochs: "<< n_epochs <<endl; 
	cout<<"The chosen min_dist parameter: "<< min_dist <<endl; 	
	cout<<"The metric to compute the distance between the points in high-D space: "<< distanceMetric <<endl; 
	cout<<"The optional variable 1 for the distance: "<< distanceV1 <<endl; 
	cout<<"The optional variable 2 for the distance: "<< distanceV2 <<endl;
	cout<<"The full path to optional array for the distance metric computation: "<< filePathOptionalArray <<endl;
	/**
	 * Size of Dataset without the header (i.e.(#Rows in dataset)-1).
	 */
	string cmd="wc -l "+filePath;
	string outputCmd = exec(cmd.c_str());
	const int N=stoi(outputCmd.substr(0, outputCmd.find(" ")))-1;
	logFile<<"The Dimension of Dataset Records (Number of Rows in inputfile w/o header ): "<< N <<endl;
	cout<<"The Dimension of Dataset Records (Number of Rows in inputfile w/o header ): "<< N <<endl;
	/**
	 * Dimension of Dataset (#Columns)
	 */
	int Dim;
	string cmd2="head -n 1 "+ filePath + " |tr '\\,' '\\n' |wc -l ";
	Dim = stoi(exec(cmd2.c_str())); 
	logFile<<"The Dimension of Dataset Features(Number of Columns in inputfile): "<< Dim <<endl;
	cout<<"The Dimension of Dataset Features(Number of Columns in inputfile): "<< Dim <<endl;

	if (K > N) {
		logFile<<" The desired number of NN has exceeded the size of dataset "<<endl;
		cout<<" The desired number of NN has exceeded the size of dataset "<<endl;   
		return 1;
	}

	logFile<<"------------END of INPUT READING------------"<< endl;	
	cout<<"------------END of INPUT READING------------"<< endl;
	srand(17);	

	/**
	 * convThreshold: Convergance Threshold of K-NN. A fixed integer is used here instead of delta*N*K. 
	 */		 
	const int convThreshold=5;
	/**
	 * indices of K-NN for each data point
	 */
	int** B_Index = new int*[N];
	for (int i = 0; i < N; ++i) { B_Index[i] = new int[K]; }	
	/**
	 * corresponding distance for K-NN indices stored in B_Index
	 */
	double** B_Dist = new double*[N];
	for (int i = 0; i < N; ++i) { B_Dist[i] = new double[K]; }

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
	computeKNNs(filePath, N, Dim, K, sampleRate, convThreshold,B_Index,B_Dist, logFile, distanceMetric, distanceV1, distanceV2,filePathOptionalArray);

	bool flag=false;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {
			if (B_Dist[i][j] < 0) {         
				logFile<<"ALERT: A distance in high-D space was computed as negative, use this program with caution"<<endl;
				cout<<"ALERT: A distance in high-D space was computed as negative, use this program with caution"<<endl; 
				flag=true;
				break; 
			}     
		}
		if (flag) break;
	}

	int* B_Index_Min = new int[N];
	double* B_Dist_Min = new double[N];
	/**
	 * Compute B_Index and B_Dist for the closest points (K-NNs) 
	 * @param B_Index indices of K-NN for each data point 	
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	 	 	 	 
	 * @return B_Index_Min B_Index for the closest point 
	 * @return B_Dist_Min B_Dist for the corresponding B_Index_Min
	 */
	findMin(B_Index,B_Dist, N,K,B_Index_Min,B_Dist_Min);

	double* SigmaValues = new double[N];
	/**
	 * Compute SigmaValues for each data point (Smooth approximator to K-NN distance) iteratively
	 * @param B_Dist corresponding distance for K-NN indices stored in B_Index
	 * @param B_Dist_Min B_Dist for the corresponding B_Index_Min
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param K the desired number of Nearest Neighbours to be computed	
	 * @return SigmaValues An array of Sigma Values for data 	 	 	 
	 */
	findSigma(B_Dist, B_Dist_Min,SigmaValues, N, K);

	/**
	 * To save memory space, "Sparse Matrix" data structure is used here
	 * SparseMatrix by default is oriented column-major
	 */
	SparseMatrix<float> adjacencyMatrixA(N,N), adjacencyMatrixAT(N,N), graphSM(N,N);	 
	typedef Eigen::Triplet<float> T;
	std::vector<T> tripletList;
	tripletList.reserve(N*K);

	for (int i=0; i<N; ++i){
		for (int j=0; j<K; ++j){
			int point2=B_Index[i][j]; 
			float tmp=exp((B_Dist_Min[i]-B_Dist[i][j])/SigmaValues[i]);
			tripletList.push_back(T(i,point2,tmp));
		}
	}

	adjacencyMatrixA.setFromTriplets(tripletList.begin(), tripletList.end());		
	adjacencyMatrixAT=adjacencyMatrixA.transpose();
	graphSM=adjacencyMatrixA+adjacencyMatrixAT;
	graphSM -=adjacencyMatrixA.cwiseProduct(adjacencyMatrixAT);

	float MaxWeight=0;
	for (int k=0; k<graphSM.outerSize(); ++k){
		float sum=0;
		int index=-1;
		for (SparseMatrix<float>::InnerIterator it(graphSM,k); it; ++it) {
			sum += it.value(); 
			if (it.value() > MaxWeight) MaxWeight=it.value();  
		}
	} 

	logFile<<"------------Setting Low-D Space Design------------"<<endl;
	cout<<"------------Setting Low-D Space Design------------"<<endl;

	/**
	 * embedding is the coordinates of the points in the low-D space  
	 */
	double** embedding = new double*[N];
	for (int i = 0; i < N; ++i) { embedding[i] = new double[DimLowSpace]; }    

	logFile<<"------------Starting Initialization in the Low-D Space------------"<<endl;
	cout<<"------------Starting Initialization in the Low-D Space------------"<<endl;
	/**
	 * Initializes the data points in low-D space
	 * @param randomInitializing the methodology for Initialization of data in low-D space
	 * @param logFile contains the errors and informational messages 
	 * @param N Size of Dataset without the header (i.e.(#Rows in dataset)-1). 
	 * @param graph contains undirected weights (similarities) in the form of a matrix of size NxN
	 * @param DimLowSpace Dimension of Low-D space 	 	 
	 * @return embedding is the coordinates of the points in the low-D space	 	 	 
	 */
	Initialization (randomInitializing, embedding, logFile, N, graphSM, MaxWeight, DimLowSpace, n_epochs);

	logFile<<"------------Starting Estimating Hyper-Parameters a and b ------------"<<endl;
	cout<<"------------Starting Estimating Hyper-Parameters a and b ------------"<<endl;
	/**
	 * Hyper-Parameters a and b which needs to be estimated by data fitting. 
	 */
	float aValue, bValue;
	float spread=1.0;
	/**
	 *  Estimation of Hyper-Parameters a and b by curve fitting and using Levenberg-Marquardt solution
	 */		
	estimateParameters(aValue, bValue, min_dist, spread, logFile);

	logFile<<"The Estimated Values for a is "<< aValue << " and for b is "<< bValue <<endl;
	cout<<"The Estimated Values for a is "<< aValue << " and for b is "<< bValue <<endl;

	logFile<<"------------Starting Solution for Stochastic Gradient Descent (SGD)------------"<<endl;	
	cout<<"------------Starting Solution for Stochastic Gradient Descent (SGD)------------"<<endl;
	/**
	 *  alpha is the initial learning rate for the SGD. alpha starts from 1 and decreases in each epoch iteration
	 */	
	float alpha=1.0;  
	/**
	 * epochs_per_sample is a vector of edges with the values proportional to the values in graph 
	 * epochs_per_sample represents the epoch weight for edges where the edge with the highest similarity will get the value of 1 
	 * and all other edges will get a proportional epoch weight scaled from it. epochs_per_sample is used as a measure to include an edge in 
	 * SGD computations. The edge with the highest similarity will be used at every epoch iteration. 
	 * head is a vector containing the head index of the edge
	 * tail is a vector containing the tail index of the edge
	 */
	vector<float> head, tail;
	vector<float> epochs_per_sample;

	for (int k=0; k<graphSM.outerSize(); ++k){
		for (SparseMatrix<float>::InnerIterator it(graphSM,k); it; ++it) {
		    if (it.value() <  MaxWeight/n_epochs) continue;  
			epochs_per_sample.push_back(MaxWeight/it.value());
			head.push_back(it.col());
			tail.push_back(it.row()); 
		}
	}

	/**
	 * This section was adopted from SGD implementation at https://github.com/lmcinnes/umap/blob/8f2ef23ec835cc5071fe6351a0da8313d8e75706/umap/layouts.py#L136
	 * edgeCounts is total number of edges in the high-D space graph
	 * epoch_of_next_sample is an index of the epoch state of the edges. If it is less than epoch index, we will use the edge in the computation
	 * epoch_of_next_negative_sample is an index of the epoch state of the edges for sampling from non-connected surrounding points. 
	 * negative_sample_rate is the rate at which we sample from the non-connected surrounding points as compared to the connected edges. 
	 * Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
	 */	 
	int edgeCounts=epochs_per_sample.size();
	const int negative_sample_rate=5;
	int n_neg_samples;
	float epoch_of_next_sample[edgeCounts];    
	float epochs_per_negative_sample[edgeCounts]; 
	float epoch_of_next_negative_sample[edgeCounts];  

	for (int i = 0; i < edgeCounts; ++i) {
		epoch_of_next_sample[i]=epochs_per_sample[i];
		epochs_per_negative_sample[i]=epochs_per_sample[i]/negative_sample_rate;
		epoch_of_next_negative_sample[i]=epochs_per_negative_sample[i];
	}  
	/**
	 *  move_other is equal to 1 if not embedding new previously unseen points to low-D space
	 */
	const int move_other=1; 
	/**
	 *  dEpsilon is zero approximation in double precision
	 */	
	const double dEpsilon=1e-14;
	double dist_squared;
	// The main training loop     
	for (int n = 0; n < n_epochs; ++n) {

		//Loop over all edges of the graph 
		if (n/10*10 == n){
			logFile << "SGD iteration = "<<n<<" from "<< n_epochs <<endl;
			cout << "SGD iteration = "<<n<<" from "<< n_epochs <<endl;
		}

		for (int i = 0; i < edgeCounts; ++i) {  	
			if (epoch_of_next_sample[i] <= n){ 	

				int headIndex = head[i];   
				int tailIndex = tail[i];  

				dist_squared = rdist(embedding, DimLowSpace, headIndex, tailIndex);

				double grad_coeff;
				if (dist_squared<dEpsilon) grad_coeff=0;  
				else {grad_coeff= -2.0*aValue*bValue*pow(dist_squared,bValue-1)/(1.0+aValue*pow(dist_squared,bValue)); }

				for (int jj = 0; jj < DimLowSpace; ++jj) { 
					embedding[headIndex][jj] += alpha*clip(grad_coeff*(embedding[headIndex][jj]-embedding[tailIndex][jj]));

					if (move_other==1) 	{
						embedding[tailIndex][jj] += -alpha*clip(grad_coeff*(embedding[headIndex][jj]-embedding[tailIndex][jj]));							
					}
				}

				epoch_of_next_sample[i] += epochs_per_sample[i];
				n_neg_samples = int((n - epoch_of_next_negative_sample[i])/ epochs_per_negative_sample[i]);     	      

				for (int ll = 0; ll < n_neg_samples; ++ll) {	    	
					int randomIndex = rand() % N;
					if (randomIndex==headIndex) continue;

					dist_squared= rdist(embedding, DimLowSpace, headIndex, randomIndex);

					if (dist_squared < dEpsilon) grad_coeff=0; 
					else{ grad_coeff = 2.0*bValue/((0.001+dist_squared)*(1.0+aValue*pow(dist_squared,bValue))); }

					for (int jj = 0; jj < DimLowSpace; ++jj) {  
						if  (grad_coeff > 0) {
							embedding[headIndex][jj] += alpha*clip(grad_coeff*(embedding[headIndex][jj]-embedding[randomIndex][jj]));
						} else  {						
							embedding[headIndex][jj] += alpha*4.0; 
						}
					}    	        
				}       	    
				epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i]);  
			}   	
		}    	
		alpha=1.0-((float)n)/n_epochs;    	
	}

	logFile<<"------------Starting Outputing the Results------------"<<endl;
	cout<<"------------Starting Outputing the Results------------"<<endl;
	/**
	 * Output the coordinates of the projected data in the low-D space
	 */ 
	ofstream embeddedSpacefile;
	embeddedSpacefile.open(outputPath);

	for (int j = 0; j < DimLowSpace; ++j) {
		if (j != DimLowSpace-1) embeddedSpacefile<<"Dimension"<<j+1<<",";
		else embeddedSpacefile<<"Dimension"<<j+1<<endl;
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < DimLowSpace; ++j) {		
			if (j==DimLowSpace-1) {
				embeddedSpacefile<< embedding[i][j]<<endl;}
			else {embeddedSpacefile<< embedding[i][j]<<",";}
		}
	}

	embeddedSpacefile.close();
	logFile.close();
	/**
	 * copy Logfile to the file system which could be accessed outside the docker container
	 */ 
	string cmd3="cp "+ logFileName+"  "+LogoutputPath;
	// To remove the returning messages, we can switch to the following command 
	//	string cmd3="cp "+ logFileName+"  "+LogoutputPath+ " 2>&1 /dev/null";
	string outputCmd3 = exec(cmd3.c_str());

	return 0;
}



