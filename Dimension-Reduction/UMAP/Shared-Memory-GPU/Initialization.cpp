/**
 * Initializing the location of data points in the low-D space
 */
#include <iostream>
#include <armadillo>
#include <chrono>
#include <omp.h>
#include <Eigen/Sparse>

#define SCALE 0.0001

using namespace std;
using namespace arma;
using namespace Eigen;

void Initialization (bool randominitializing, double** embedding, ofstream& logFile, int N, SparseMatrix<float> & graphSM, float MaxWeight, int DimLowSpace, int n_epochs){

	srand(17);
	/**
	 *  zero approximation in float precision
	 */
	float epsilon=1e-6;		
	/**
	 * By deafult, low-D space dimensions are between -10 and 10 
	 */
	int minDimLowDSpace=-10;
	int maxDimLowDSpace=10;    

	if (!randominitializing){
		try{
			logFile<<" Spectral Initialization of Data in Lower Space"<<endl;
			cout<<" Spectral Initialization of Data in Lower Space"<<endl;

			/**
			 * graph is undirected weights (similarities) function for all the edges in the high-D space 
			 */
			float** graph = new float*[N];
			for (int i = 0; i < N; ++i) { graph[i] = new float[N]; }	

#pragma omp parallel for
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					graph[i][j]=0;
				}
			}

			for (int k=0; k<graphSM.outerSize(); ++k){
				for (SparseMatrix<float>::InnerIterator it(graphSM,k); it; ++it) {
					graph[it.row()][it.col()]=it.value();
				}
			} 

			/**
			 * Removing the small weights in accordance to https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1032
			 */
#pragma omp parallel for
			for (int i = 0; i < N; ++i){			
				for (int j = 0; j < N; ++j){
					if (graph[i][j] <  epsilon) continue; 
					if (graph[i][j] <  MaxWeight/n_epochs) graph[i][j]=0;  
				}
			}	

			/**
			 * DegreeMatrix is a diagonal matrix contains information about the degree of each vertex 
			 * sqrtDegreeMatrix transforms the diagonal values of DegreeMatrix by 1.0/sqrt()
			 */
			float** sqrtDegreeMatrix = new float*[N];
			for (int i = 0; i < N; ++i) { sqrtDegreeMatrix[i] = new float[N]; }

			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					sqrtDegreeMatrix[i][j]=0;
				}
			}

			for (int i = 0; i < N; ++i) {
				float sum=0;
				for (int j = 0; j < N; ++j) {	
					sum+=graph[i][j];
				}
				sqrtDegreeMatrix[i][i]=1.0/sqrt(sum);
			}
			/**
			 * aux_mem is the column-wise transformation of sqrtDegreeMatrix as needed by armadillo function fmat
			 */
			float* aux_mem = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem[j*N+i]=sqrtDegreeMatrix[i][j];      
				}
			}
			delete [] sqrtDegreeMatrix;
			/**
			 * Making an armadillo sparse matrix spmatDegreeMatrix from sqrtDegreeMatrix
			 */
			fmat matDegreeMatrix(aux_mem,N,N,false,true);
			sp_fmat spmatDegreeMatrix(matDegreeMatrix);    
			/**
			 * aux_mem2 is the column-wise transformation of adjacencyMatrix as needed by armadillo function fmat
			 */
			float* aux_mem2 = new float[N*N];  
			for (int i = 0; i < N; ++i){
				for (int j = 0; j < N; ++j){
					aux_mem2[j*N+i]=graph[i][j];   //column-wise
				}
			}

			delete[] graph;

			/**
			 * Making an armadillo sparse matrix spmatadjacencyMatrix from adjacencyMatrix
			 */
			fmat matadjacencyMatrix(aux_mem2,N,N,false,true);
			sp_fmat spmatadjacencyMatrix(matadjacencyMatrix);
			/**
			 * Making an armadillo sparse matrix of identity 
			 */			    
			sp_fmat Unity = speye<sp_fmat>(N,N); 
			/**
			 * Making an armadillo sparse matrix of Laplacian 
			 */	
			sp_fmat laplacianMatrix;
			laplacianMatrix= Unity-spmatDegreeMatrix*spmatadjacencyMatrix*spmatDegreeMatrix;
			/**
			 * Solving eigenvalue and eigenvector for Laplacian matrix
			 */
			fvec eigval;
			fmat eigvec;
			eigs_sym(eigval, eigvec, laplacianMatrix, DimLowSpace+1 , "sm"); 
			/**
			 * Converting eigenvectors to tmpvector 
			 * will throw "error: Mat::col(): index out of bounds" if no eigvec was available
			 */
			typedef std::vector<float> stdvec;
			std::vector< std::vector<float> > tmpvector;

			for (int i = 1; i < DimLowSpace+1; ++i) {
				stdvec vectest = arma::conv_to< stdvec >::from(eigvec.col(i));			
				tmpvector.push_back(vectest);  
			}
			/**
			 * using tmpvector to intialize the locations of the points in low-D space
			 * embedding should not be outside the chosen dimensions for low-D space
			 */
			double maxembedding=0;
			for (int j = 0; j < DimLowSpace; ++j) {    
				for (int i = 0; i < N; ++i) {
					double tmp=tmpvector[j][i];
					embedding[i][j]= tmp;

					if (abs(tmp) > maxembedding) maxembedding=tmp;					
				}
			}

			double expansion=double(maxDimLowDSpace)/maxembedding;

			// Also adding a noise as prescribed in https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1040
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator (seed);
			std::normal_distribution<double> distribution (0.0,1.0);					

			for (int i = 0; i < N; ++i) {			
				for (int j = 0; j < DimLowSpace; ++j) { 			
					embedding[i][j] =embedding[i][j]* expansion+ SCALE*distribution(generator);
				}
			}

		} catch(std::exception& e){
			logFile<<" Spectral Initialization Failed. Will proceed with random initialization."<<endl; 
			cout<<" Spectral Initialization Failed. Will proceed with random initialization."<<endl; 
			randominitializing=true ; }
	}
	/**
	 * If the above procedure fails or randominitializing=1 as an input argument, the
	 * location of the points are determined randomly 
	 */
	if (randominitializing){
		logFile<<" Random Initialization of Data in low-D Space"<<endl;
		cout<<" Random Initialization of Data in low-D Space"<<endl;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < DimLowSpace; ++j) {
				double tmp=(double)rand()/RAND_MAX;
				embedding[i][j]=minDimLowDSpace+(maxDimLowDSpace-minDimLowDSpace)*tmp;				
			}
		}
	}

	return;
}
