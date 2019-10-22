/**
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * October 2019
 * This is the Implementation of K-NN Algorithm in Distributed Systems as developed 
 * in "PANDA: Extreme Scale Parallel K-Nearest Neighbor on Distributed Architectures", Patwary et a., 2016
 */

#include <iostream>
#include <string>
#include <fstream>
#include <mpi.h>
#include <math.h>
#include <vector>
#include <stack>
#include <boost/iostreams/device/mapped_file.hpp> 
#include <boost/iostreams/stream.hpp>             
#include <set>
#include <omp.h>

using namespace std;
/**
 * Read the output of linux command execution 
 * @param  cmd  is the inux command to be executed
 * @return the output from the execution of the linux command
 */
std::string exec(const char* cmd) {
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}
/**
 * Defining the criteria for Sorting the data in a pair container from the biggest value to the smallest
 */
bool sortinrev(const pair<double,int> &a,const pair<double,int> &b) { 
	return (a.first > b.first); 
} 
/**
 * Compute the variance of a sampled data over data dimensions and Sort dimensions according to their variability
 * @param  DataCounts  Number of total data from which we take the samples
 * @paramn  odeData0 Dataset containing the data available for sampling
 * @param   featureCounts Number of features in dataset (equal to number of columns in the input csv file)
 * @param   world_size  Total number of MPI processors
 * @param	globalKdTreeSamples Number of Samples from dataset for computation here 	 
 * @return VectorGlobalSqrtSum A sorted pair containig the index of the dimensions with the highest variability 
 */
auto findMaxVarDims(int DataCounts,double **nodeData0, int featureCounts, int world_size, int globalKdTreeSamples) {
	double samplingData[globalKdTreeSamples][featureCounts];
	double localSum[featureCounts], globalSum[featureCounts];
	double localSqrtSum[featureCounts], globalSqrtSum[featureCounts];	

	for (int j=0; j<featureCounts; ++j){
		localSum[j]=0;
		localSqrtSum[j]=0;
	}

	for (int i=0; i< globalKdTreeSamples; ++i){
		int randomIndex=rand()%DataCounts;
		for (int j=0; j<featureCounts; ++j){
			samplingData[i][j]=nodeData0[randomIndex][j];
			localSum[j]+=samplingData[i][j];	
		}
	}	

	MPI_Allreduce(localSum,globalSum,featureCounts,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);		

	for (int i=0; i< globalKdTreeSamples; ++i){
		for (int j=0; j<featureCounts; ++j){
			localSqrtSum[j]+=pow((samplingData[i][j]-(globalSum[j]/world_size/globalKdTreeSamples)),2) ;
		}
	}

	MPI_Allreduce(localSqrtSum,globalSqrtSum,featureCounts,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	vector<pair<double,int>> VectorGlobalSqrtSum;

	for (int j=0; j<featureCounts; ++j){
		VectorGlobalSqrtSum.push_back(make_pair(globalSqrtSum[j],j));
	}
	sort(VectorGlobalSqrtSum.begin(), VectorGlobalSqrtSum.end(),sortinrev);
	return VectorGlobalSqrtSum;			
}
/**
 * Compute the distance between 2 data points within the same bucket
 * @param  index Index of the first data point
 * @paramn  index2 Index of the second data point
 * @param   mappedData2 2D array containing dataset	owned by each processor	 
 * @param   featureCounts Number of features in dataset (equal to number of columns in the input csv file)
 * @return sqrt(dist) The distance between 2 data points within the same bucket
 */
double computeDistance (int index,int index2, double** mappedData2, int featureCounts){		
	double dist=0;
	for (int i=0; i<featureCounts; ++i){
		double differences=mappedData2[index][i]-mappedData2[index2][i];
		dist+=differences*differences;
	}
	return sqrt(dist);	
}
/**
 * Compute the distance between 2 data points during querying
 * @param  index Index of the first data point
 * @paramn  i Index of the processor that has sent query
 * @param   jj Beginning index of the desired data point in the received data from the querying processor
 * @param   mappedData2 2D array containing dataset	owned by the current processor	 	 
 * @param   receivingPointCoordinates 2D array containing data received from the querying processors	 
 * @param   featureCounts Number of features in dataset (equal to number of columns in the input csv file)
 * @return sqrt(dist) The distance between 2 data points 
 */	
double computeDistance2 (int index, int i, int jj, double** mappedData2, double** receivingPointCoordinates,int featureCounts ){	
	double dist=0;
	for (int k=0; k<featureCounts; ++k){
		double differences= mappedData2[index][k]-receivingPointCoordinates[i][jj+k];
		dist+=differences*differences;
	}
	return sqrt(dist);	
}
/**
 * Compute the median of data at a dividing node of the global Kd Tree
 * @param  maxVarDimension Index of the chosen dimension for computing median
 * @paramn  nodeDataIndex0 Vector containing the indices of data available at the dividing node
 * @param   globalKdTreeSamplesMedian Number of data sampled by each processor to collaboratively compute the median at the dividing node of the global Kd tree
 * @param   Epsilon The acceptable buffer in estimating the median	 
 * @param   world_size Total number of MPI processors
 * @param   world_rank Rank of each MPI processor
 * @param	data 2D array containing the datapoint coordinates owned by each processor
 * @return  MedianCandidate The estimated value of median at the dividing node 
 */		
double globalFindMedian (int maxVarDimension, vector<int> nodeDataIndex0, int globalKdTreeSamplesMedian, double Epsilon, int world_size, int world_rank, double** data) {	
	int randomIndex;	
	vector <double> sampledDataValues, leftSampledDataValues, rightSampledDataValues;
	sampledDataValues.reserve(globalKdTreeSamplesMedian);
	leftSampledDataValues.reserve(globalKdTreeSamplesMedian);
	rightSampledDataValues.reserve(globalKdTreeSamplesMedian);

	for (int i=0; i< globalKdTreeSamplesMedian; ++i){
		randomIndex=rand()%nodeDataIndex0.size();
		int index=nodeDataIndex0[randomIndex];  
		sampledDataValues.push_back(data[index][maxVarDimension]);
	}

	int randomRank;
	double MedianCandidate;
	int totalCountsData=world_size*globalKdTreeSamplesMedian;
	int accumulatedLeftCounts=0;
	bool whileFlag=true;
	int whileCount=0;

	while(whileFlag){
		if (world_rank==0) {randomRank=rand()%world_size;}
		MPI_Bcast(&randomRank,1,MPI_INT,0,MPI_COMM_WORLD);       

		if (world_rank==randomRank) {
			randomIndex=rand()%sampledDataValues.size();
			MedianCandidate=sampledDataValues[randomIndex];
		}
		MPI_Bcast(&MedianCandidate,1,MPI_DOUBLE,randomRank,MPI_COMM_WORLD);

		int leftCounts=0; int rightCounts=0;int globalleftCounts=0;
		leftSampledDataValues.clear();
		rightSampledDataValues.clear();

		for (int i=0; i<sampledDataValues.size() ; ++i){
			if (sampledDataValues[i] < MedianCandidate) {
				leftSampledDataValues.push_back(sampledDataValues[i]);
				++leftCounts;
			}
			else{ 
				rightSampledDataValues.push_back(sampledDataValues[i]);
				++rightCounts;        
			}
		}

		MPI_Allreduce(&leftCounts,&globalleftCounts,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		globalleftCounts+=accumulatedLeftCounts;
		double ratio= double(globalleftCounts)/totalCountsData;

		if ( ratio < 0.5+Epsilon && ratio > 0.5-Epsilon ) {
			whileFlag=false;
			return MedianCandidate ;}
		else if (ratio < 0.5-Epsilon){
			accumulatedLeftCounts=globalleftCounts;
			sampledDataValues.clear();
			sampledDataValues=rightSampledDataValues;    
			whileFlag=true;
		}
		else if (ratio > 0.5+Epsilon){
			whileFlag=true;
		}
		++whileCount;
		if (whileCount/10000*10000 == whileCount) cout<< "Too Many Trials for Global KD Tree Median, Processor =  = "<<world_rank<<endl;
	}
}
/**
 * Compute the median of data at a dividing node of the local Kd Tree
 * @param  localKdTreeSamplesMedian Number of samples used to compute the median
 * @paramn  sampledDataValues The coordinates of the sampled data
 * @param   Epsilon The acceptable buffer in estimating the median	 
 * @param   world_rank Rank of each MPI processor
 * @return  MedianCandidate The estimated value of median at the dividing node 
 */		
double localFindMedian (int localKdTreeSamplesMedian,vector<double> sampledDataValues, double Epsilon, int world_rank) {	
	vector <double> leftSampledDataValues, rightSampledDataValues;
	leftSampledDataValues.reserve(localKdTreeSamplesMedian);
	rightSampledDataValues.reserve(localKdTreeSamplesMedian);

	int accumulatedLeftCounts=0;
	bool whileFlag=true;
	int whileCount=0;

	while(whileFlag){    
		int randomIndex=rand()%sampledDataValues.size();
		double MedianCandidate=sampledDataValues[randomIndex];	
		int leftCounts=0;
		int rightCounts=0;
		leftSampledDataValues.clear();
		rightSampledDataValues.clear();

		for (int i=0; i<sampledDataValues.size() ; ++i){
			if (sampledDataValues[i] < MedianCandidate) {
				leftSampledDataValues.push_back(sampledDataValues[i]);
				++leftCounts;
			}
			else{ 
				rightSampledDataValues.push_back(sampledDataValues[i]);
				++rightCounts;        
			}
		}
		leftCounts+=accumulatedLeftCounts;
		double ratio= double(leftCounts)/localKdTreeSamplesMedian;

		if ( ratio < 0.5+Epsilon && ratio > 0.5-Epsilon ) {
			whileFlag=false;	
			return MedianCandidate;
		}
		else if (ratio < 0.5-Epsilon){
			accumulatedLeftCounts=leftCounts;
			sampledDataValues.clear();
			sampledDataValues=rightSampledDataValues;    
			whileFlag=true;
		}
		else if (ratio > 0.5+Epsilon){
			whileFlag=true;
		}
		++whileCount;

		if (whileCount/10000*10000==whileCount) {
			if (Epsilon<0.25) Epsilon*=2; 
			else return MedianCandidate;
		}
	}
}
/**
 * Sort the max-heap data structure for a new data inserted at its index i 
 * @param  ID The ID of the point data
 * @paramn  i Index of the inserted data in the Heap 
 * @param   KNNDistanceinBuckets The values of distances for selected K-NNs    	 
 * @param   KNNIDsinBuckets The IDs of the selected K-NNs
 * @param   KNNCounts  Desired count of K-NNs to be computed in this program	  
 */			
void Max_Heapify(int ID, int i, double ** KNNDistanceinBuckets, int ** KNNIDsinBuckets,int KNNCounts) {
	int largest = 0;
	int l = 2*i + 1; 
	int r = 2*i + 2;

	if ((l < KNNCounts) && (KNNDistanceinBuckets[ID][l] > KNNDistanceinBuckets[ID][i])) {
		largest = l;
	}
	else {
		largest = i;
	}

	if ((r < KNNCounts) && (KNNDistanceinBuckets[ID][r] > KNNDistanceinBuckets[ID][largest])) {
		largest = r;
	}

	if (largest != i) {
		std::swap(KNNDistanceinBuckets[ID][i], KNNDistanceinBuckets[ID][largest]);
		std::swap(KNNIDsinBuckets[ID][i], KNNIDsinBuckets[ID][largest]);
		Max_Heapify(ID, largest, KNNDistanceinBuckets, KNNIDsinBuckets,KNNCounts);
	}
}
/**
 * Build Max-Heap datat structure for the first time
 * @param  ID The ID of the point data
 * @param   KNNCounts  Desired count of K-NNs to be computed in this program
 * @param   KNNDistanceinBuckets The values of distances for selected K-NNs    	 
 * @param   KNNIDsinBuckets The IDs of the selected K-NNs	  
 */	
void Build_Max_Heap(int ID,int KNNCounts, double** KNNDistanceinBuckets, int** KNNIDsinBuckets) {
	for (int i = floor((KNNCounts - 1) / 2); i >= 0; i--) {
		Max_Heapify(ID, i,KNNDistanceinBuckets, KNNIDsinBuckets,KNNCounts);
	}
}
/**
 * Sort the max-heap data structure for a newly inserted data
 * @param   k The index of the inserted point data
 * @param   receivingHeapArrayDistances2DCopy The values of distances for selected K-NNs    	 
 * @param   receivingHeapArray2DCopy The IDs of the selected K-NNs
 * @param   KNNCounts  Desired count of K-NNs to be computed in this program	  
 */		
void Max_Heapify2 (int k, double * receivingHeapArrayDistances2DCopy, int * receivingHeapArray2DCopy,int KNNCounts) {
	int largest = 0;
	int l = 2*k + 1; 
	int r = 2*k + 2;

	if ((l < KNNCounts) && (receivingHeapArrayDistances2DCopy[l] > receivingHeapArrayDistances2DCopy[k])) {
		largest = l;
	}
	else {
		largest = k;
	}

	if ((r < KNNCounts) && (receivingHeapArrayDistances2DCopy[r] > receivingHeapArrayDistances2DCopy[largest])) {
		largest = r;
	}

	if (largest != k) {
		std::swap(receivingHeapArrayDistances2DCopy[k], receivingHeapArrayDistances2DCopy[largest]);
		std::swap(receivingHeapArray2DCopy[k], receivingHeapArray2DCopy[largest]);
		Max_Heapify2(largest,receivingHeapArrayDistances2DCopy,receivingHeapArray2DCopy,KNNCounts);
	}
}
/**
 * Build Max-Heap datat structure for the first time
 * @param   KNNCounts  Desired count of K-NNs to be computed in this program
 * @param   receivingHeapArrayDistances2DCopy The values of distances for selected K-NNs    	 
 * @param   receivingHeapArray2DCopy The IDs of the selected K-NNs	  
 */	
void Build_Max_Heap2(int KNNCounts, double* receivingHeapArrayDistances2DCopy, int* receivingHeapArray2DCopy) {
	for (int ii = floor((KNNCounts - 1) / 2); ii >= 0; ii--) {
		Max_Heapify2(ii,receivingHeapArrayDistances2DCopy,receivingHeapArray2DCopy,KNNCounts);
	}
}
/**
 * Main Function of the Code
 */			
int main(int argc, char * const argv[]) {
	/**
	 * The three following arguments are passed to the code (in order) from the command line:
	 * fileName is the full path to the input csv dataset
	 * featureCounts is the number of columns in the input csv datastet (number of data dimensions)
	 * KNNCounts is the number of K-NNs for each data point to be computed in this program
	 */	
	string fileName = argv[1]; 
	const int featureCounts = atoi(argv[2]); 	
	const int KNNCounts = atoi(argv[3]); 
	/**	
	 * The following important parameters are used in the design of algorithm. Their values are
	 * initialized according to the suggested values in the referencing paper.
	 * globalKdTreeSamples is the number of data sampled by each processor to collaboratively compute dimensions with the highest variability.
	 * globalKdTreeSamplesMedian is the number of data sampled by each processor to collaboratively compute the median of the chosen dimension for each splitting node within the global Kd Tree.
	 * localKdTreeSamplesMedian is the number of data sampled by each processor separately to compute the median of the chosen dimension for each splitting node within the local Kd Tree.
	 * Epsilon is a buffer in accepting the Median value
	 * Parallel_IO is a flag that defines if the input csv file can be read in parallel by all the processors
	 * bucketSize is the size of a bucket (or a leaf) in the local Kd Tree
	 * estimatedExtraLayers: To limit the growing size of the local Kd Trees, the growth of the tree is limited by a number of layers defined here from the initial guess of the required buckets
	 */		
	const int globalKdTreeSamples=256;	
	const int globalKdTreeSamplesMedian=256;
	int localKdTreeSamplesMedian=1024;
	double Epsilon=0.01; 
	const int Parallel_IO = 1; 
	const int bucketSize=32;
	const int estimatedExtraLayers=1;
	/**	
	 * The precision of cout outputs are defined here
	 */	
	cout.precision(17);
	/**	
	 * Seed for random number generation
	 */	
	srand(17);	
	/**	
	 * Beginning MPI communications
	 */			
	MPI_Init(NULL, NULL);
	/**	
	 * world_size is defined here as total number of MPI processors
	 */	
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	/**	
	 * world_rank is defined here as the rank of MPI processors
	 */	
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	/**	
	 * total number of MPI processors should be a power of 2 due to algorithm design for global Kd Tree.
	 * Otherwise, output an error and exit the program
	 */	
	bool powerOfTwo = !(world_size == 0) && !(world_size & (world_size - 1));
	if (powerOfTwo!=true) {
		if (world_rank==0) cout <<"Number of Processors should be a power of 2"<<endl;
		MPI_Finalize();
		return 0;
	}
	/**	
	 * The master processor splits the input csv file as each processor could have its own non-overlapping set of input data
	 */
	if (world_rank==0) {
		string cmd=string("split -n l/")+to_string(world_size)+" "+ fileName+" -d tmpFile --additional-suffix=.csv"; 
		system(cmd.c_str());
	}
	/**	
	 * All procesors neeed to stop here until master processor returns
	 */
	MPI_Barrier(MPI_COMM_WORLD);
	/**	
	 * Each processor read its own set of data from a unique csv file named localFileName
	 */
	string localFileName;
	if (world_size <11) localFileName="tmpFile0"+to_string(world_rank)+".csv";
	else if (world_size <101) {
		localFileName="tmpFile"+to_string(world_rank)+".csv";
		if (world_rank<10) localFileName="tmpFile0"+to_string(world_rank)+".csv";
	}	
	else if (world_size <1001){
		localFileName="tmpFile"+to_string(world_rank)+".csv";
		if (world_rank<10) localFileName="tmpFile00"+to_string(world_rank)+".csv";
		else if (world_rank>=10 && world_rank<100) localFileName="tmpFile0"+to_string(world_rank)+".csv";
	}
	else cout << "Error: Too Many Processes" << endl;

	ifstream infile;   
	infile.open(localFileName); 
	/**	
	 * Output error in case the localFileName was not opened for reading
	 */	
	if(infile.fail())  
	{ 
		cout << "error in opening the input file" << endl; 
		return 1; 
	} 
	/**	
	 * Each processor finds out about the number of records in its localFileName
	 */	
	string cmd3="wc -l "+localFileName;
	string outputCmd3 = exec(cmd3.c_str());
	int tmpFileLineCounts=stoi(outputCmd3.substr(0, outputCmd3.find(" ")));
	/**	
	 * The master node needs to subtract 1 record which is for header information
	 */	
	if (world_rank==0) {
		string dummyLine;
		getline(infile, dummyLine);
		--tmpFileLineCounts;
	} 
	/**	
	 * MPI communication between the processors as they all need to know how many data the other processors have
	 */
	int tmpFileLineCountsArray[world_size], tmpFileLineCountsArrayCum[world_size] ;
	int sendBuffer0[0];
	sendBuffer0[0]=tmpFileLineCounts;
	MPI_Allgather(sendBuffer0,1,MPI_INT,tmpFileLineCountsArray,1,MPI_INT,MPI_COMM_WORLD);
	/**	
	 * All Processors make an array tmpFileLineCountsArrayCum that cummulatively stores the number of data in the other processors 
	 */
	for (int i=0; i<world_size; ++i) {
		tmpFileLineCountsArrayCum[i]=0;
	}

	for (int i=0; i<world_size; ++i) {	
		for (int j=0; j<i+1; ++j) {
			tmpFileLineCountsArrayCum[i]+=tmpFileLineCountsArray[j];
		}
	}
	/**	
	 * Parse data from csv file and store them in a 2D array
	 */
	double ** inputdata= new double*[tmpFileLineCounts];;
	for (int i=0; i<tmpFileLineCounts; ++i) { inputdata[i] = new double[featureCounts]; }

	for (int i=0; i<tmpFileLineCounts; ++i) {
		string temp, temp2;
		getline(infile, temp);	
		for (int j=0; j<featureCounts; ++j){
			temp2 =temp.substr(0, temp.find(","));
			inputdata[i][j]=atof(temp2.c_str());
			temp.erase(0, temp.find(",") + 1);
		}
	}
	/**	
	 * Remove the local input files as their data has been already parsed and read
	 */
	infile.close();
	string cmd2= string("rm ")+localFileName;
	system(cmd2.c_str());
	/**	
	 * Compute dimensions with the highest variance
	 */	
	vector<pair<double,int>> VectorGlobalSqrtSum;
	vector <int> nodeDataIndex[world_size];
	nodeDataIndex[0].reserve(tmpFileLineCounts);
	for (int i=0; i<tmpFileLineCounts; ++i){nodeDataIndex[0].push_back(i);}		
	VectorGlobalSqrtSum=findMaxVarDims(nodeDataIndex[0].size(), inputdata, featureCounts, world_size, globalKdTreeSamples);	
	/**	
	 * Constructing the global Kd Tree collaboratively by all the processors
	 */		
	vector<double> globalMedianValuesforNodes;
	vector <int> nextLayerNodeDataIndex[world_size];
	int nodeCounts=1, nodesLayer=0;	
	double medianNodeData;

	while (nodeCounts!= world_size){ 
		if (world_rank ==0) cout << "Constructing Global Kd Tree: Layer =" << nodesLayer<< endl;
		int indexMaxVarDim=VectorGlobalSqrtSum[nodesLayer].second; 

		for (int i=0; i<nodeCounts; ++i){
			int countLeft=0, countRight=0;
			medianNodeData=globalFindMedian(indexMaxVarDim,nodeDataIndex[i], globalKdTreeSamplesMedian,Epsilon, world_size,world_rank, inputdata);						
			globalMedianValuesforNodes.push_back(medianNodeData); 

			for (int j=0; j< nodeDataIndex[i].size(); ++j){ 
				int index=nodeDataIndex[i][j]; 
				if (inputdata[index][indexMaxVarDim] < medianNodeData){ 
					nextLayerNodeDataIndex[i*2].push_back(index);
					++countLeft;
				}
				else{
					nextLayerNodeDataIndex[i*2+1].push_back(index);
					++countRight;
				}   
			}
		}
		nodeCounts*=2;
		++nodesLayer;

		for (int i=0; i<nodeCounts; ++i){
			nodeDataIndex[i].clear();
			nodeDataIndex[i]=nextLayerNodeDataIndex[i];	
			nextLayerNodeDataIndex[i].clear();	
		}
	}		
	/**	
	 * Once the number of dividing nodes in the global Kd Tree became equal to the number of MPI processors
	 * each processor will be responsible for the data of one dividing node
	 * Index of data for each processor is stored at ProcessorLocalDataIndex
	 */	
	int *ProcessorLocalDataIndex;
	int cnts; 

	for (int i=0; i<nodeCounts; ++i){
		int rcount[world_size];
		int send_buffer[0];
		int displs[nodeCounts];
		displs[0]=0;	
		int myDATA[nodeDataIndex[i].size()];

		for (int j=0; j< nodeDataIndex[i].size(); ++j){ 
			if (world_rank>0) myDATA[j]= nodeDataIndex[i][j]+tmpFileLineCountsArrayCum[world_rank-1];
			else if(world_rank==0) myDATA[j]= nodeDataIndex[i][j];
		}  

		int Totalcounts=(int)nodeDataIndex[i].size();			            			  
		send_buffer[0]=nodeDataIndex[i].size();								
		MPI_Gather(send_buffer,1, MPI_INT,rcount,1, MPI_INT,i,MPI_COMM_WORLD);

		if (world_rank==i){	
			cnts=0;
			for (int k=0; k<nodeCounts; ++k){cnts+=rcount[k];}			
			for (int k=1; k<nodeCounts; ++k){displs[k]=displs[k-1]+rcount[k-1];}
			ProcessorLocalDataIndex = new int[cnts]; 
		}	
		MPI_Gatherv(myDATA,Totalcounts,MPI_INT,ProcessorLocalDataIndex,rcount,displs,MPI_INT,i,MPI_COMM_WORLD);     		 
	}
	/**	
	 * Now, each processor only reads its own data from the input csv file according to the indices of ProcessorLocalDataIndex
	 * If parallel I/O is not available (Parallel_IO=0), each processor reads the file at a time
	 * the main output of this section is mappedData which is a 2D array storing dataset
	 */
	int rankOfProcess=0;
	double mappedData[cnts][featureCounts];  
	int indexLookupArray[tmpFileLineCountsArrayCum[world_size-1]];

	if (Parallel_IO){
		using boost::iostreams::mapped_file_source;
		using boost::iostreams::stream;
		mapped_file_source mmap(fileName);
		stream<mapped_file_source> is(mmap, std::ios::binary);
		string tempString,tempString2;
		int m_numLines = 0;
		string dummyLine;
		getline(is, dummyLine);    

		for (int i=0; i<cnts; ++i){      
			int lineIndex=ProcessorLocalDataIndex[i];
			bool flag=true;

			while (flag==true){
				if (m_numLines==lineIndex) {  
					indexLookupArray[lineIndex]=i; 
					getline(is, tempString);      
					for (int j=0;j<featureCounts;++j){
						tempString2 =tempString.substr(0, tempString.find(","));
						mappedData[i][j]=atof(tempString2.c_str());
						tempString.erase(0, tempString.find(",") + 1);
					}
					m_numLines++;
					flag=false;
				}
				else {
					getline(is, dummyLine);  
					m_numLines++;
					flag=true;
					if(!is) {flag=false; break;}
				}				
			}
		}
		mmap.close();
	}
	else{
		while (rankOfProcess < world_size){
			if (world_rank==rankOfProcess){
				using boost::iostreams::mapped_file_source;
				using boost::iostreams::stream;
				mapped_file_source mmap(fileName);
				stream<mapped_file_source> is(mmap, std::ios::binary);
				string tempString,tempString2;
				int m_numLines = 0;
				string dummyLine;
				getline(is, dummyLine);    

				for (int i=0; i<cnts; ++i){      
					int lineIndex=ProcessorLocalDataIndex[i];
					bool flag=true;

					while (flag==true){
						if (m_numLines==lineIndex) {  
							indexLookupArray[lineIndex]=i; 
							getline(is, tempString);      
							for (int j=0;j<featureCounts;++j){
								tempString2 =tempString.substr(0, tempString.find(","));
								mappedData[i][j]=atof(tempString2.c_str());
								tempString.erase(0, tempString.find(",") + 1);
							}
							m_numLines++;
							flag=false;
						}
						else {
							getline(is, dummyLine);  
							m_numLines++;
							flag=true;
							if(!is) {flag=false; break;}
						}				
					}
				}
				mmap.close();
			}
			++rankOfProcess;
			MPI_Barrier(MPI_COMM_WORLD);
		}  
	}
	/**	
	 * Now, it is the time to construct the local Kd Tree by each processor separately
	 * Tree construction continues until all data is stored in the buckets of size bucketSize
	 * or maxAllowedLayers is reached 
	 */
	if (world_rank==0) cout << "Constructing the Local Kd Tree"<<endl;

	int layerNodeCounts=1;  
	int localNodesLayer=0;
	vector < vector<int> > localNodeDataIndex;
	vector <int> tmpvector;
	vector <double> localMedianNodeData;
	vector <int> isBucket;
	bool localFlag=true;
	int numberofNodeSofar;
	int nodeIndexofaPoint[cnts];      
	tmpvector.reserve(cnts);

	for (int i=0; i<cnts; ++i){
		tmpvector.push_back(ProcessorLocalDataIndex[i]); 
		nodeIndexofaPoint[i]=-1;
	}
	localNodeDataIndex.push_back(tmpvector);		

	isBucket.reserve(localNodeDataIndex[0].size()/bucketSize);
	localMedianNodeData.reserve(localNodeDataIndex[0].size()/bucketSize);
	/**	
	 * Ideally we need estimatedLayers number of layers in the local Kd tree
	 */
	int estimatedLayers=int(log2(localNodeDataIndex[0].size()/bucketSize))+1;  
	int maxAllowedLayers=estimatedLayers+estimatedExtraLayers;
	if (maxAllowedLayers+nodesLayer > featureCounts) cout << "Error in Exceeding Dimensions, increase BucketSize"<<endl;    

	if (localNodeDataIndex[0].size() <= bucketSize) {isBucket.push_back(1); localFlag=false;}
	else {isBucket.push_back(0);}	

	while (localFlag){
		int indexMaxVarDim=VectorGlobalSqrtSum[localNodesLayer+nodesLayer].second; 
		if (localNodesLayer==0) {numberofNodeSofar=0;}
		else {numberofNodeSofar=pow(2,localNodesLayer)-1;}

		for (int i=0; i<layerNodeCounts; ++i){
			int globalID=numberofNodeSofar +i;
			int countLeft=0, countRight=0;
			int leftNodeGlobalIndex=numberofNodeSofar+layerNodeCounts+(i*2);
			int rightNodeGlobalIndex=numberofNodeSofar+layerNodeCounts+(i*2)+1;
			localNodeDataIndex.push_back(std::vector<int>());
			localNodeDataIndex.push_back(std::vector<int>());

			if (isBucket[globalID]==1) {isBucket.push_back(0); isBucket.push_back(0); continue;}
			if (localNodeDataIndex[globalID].size()==0) {isBucket.push_back(0); isBucket.push_back(0); continue;}		
			if (localKdTreeSamplesMedian > localNodeDataIndex[globalID].size()/2) localKdTreeSamplesMedian=localNodeDataIndex[globalID].size()/2;			

			vector <double> sampledDataValues;	
			for (int i=0; i< localKdTreeSamplesMedian; ++i){
				int randomIndex=rand()%localNodeDataIndex[globalID].size();
				int index=localNodeDataIndex[globalID][randomIndex];			
				int index1=indexLookupArray[index];	         
				sampledDataValues.push_back(mappedData[index1][indexMaxVarDim]);
			}

			double temp=localFindMedian(localKdTreeSamplesMedian,sampledDataValues,Epsilon,world_rank);
			localMedianNodeData.push_back(temp);

			for (int j=0; j< localNodeDataIndex[globalID].size(); ++j){ 
				int index0=localNodeDataIndex[globalID][j];
				int index=indexLookupArray[index0];

				if (mappedData[index][indexMaxVarDim] < localMedianNodeData[globalID]){ 
					localNodeDataIndex[leftNodeGlobalIndex].push_back(index0);
					++countLeft;
				}
				else{
					localNodeDataIndex[rightNodeGlobalIndex].push_back(index0);
					++countRight;
				}   
			}

			if ((countLeft <= bucketSize && countLeft >0)|| ((localNodesLayer == maxAllowedLayers-1) && countLeft >0) ) {
				isBucket.push_back(1);			

				for (int j=0; j< localNodeDataIndex[leftNodeGlobalIndex].size(); ++j){ 
					int index0=localNodeDataIndex[leftNodeGlobalIndex][j];
					int index=indexLookupArray[index0];
					nodeIndexofaPoint[index]=leftNodeGlobalIndex;		
				} 
			}
			else {isBucket.push_back(0);}

			if ((countRight <= bucketSize && countRight >0) || ((localNodesLayer == maxAllowedLayers-1) && countRight >0) ) {
				isBucket.push_back(1);

				for (int j=0; j< localNodeDataIndex[rightNodeGlobalIndex].size(); ++j){ 
					int index0=localNodeDataIndex[rightNodeGlobalIndex][j];
					int index=indexLookupArray[index0];              
					nodeIndexofaPoint[index]=rightNodeGlobalIndex;		
				}
			}
			else {isBucket.push_back(0);}		
		}

		localFlag=false;
		for (int i=0; i<layerNodeCounts; ++i){
			int globalID=numberofNodeSofar + i;	    
			if (isBucket[globalID]==0 && localNodeDataIndex[globalID].size()>0 ) {localFlag=true; break;}
		}
		layerNodeCounts*=2;
		++localNodesLayer;	
	}
	/**	
	 * For performance, it is better to refer to local Kd tree later
	 * from the ID of the first dividing node which has been converted to a bucket
	 */
	int FirstBucket;
	for (int i=0; i< localNodeDataIndex.size(); ++i){ 
		if (isBucket[i] == 1) {FirstBucket=i;break;}
	}
	/**	
	 * Now, it is the time to start computing K-NNs from the data points within each bucket in the local Kd Tree
	 * and store them in KNNIDsinBuckets and KNNDistanceinBuckets
	 * To improve the performance, the data locality was considered for main arrays of localNodeDataIndex2 and mappedData2 
	 * and the data within the same bucket arranged close to each other in the new arrays
	 */
	if (world_rank==0) cout <<"Computing K-NNs for the points within the Same Bucket"<<endl;	

	int KNNIDsinBucketsFilledCounts[cnts];
	int localIndexConvertor[cnts];
	int counter=0;
	vector<vector<int>> localNodeDataIndex2;  

	int **KNNIDsinBuckets = new int*[cnts];
	for (int i=0; i<cnts; ++i) { KNNIDsinBuckets[i] = new int[KNNCounts]; }	

	double ** KNNDistanceinBuckets = new double*[cnts];
	for (int i=0; i<cnts; ++i) { KNNDistanceinBuckets[i] = new double[KNNCounts]; }	

	for (int i=0; i< localNodeDataIndex.size(); ++i){ 
		localNodeDataIndex2.push_back(std::vector<int>());	
		if (isBucket[i] == 0) {continue;}

		for (int j=0; j< localNodeDataIndex[i].size(); ++j){ 
			localIndexConvertor[counter]=localNodeDataIndex[i][j];
			localNodeDataIndex2[i].push_back(counter); 
			++counter;
		}
	}

	double** mappedData2=new double*[cnts];
	for (int i=0; i<cnts; ++i) { mappedData2[i] = new double[featureCounts]; }

	int nodeIndexofaPoint2[cnts];

	for (int i=0; i< cnts; ++i){ 
		int pointID=localIndexConvertor[i];
		int index=indexLookupArray[pointID]; 
		nodeIndexofaPoint2[i]=nodeIndexofaPoint[index];

		for (int j=0; j<featureCounts; ++j){
			mappedData2[i][j]=mappedData[index][j];
		}
	}

	for (int i=0; i< cnts; ++i){ 
		for (int j=0; j< KNNCounts; ++j){ 
			KNNIDsinBuckets[i][j]=-1;
		}
	}

	for (int i=0; i< cnts; ++i){ 
		KNNIDsinBucketsFilledCounts[i]=0;
	}

	for (int i=FirstBucket; i< localNodeDataIndex2.size(); ++i){ 
		if (isBucket[i] == 0) {continue;}

		for (int j=0; j< localNodeDataIndex2[i].size()-1; ++j){ 
			int index=localNodeDataIndex2[i][j];

			for (int k=j+1; k<localNodeDataIndex2[i].size(); ++k){ 
				int index2=localNodeDataIndex2[i][k];			
				int emptyIndex = KNNIDsinBucketsFilledCounts[index];
				double dist=computeDistance(index,index2,mappedData2,featureCounts); 

				if  (emptyIndex < KNNCounts) {
					KNNIDsinBuckets[index][emptyIndex]=localIndexConvertor[index2];                    
					++KNNIDsinBucketsFilledCounts[index];                      
					KNNDistanceinBuckets[index][emptyIndex]=dist;          
					if (emptyIndex==(KNNCounts-1)) Build_Max_Heap(index,KNNCounts,KNNDistanceinBuckets,KNNIDsinBuckets);
				}
				else { 
					if (dist < KNNDistanceinBuckets[index][0]) {         
						KNNIDsinBuckets[index][0]=localIndexConvertor[index2];                                         
						KNNDistanceinBuckets[index][0]=dist;     
						Max_Heapify(index, 0, KNNDistanceinBuckets, KNNIDsinBuckets,KNNCounts);
					}    
				}

				int emptyIndex2 = KNNIDsinBucketsFilledCounts[index2];
				if  (emptyIndex2 < KNNCounts) {
					KNNIDsinBuckets[index2][emptyIndex2]=localIndexConvertor[index];	                     
					++KNNIDsinBucketsFilledCounts[index2];                      
					KNNDistanceinBuckets[index2][emptyIndex2]=dist;       
					if (emptyIndex2==(KNNCounts-1)) Build_Max_Heap(index2,KNNCounts,KNNDistanceinBuckets,KNNIDsinBuckets);                        
				}
				else {
					if (dist < KNNDistanceinBuckets[index2][0]) {  
						KNNIDsinBuckets[index2][0]=localIndexConvertor[index];                                         
						KNNDistanceinBuckets[index2][0]=dist;
						Max_Heapify(index2, 0, KNNDistanceinBuckets, KNNIDsinBuckets, KNNCounts);}
				}
			}
		}
	}
	/**	
	 * Now, it is the time to find the IDs of processors that contain the neighboring sub-spaces 
	 * A neighboring processor is selected if its distance from the given point is less than 
	 * the maximum distance in the heap of that point (first entry of heap)
	 */
	if (world_rank==0) cout <<"Finding the Spatial Neighboring Processors"<<endl;	

	vector<int> ScatterVlocalNodeDataIndex[world_size];
	vector<int> ScatterVKNNIDsinBucketsFilledCounts[world_size];

	int globalLayerID = int(log2(world_size));
	int lowestNodeID=pow(2,globalLayerID)-1; 
	int highestNodeID=lowestNodeID+world_size-1;
	int NeighboringNodes[cnts][world_size-1];

	if (world_size != 1){ 
		for (int i=0; i<cnts; ++i){
			for (int j=0; j<world_size-1; ++j){
				NeighboringNodes[i][j]=-1;
			}
		}

		for (int i=FirstBucket; i< localNodeDataIndex2.size(); ++i){ 
			if (isBucket[i] == 0) {continue;}
			for (int j=0; j< localNodeDataIndex2[i].size(); ++j){ 
				int index1= localNodeDataIndex2[i][j];

				double rPrime=KNNDistanceinBuckets[index1][0];
				stack<pair<int,double>> globalStack;  
				globalStack.push(make_pair(0,0));
				/**	 
				 * C1NodeID is the closer child, and C2NodeID is the other child
				 */
				int C1NodeID,C2NodeID;
				int jcounts=0;

				while (!globalStack.empty()){
					pair<int,double> topPairinStack=globalStack.top();
					int nodeID=topPairinStack.first;
					double dValue=topPairinStack.second;
					globalStack.pop();
					int nodesLayer0=int(log2(nodeID+1));
					int indexMaxVarDim=VectorGlobalSqrtSum[nodesLayer0].second;

					if (dValue < rPrime){
						double dPrime= mappedData2[index1][indexMaxVarDim] - globalMedianValuesforNodes[nodeID];
						if (dPrime < 0) { 
							C1NodeID=2*nodeID+1; 
							C2NodeID=2*nodeID+2; 
						}
						else{
							C1NodeID=2*nodeID+2; 
							C2NodeID=2*nodeID+1; 
						}

						dPrime=sqrt(dValue*dValue+dPrime*dPrime);
						if (dPrime<rPrime) { 
							if (C2NodeID <= highestNodeID) {
								globalStack.push(make_pair(C2NodeID,dPrime));
								if (C2NodeID >= lowestNodeID && (C2NodeID-lowestNodeID)!=world_rank) {
									NeighboringNodes[index1][jcounts]=C2NodeID-lowestNodeID;
									ScatterVlocalNodeDataIndex[C2NodeID-lowestNodeID].push_back(index1);
									ScatterVKNNIDsinBucketsFilledCounts[C2NodeID-lowestNodeID].push_back(KNNIDsinBucketsFilledCounts[index1]);
									++jcounts;
								}
							}
						}

						if (C1NodeID <= highestNodeID) {
							globalStack.push(make_pair(C1NodeID,dValue));
							if (C1NodeID >= lowestNodeID && (C1NodeID-lowestNodeID)!=world_rank) {
								NeighboringNodes[index1][jcounts]=C1NodeID-lowestNodeID;
								ScatterVlocalNodeDataIndex[C1NodeID-lowestNodeID].push_back(index1);
								ScatterVKNNIDsinBucketsFilledCounts[C1NodeID-lowestNodeID].push_back(KNNIDsinBucketsFilledCounts[index1]);
								++jcounts;
							}
						}
					}
				}	
			} 
		}
	}	
	/**	
	 * Now, send the data of the given point to the neighboring processors identified above 
	 * for further computation of possible K-NNs in those processors
	 */
	int displ[world_size],displ2[world_size],displ3[world_size];
	int bufferCounts[world_size],bufferCounts2[world_size],bufferCounts3[world_size];
	bufferCounts[world_rank]=0;
	bufferCounts2[world_rank]=0;  
	bufferCounts3[world_rank]=0;

	if (world_size != 1){	
		for (int i=0; i<cnts; ++i){      
			ScatterVlocalNodeDataIndex[world_rank].push_back(i);      
			ScatterVKNNIDsinBucketsFilledCounts[world_rank].push_back(KNNIDsinBucketsFilledCounts[i]);					
		}

		for (int i=0; i<world_size; ++i){ 
			bufferCounts[i]=ScatterVlocalNodeDataIndex[i].size(); 
			bufferCounts2[i]=ScatterVlocalNodeDataIndex[i].size()*KNNCounts;
			bufferCounts3[i]=ScatterVlocalNodeDataIndex[i].size()*featureCounts;
		}

		displ[0]=0;
		displ2[0]=0;
		displ3[0]=0;
		for (int i=1; i<world_size; ++i){
			displ[i]= displ[i-1]+bufferCounts[i-1];
			displ2[i]= displ2[i-1]+bufferCounts2[i-1];
			displ3[i]= displ3[i-1]+bufferCounts3[i-1];
		}
	}

	const int ArraySizeScatterV=displ[world_size-1]+bufferCounts[world_size-1];
	int sendbuffer[ArraySizeScatterV];
	int sendbuffer2[ArraySizeScatterV];
	int sendbuffer4[ArraySizeScatterV*KNNCounts];
	double sendbuffer5[ArraySizeScatterV*featureCounts];
	double sendbuffer6[ArraySizeScatterV*KNNCounts];

	if (world_size != 1){		
		for (int i=0; i<world_size; ++i){
			int KIndex=displ[i];	
			for (int j=0; j<ScatterVlocalNodeDataIndex[i].size(); ++j){
				sendbuffer[KIndex+j]=ScatterVlocalNodeDataIndex[i][j]; 
				sendbuffer2[KIndex+j]=ScatterVKNNIDsinBucketsFilledCounts[i][j];
				for (int kk=0; kk<KNNCounts; ++kk){
					sendbuffer4[(KIndex+j)*KNNCounts+kk]= KNNIDsinBuckets[ScatterVlocalNodeDataIndex[i][j]][kk];
					sendbuffer6[(KIndex+j)*KNNCounts+kk]= KNNDistanceinBuckets[ScatterVlocalNodeDataIndex[i][j]][kk];
				}
				for (int ll=0; ll<featureCounts; ++ll){
					sendbuffer5[(KIndex+j)*featureCounts+ll]= mappedData2[ScatterVlocalNodeDataIndex[i][j]][ll];
				}
			}
		}
	}

	int receivingCountsMatrix[world_size];
	int receiveCounts,TotalReceiveCounts=0;
	int *receivingIndices[world_size];
	int *receivingHeapSize[world_size];
	int *receivingHeapArray[world_size];
	double *receivingHeapArrayDistances[world_size];
	double *receivingPointCoordinates[world_size];

	if (world_size != 1){	
		for (int i=0; i<world_size; ++i){
			MPI_Scatter (bufferCounts,1,MPI_INT,&receiveCounts,1 ,MPI_INT,i,MPI_COMM_WORLD); 
			receivingIndices[i]=new int[receiveCounts];  
			receivingHeapSize[i]=new int[receiveCounts];  
			receivingHeapArray[i]=new int[receiveCounts*KNNCounts];
			receivingHeapArrayDistances[i]=new double[receiveCounts*KNNCounts];
			receivingPointCoordinates[i]=new double[receiveCounts*featureCounts];
			receivingCountsMatrix[i]=receiveCounts;
			TotalReceiveCounts+=receiveCounts; 

			MPI_Scatterv (&sendbuffer ,bufferCounts, displ, MPI_INT,&receivingIndices[i][0],receiveCounts,MPI_INT,i,MPI_COMM_WORLD); 
			MPI_Scatterv (&sendbuffer2,bufferCounts, displ, MPI_INT,&receivingHeapSize[i][0],receiveCounts,MPI_INT,i,MPI_COMM_WORLD); 
			MPI_Scatterv (&sendbuffer4,bufferCounts2,displ2,MPI_INT,&receivingHeapArray[i][0],receiveCounts*KNNCounts,MPI_INT,i,MPI_COMM_WORLD); 
			MPI_Scatterv (&sendbuffer5,bufferCounts3,displ3,MPI_DOUBLE,&receivingPointCoordinates[i][0],receiveCounts*featureCounts,MPI_DOUBLE,i,MPI_COMM_WORLD); 
			MPI_Scatterv (&sendbuffer6,bufferCounts2,displ2,MPI_DOUBLE,&receivingHeapArrayDistances[i][0],receiveCounts*KNNCounts,MPI_DOUBLE,i,MPI_COMM_WORLD); 
		}
	}
	else{
		receivingIndices[0]=new int[cnts]; 
		receivingHeapArray[0]=new int[cnts*KNNCounts];	
		receivingHeapArrayDistances[0]=new double[cnts*KNNCounts];	
		receivingPointCoordinates[0]=new double[cnts*featureCounts];
		receivingCountsMatrix[0]=cnts;
		receivingHeapSize[0]=new int[cnts];

		for (int i=0; i<cnts; ++i){      
			receivingIndices[0][i]=i;
			receivingHeapSize[0][i]=KNNIDsinBucketsFilledCounts[i];

			for (int j=0; j<KNNCounts; ++j){ 
				receivingHeapArrayDistances[0][i*KNNCounts+j]=KNNDistanceinBuckets[i][j];
				receivingHeapArray[0][i*KNNCounts+j]=KNNIDsinBuckets[i][j];
			}

			for (int j=0; j<featureCounts; ++j){    
				receivingPointCoordinates[0][i*featureCounts+j]=mappedData2[i][j];
			}
		}
	}
	delete[] ProcessorLocalDataIndex;
	/**	
	 * Now, follow querying to compute possible K-NNs for each given point
	 * For each point, querying is performed on the local Kd Tree of its hosting processor as well as 
	 * the local Kd Tree of the neighboring processors identified above
	 * This section is the implementation of Algorithm 1 in the referencing paper and is computationally the most expensive part of the code
	 */
	if (world_rank==0) cout<<"Beginning Algorithm-1 of the Paper"<<endl;

	/**	 
	* C1NodeID is the closer child, and C2NodeID is the other child
	*/
	int C1NodeID,C2NodeID;

	for (int i=0; i<world_size; ++i){
	    /**	
	     * To improve the performance, multi-threading using OpenMP is implemented here
	     */
        #pragma omp parallel for private(C1NodeID,C2NodeID)
		for (int j=0; j<receivingCountsMatrix[i]; ++j){

			int tmpReceivingHeapSize=receivingHeapSize[i][j];
			double rPrimeValue=receivingHeapArrayDistances[i][j*KNNCounts];

			double * receivingHeapArrayDistances2DCopy=new double[KNNCounts];
			int * receivingHeapArray2DCopy=new int[KNNCounts];
	        /**	
	        * To improve the performance, 1D arrays receivingHeapArray2DCopy and receivingHeapArrayDistances2DCopy are used here
	        */
			for (int k=0; k<KNNCounts; ++k){
				receivingHeapArray2DCopy[k]=receivingHeapArray[i][j*KNNCounts+k];
				receivingHeapArrayDistances2DCopy[k]=receivingHeapArrayDistances[i][j*KNNCounts+k];
			}

			int tmpIsHeapChanged=0;
			stack<pair<int,double>> globalStack; 
			globalStack.push(make_pair(0,0)); 

			while (!globalStack.empty()){
				pair<int,double> topPairinStack=globalStack.top();
				int nodeID=topPairinStack.first;
				double dValue=topPairinStack.second;
				globalStack.pop();
				int nodesLayer0=int(log2(nodeID+1));
				int indexMaxVarDim=VectorGlobalSqrtSum[nodesLayer0+nodesLayer].second;

				if (isBucket[nodeID] == 1) {
					for (int kk=0; kk<localNodeDataIndex2[nodeID].size(); ++kk){ 					

						if (i==world_rank) {
							int index=receivingIndices[i][j];
							if (nodeIndexofaPoint2[index]==nodeID) break;
						}

						int index= localNodeDataIndex2[nodeID][kk];
						double distance=computeDistance2(index,i,j*featureCounts,mappedData2,receivingPointCoordinates,featureCounts);

						if (distance < rPrimeValue){
							if (tmpReceivingHeapSize < KNNCounts){
								receivingHeapArray2DCopy[tmpReceivingHeapSize]=localIndexConvertor[index];
								receivingHeapArrayDistances2DCopy[tmpReceivingHeapSize]=distance;
								++tmpReceivingHeapSize;
								tmpIsHeapChanged=1;     
								if(tmpReceivingHeapSize==KNNCounts) {
									Build_Max_Heap2(KNNCounts,receivingHeapArrayDistances2DCopy,receivingHeapArray2DCopy);
									rPrimeValue=receivingHeapArrayDistances2DCopy[0];
								}     
							}
							else if (distance < receivingHeapArrayDistances2DCopy[0]){
								receivingHeapArrayDistances2DCopy[0]=distance;
								receivingHeapArray2DCopy[0]=localIndexConvertor[index];							
								Max_Heapify2(0,receivingHeapArrayDistances2DCopy,receivingHeapArray2DCopy,KNNCounts);
								tmpIsHeapChanged=1;
								rPrimeValue=receivingHeapArrayDistances2DCopy[0];
							}                
						}
					}
				}
				else {
					if (dValue < rPrimeValue){
						double dPrime= receivingPointCoordinates[i][j*featureCounts+indexMaxVarDim] - localMedianNodeData[nodeID];
						if (dPrime < 0) { 
							C1NodeID=2*nodeID+1; 
							C2NodeID=2*nodeID+2;
						}
						else{
							C1NodeID=2*nodeID+2; 
							C2NodeID=2*nodeID+1; 
						}

						dPrime=sqrt(dValue*dValue+dPrime*dPrime);
						if (dPrime < rPrimeValue) { 
							if (C2NodeID <= localNodeDataIndex2.size()) {
								globalStack.push(make_pair(C2NodeID,dPrime));
							}
						}

						if (C1NodeID <= localNodeDataIndex2.size()) {
							globalStack.push(make_pair(C1NodeID,dValue));
						}
					}
				}
			}

			if (tmpIsHeapChanged==1) {
				for (int k=0; k<KNNCounts; ++k){
					receivingHeapArray[i][j*KNNCounts+k]=receivingHeapArray2DCopy[k];
					receivingHeapArrayDistances[i][j*KNNCounts+k]=receivingHeapArrayDistances2DCopy[k];
				}
			}
		}
	}
	/**	
	 * Now, Send the newly computed K-NNs from the above (Algorithm 1) to the original processor contained it
	 */
	if (world_rank==0) cout <<"Sending the Outputs of Algorithm-1 Back to the Original Node"<<endl;

	int sendbuffer4return[TotalReceiveCounts*KNNCounts];
	double sendbuffer6return[TotalReceiveCounts*KNNCounts];

	if (world_size != 1){	
		int indexreturn=0;
		for (int i=0; i<world_size; ++i){
			for (int j=0; j<receivingCountsMatrix[i]; ++j){
				for (int k=0; k<KNNCounts; ++k){
					sendbuffer4return[indexreturn]=receivingHeapArray[i][j*KNNCounts+k];
					sendbuffer6return[indexreturn]=receivingHeapArrayDistances[i][j*KNNCounts+k];
					++indexreturn;
				}
			}
		}
	}

	int displreturn[world_size];
	displreturn[0]=0;
	for (int i=1; i<world_size; ++i){displreturn[i]= displreturn[i-1]+receivingCountsMatrix[i-1]*KNNCounts;}
	int sendCounts[world_size];
	for (int i=0; i<world_size; ++i){sendCounts[i]= receivingCountsMatrix[i]*KNNCounts;}
	int *originalNodereceivingHeapArray[world_size];
	double *originalNodereceivingHeapArrayDistances[world_size];

	if (world_size != 1){
		for (int i=0; i<world_size; ++i){
			originalNodereceivingHeapArray[i]=new int[bufferCounts2[i]];
			originalNodereceivingHeapArrayDistances[i]=new double[bufferCounts2[i]];
			MPI_Scatterv (&sendbuffer4return,sendCounts,displreturn,MPI_INT,&originalNodereceivingHeapArray[i][0],bufferCounts2[i],MPI_INT,i,MPI_COMM_WORLD); 
			MPI_Scatterv (&sendbuffer6return,sendCounts,displreturn,MPI_DOUBLE,&originalNodereceivingHeapArrayDistances[i][0],bufferCounts2[i],MPI_DOUBLE,i,MPI_COMM_WORLD); 
		}
	}
	else {
		originalNodereceivingHeapArray[0]=new int[cnts*KNNCounts];
		originalNodereceivingHeapArrayDistances[0]=new double[cnts*KNNCounts];

		for (int i=0; i<cnts*KNNCounts; ++i){
			originalNodereceivingHeapArray[0][i]=receivingHeapArray[0][i];
			originalNodereceivingHeapArrayDistances[0][i]=receivingHeapArrayDistances[0][i];
		}
	}
	/**	
	 * Now, organize and sort the K-NNs for each given point either 
	 * received from the neighboring processors or was initially computed from the points within the same bucket
	 * after sorting, choose only the desired number (KNNCounts) of K-NNs with the shortest distance 
	 */
	if (world_rank==0) cout<<"Preparing the Final Outputs"<<endl;

	for (int i=0; i<cnts; ++i){
	    /**	
	    * Set container removes the duplicates and sort data accroding to their distance
	    */
		set <pair<double,int>> setContainer;
		int pointID=localIndexConvertor[i];
	    /**	
	    * Insert into Set container the K-NNs initially computed from the points within the same bucket
	    */
		for (int j=0; j<KNNCounts; ++j){
			if (KNNIDsinBuckets[i][j] != -1) setContainer.insert(make_pair(KNNDistanceinBuckets[i][j],KNNIDsinBuckets[i][j]));
		}
	    /**	
	    * Insert into Set container the K-NNs computed from the querying in the same processor
	    */
		for (int k=0; k<KNNCounts; ++k){
			setContainer.insert(make_pair(originalNodereceivingHeapArrayDistances[world_rank][i*KNNCounts+k],originalNodereceivingHeapArray[world_rank][i*KNNCounts+k]));
		} 
	    /**	
	    * Insert into Set container the K-NNs computed from the querying of the other neighboring processors
	    */
		for (int j=0; j<world_size-1; ++j){
			int neighborID=NeighboringNodes[i][j];
			if (neighborID == -1) continue;

			vector<int>::iterator it = std::find(ScatterVlocalNodeDataIndex[neighborID].begin(), ScatterVlocalNodeDataIndex[neighborID].end(), i);			
			int index = std::distance(ScatterVlocalNodeDataIndex[neighborID].begin(), it);  		

			for (int k=0; k<KNNCounts; ++k){
				setContainer.insert(make_pair(originalNodereceivingHeapArrayDistances[neighborID][index*KNNCounts+k],originalNodereceivingHeapArray[neighborID][index*KNNCounts+k]));  
			} 
		}
	    /**	
	    * Output the results as sorted in the Set container
	    */
		set <pair<double,int>>::iterator pairIt;
		int countsofKNN=0;
		for (pairIt=setContainer.begin(); pairIt!=setContainer.end(); ++pairIt){
			if ((*pairIt).second==-1) continue;
			//cout<<pointID<<"  "<<(*pairIt).first<<"    "<<(*pairIt).second<<endl; 
			if (pointID==1190) cout<<pointID<<"  "<<(*pairIt).first<<"    "<<(*pairIt).second<<endl; 
			++countsofKNN;
			if (countsofKNN==KNNCounts) break;
		}
		//cout<<endl;

	} //loop cnts


	delete[] receivingHeapArray[world_size];
	delete[] receivingHeapArrayDistances[world_size];

	MPI_Finalize();
	return 0;			
}

