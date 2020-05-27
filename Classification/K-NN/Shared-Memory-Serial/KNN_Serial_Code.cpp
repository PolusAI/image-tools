/**
 * This code is an implementation of the algorithm presented by Dong et al., 2012,
 *"Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"  
 * @author:   Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * August 2019
 */

#include <vector>
#include <iostream> 
#include <list>
#include <string>
#include <math.h>
#include <fstream>
#include <float.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp> 
#include <boost/iostreams/stream.hpp>    

using boost::iostreams::mapped_file_source;
using boost::iostreams::stream;		
using namespace std;

/**
 * Read the output of linux command execution 
 * @param  cmd  is the linux command to be executed
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

int main(int argc, char * const argv[]) {
	/**
	 * The errors and informational messages are outputted to the log file 
	 */
	ofstream logFile;
	string logFileName="Setting.txt";
	logFile.open(logFileName);

	/**
	 * The input parameters are read from command line which are as follow.
	 * inputPath: The full path to the input file containig the dataset.	 
	 * outputPath: The full path to the output csv files.
	 * K: K in K-NN that means the desired number of Nearest Neighbours to be computed.
	 * sampleRate: The rate at which we do sampling. This parameter plays a key role in the performance.
	 * This parameter is a trades-off between the performance and the accuracy of the results.
	 * Values closer to 1 provides more accurate results but the execution takes longer.	 
	 * convThreshold: Convergance Threshold. A fixed integer is used here instead of delta*N*K.	
	 * colIndex1 and colIndex2 (optional): The indices of columns from the input csv file where raw data exists continuously in between.
	 * If these two arguments were left blank, the code assumes that the entire input csv file is raw data 
	 * and automatically computes the number of columns in the input csv file. 	
	 */	 
	string filePath, outputPath, outputPath2, inputPath,LogoutputPath;
	int K,convThreshold, colIndex1=-1, colIndex2=-1;
	float sampleRate;	

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
		else if (string(argv[i])=="--convThreshold") convThreshold=stof(argv[i+1]);
		else if (string(argv[i])=="--outputPath"){
			boost::filesystem::path p(argv[i+1]);

			if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
			{
				logFile << "Incorrect output path";
				cout << "Incorrect output path";
				return 1;
			}

			LogoutputPath=argv[i+1];
			boost::filesystem::path joinedPath = p / boost::filesystem::path("KNN_Indices.csv");
			outputPath = joinedPath.string();
			boost::filesystem::path joinedPath2 = p / boost::filesystem::path("KNN_Distances.csv");
			outputPath2 = joinedPath2.string();
		}
		else if (string(argv[i])=="--colIndex1") {
			colIndex1=stof(argv[i+1]);
			if (colIndex1<1) {
				logFile << "colIndex1 should be greater than 1";
				cout << "colIndex1 should be greater than 1";
				return 1;
			}
		}
		else if (string(argv[i])=="--colIndex2") {
			colIndex2=stof(argv[i+1]); 
			if (colIndex2<1) {
				logFile << "colIndex2 should be greater than 1";
				cout << "colIndex2 should be greater than 1";
				return 1;
			}
		}    
	}	

	logFile<<"------------The following Input Arguments were read------------"<<endl;
	logFile<<"The full path to the input file: "<< filePath<<endl;	
	logFile<<"The desired number of NN to be computed: "<< K <<endl;
	logFile<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl; 
	logFile<<"The convergance threshold: "<< convThreshold <<endl; 				
	logFile<<"The full path to the output file1: "<< outputPath<<endl;
	logFile<<"The full path to the output file2: "<< outputPath2<<endl;
	if (colIndex1 != -1) {logFile<<"The optioanl column index starts from: "<< colIndex1<<endl;}	
	if (colIndex2 != -1) {logFile<<"The optioanl column index ends at: "<< colIndex2<<endl;}		

	cout<<"------------The following Input Arguments were read------------"<<endl;	
	cout<<"The full path to the input file: "<< filePath<<endl;
	cout<<"The desired number of NN to be computed: "<< K <<endl;	
	cout<<"The sampleRate(The rate at which we do sampling): "<< sampleRate <<endl; 	
	cout<<"The convergance threshold: "<< convThreshold <<endl; 	
	cout<<"The full path to the output file1: "<< outputPath<<endl;
	cout<<"The full path to the output file2: "<< outputPath2<<endl;
	if (colIndex1 != -1) {cout<<"The optioanl column index starts from: "<< colIndex1<<endl;}	
	if (colIndex2 != -1) {cout<<"The optioanl column index ends at: "<< colIndex2<<endl;}	

	/**
	 * Size of Dataset without the header (i.e.(#Rows in dataset)-1).
	 */	
	string cmd="wc -l "+filePath;
	string outputCmd = exec(cmd.c_str());
	const int N=stoi(outputCmd.substr(0, outputCmd.find(" ")))-1;
	/**
	 * Dimension of Dataset (#Columns)
	 * is computed automatically if not passed as argument in command line
	 * otherwise, the range (beginning and end) for the column index of input csv file is needed
	 */
	int Dim;
	string cmd2="head -n 1 "+ filePath + " |tr '\\,' '\\n' |wc -l ";
	Dim = stoi(exec(cmd2.c_str())); 

	logFile<<"The input csv file contains "<<N<<" rows of raw data with "<< Dim<< " columns(features)"<<endl; 
	cout<<"The input csv file contains "<<N<<" rows of raw data with "<< Dim<< " columns(features)"<<endl; 
	/**
	 * A 2D Array containing the entire input dataset (read from filePath).
	 */
	double** dataPoints = new double*[N];
	for (int i = 0; i < N; ++i) { dataPoints[i] = new double[Dim]; }
	/**
	 * indices of K-NN for all the points in dataset
	 */
	int** B_Index = new int*[N];
	for (int i = 0; i < N; ++i) { B_Index[i] = new int[K]; }
	/**
	 * corresponding distance for K-NN indices stored in B_Index
	 */
	double** B_Dist = new double*[N];
	for (int i = 0; i < N; ++i) { B_Dist[i] = new double[K]; }
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
	list<int> *New_Final_List = new list<int>[N];
	/**
	 * Iterators to access data stored in the list
	 */
	list<int>::iterator it, it2, it_temp;
	/**
	 * An approximation of zero in computing distances. Two points with the distance
	 * smaller than epsilon are considered as one point.
	 */
	double epsilon = 1e-10; //
	short* allEntriesFilled = new short[N];
	/**
	 * At first, let's Read Dataset from Input File Using Memory Mapping
	 */
	mapped_file_source mmap(filePath);
    stream<mapped_file_source> is(mmap, std::ios::binary);
    if (is.fail())
	{
		logFile << "error in Opening Input File" << endl;
		cout << "error in Opening Input File" << endl;
		return ;
	}
	/**
	 * Remove the header info
	 */
	string dummyLine;
	getline(is, dummyLine);
	/**
	 * Reading the Entire Dataset
	 */
	if (argc==11){
		for (int i = 0; i < N; ++i) {
			string temp, temp2;
			getline(is, temp);
			for (int j = 0; j < Dim; ++j) {
				temp2 = temp.substr(0, temp.find(","));
				dataPoints[i][j] = atof(temp2.c_str());
				temp.erase(0, temp.find(",") + 1);
			}
		}
	} else {
		for (int i = 0; i < N; ++i) {
			string temp, temp2;
			getline(is, temp);
			for (int j = 0; j < Dim; ++j) {
				temp2 = temp.substr(0, temp.find(","));
				if (j >= colIndex1-1 && j < colIndex2) dataPoints[i][j] = atof(temp2.c_str());
				temp.erase(0, temp.find(",") + 1);
			}
		}	
	}
	mmap.close();

	if (colIndex1 != -1) Dim=colIndex2-colIndex1+1;
	if (Dim <1) {
		logFile << "Error in Computing the Dimension of input csv file" << endl;
		cout << "Error in Computing the Dimension of input csv file" << endl;
		return 1;	
	}
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
	auto UpdateNN = [&](int u1, int u2, double distance, int flag = 1) {

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
			if (index == -1) { cout << "Error"; } 
			if (distance < max) {
				B_Dist[u1][index] = distance;
				B_Index[u1][index] = u2;
				B_IsNew[u1][index] = flag;
				return 1;
			}
			else { return 0; }
		}
	};
	/**
	 * Main Loop of the Algorithm
	 */
	bool iterate = true;
	while (iterate) {
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
		 * c=c+UPDATENN(B[u1],<u2,l,true>)
		 */
		int c_criteria = 0;
		for (int i = 0; i < N; ++i) {
			for (it = New_Final_List[i].begin(); it != New_Final_List[i].end(); it++) {
				it_temp = it;
				advance(it_temp, 1);
				for (it2 = it_temp; it2 != New_Final_List[i].end(); it2++) {
					if (*it != *it2) {
						double dist = 0;
						for (int i = 0; i < Dim; ++i) {
							dist += pow((dataPoints[*it][i] - dataPoints[*it2][i]), 2);
						}
						double dista = sqrt(dist);
						if (dista < epsilon) {
							logFile << "Found Duplicate Data for Points "<< *it << " and " << *it2<<endl;; 
							cout << "Found Duplicate Data for Points "<< *it << " and " << *it2<<endl; 
							return 1;
						}

						c_criteria += UpdateNN(*it, *it2, dista, 1);
						c_criteria += UpdateNN(*it2, *it, dista, 1);
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
	/**
	 * Sort and output the results
	 */
	ofstream outputFileIndex,outputFileDistance;
	outputFileIndex.open(outputPath);	
	outputFileDistance.open(outputPath2);

	for (int i=0; i<N; ++i){
		vector<pair<double,int>> aggregateResults;
		for (int j=0; j<K; ++j){
			aggregateResults.push_back(make_pair(B_Dist[i][j], B_Index[i][j]));
		}		
		sort(aggregateResults.begin(), aggregateResults.end());	

		for (int j=0; j<K; ++j){
			if (j != K-1) {outputFileIndex<<aggregateResults[j].second<<",";}
			else {outputFileIndex<<aggregateResults[j].second<<endl; }
		}

		for (int j=0; j<K; ++j){
			if (j != K-1) {outputFileDistance<<aggregateResults[j].first<<",";}
			else {outputFileDistance<<aggregateResults[j].first<<endl; }
		}
	}

	outputFileIndex.close();
	outputFileDistance.close();
	logFile.close();
	/**
	 * copy Logfile to the file system which could be accessed outside the docker container
	 */ 
	string cmd3="cp "+ logFileName+"  "+LogoutputPath;
	string outputCmd3 = exec(cmd3.c_str());

	return 0;
}


