/**
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * August 2019
 * Please note that 2 parameters of sampleRate and largestDistance have significant
 * imapct on the performance. Their modification for the given problem is advised
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

using namespace std;


int main(int argc, char * const argv[]) {
	/**
	* The full path to the input file containig the dataset.
	*/
	string filePath = argv[1]; 
	/**
	* Size of Dataset without the header (i.e.(#Rows in dataset)-1).
	*/
	const int N = atoi(argv[2]); 
	/**
	* Dimension of Dataset (#Columns)
	*/
	const int Dim = atoi(argv[3]); 
	/**
	* K in K-NN that means the desired number of Nearest Neighbours to be computed.
	*/
	const int K = atoi(argv[4]); 
	/**
	* The rate at which we do sampling. This parameter plays a key role in the performance.
	* This parameter is a trades-off between the performance and the accuracy of the results.
	* Values closer to 1 provides more accurate results but the execution takes longer.
	*/
	float sampleRate = stof(argv[5]); 
	/**
	* Convergance Threshold. A fixed integer is used here instead of delta*N*K.
	*/
	const int convThreshold = atoi(argv[6]); 
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
	int** B_IsNew = new int*[N];
	for (int i = 0; i < N; ++i) { B_IsNew[i] = new int[K]; }
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
	double epsilon = 1e-20; //
	/**
	* An approximation for Infinity. The distance between pairs of points can not be
	* larger than this parameter. If the data points are closer to each other, we can
	* reduce this parameter which will have significant impact on the performance.
	*/
	int largestDistance = 1000000000;
	/**
	* At first, let's Read Dataset from Input File
	*/
	ifstream infile;
	infile.open(filePath);
	if (infile.fail())
	{
		cout << "error in Opening Input File" << endl;
		return 1;
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
			dataPoints[i][j] = atof(temp2.c_str());
			temp.erase(0, temp.find(",") + 1);
		}
	}
	infile.close();
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
		for (int j = 0; j < K; ++j) {
			B_IsNew[i][j] = 1;
			B_Dist[i][j] = largestDistance;
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
	 * Update list of K-NN indices for u1 (B_Index) if u2 is closer
	 * <p>
	 * This method correspondd to UPDATENN(B[u1],<u2,l,true>) in the paper
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
		for (int j = 0; j < K; j++) {
			if (B_Index[u1][j] == u2 && B_Dist[u1][j] != largestDistance) { return 0; }
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
	};
	 /**
	 * Main Loop of the Algorithm
	 */
	int iterate = 1;
	while (iterate) {
		/**
		* Create "New" for each Datapoint
		*/
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < K; ++j) {
				if (float(rand() % 100) / 100 < sampleRate) {
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
				if (float(rand() % 100) / 100 < sampleRate) {
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
                	/**
	                 * computes spatial distance between two points
	                 * @param  a and b represents the indice of two points in dataset,
	                 * data represents input dataPoints read from filePath, Dim is #columns in dataset
	                 * @return spatial distance between points a and b
	                */
						double dist = 0;
						for (int i = 0; i < Dim; ++i) {
							dist += pow((dataPoints[*it][i] - dataPoints[*it2][i]), 2);
						}
						double dista = sqrt(dist);
						if (dista < epsilon) {cout << "Found Duplicate Data for Points "<< *it << " and " << *it2; }

						c_criteria += UpdateNN(*it, *it2, dista, 1);
						c_criteria += UpdateNN(*it2, *it, dista, 1);
					}
				}
			}
		}
		cout << "c_criteria = " << c_criteria << " With Threshold Convergence of " << convThreshold << endl;
		if (c_criteria < convThreshold) { iterate = 0; }
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
	* Output the computed K-NN for any desired point "index1"
	*/
	int index1 = 1087;
	for (int j = 0; j < K; ++j) {
		cout << B_Index[index1][j] << ",";
	}
	cout << endl;
	/**
	* compute the theoretical K-NN for the point "index2" just for verification with the above results
	*/
	int index2 = 1087;
	ofstream outputFile;
	outputFile.open("Actual_Distances.csv");
	vector<int> Vector_for_Index;
	vector<double> Vector_for_Index2;
	for (int i = 0; i < N; ++i) {
		if (i == index2) { continue; }

		double dist = 0;
		for (int j = 0; j < Dim; ++j) {
			dist += pow((dataPoints[index2][j] - dataPoints[i][j]), 2);
		}
		dist = sqrt(dist);

		Vector_for_Index.push_back(i);
		Vector_for_Index2.push_back(dist);
	}
	for (int i = 0; i < Vector_for_Index.size(); ++i) {
		outputFile << Vector_for_Index[i] << "," << Vector_for_Index2[i] << endl;
	}
	outputFile.close();

	return 0;
}


