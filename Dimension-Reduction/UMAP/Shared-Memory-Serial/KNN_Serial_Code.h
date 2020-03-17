#include <string>

using namespace std;

string exec(const char* cmd);

void computeKNNs(string filePath, const int N, const int Dim, const int K, float sampleRate, const int convThreshold, int** B_Index,double** B_Dist, ofstream& logFile, string distanceMetric, float distanceV1, float distanceV2, string filePathOptionalArray);


