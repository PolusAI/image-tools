#include <string>

using namespace std;

/**
 * Error handling of CUDA directives
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void computeKNNs(string filePath, const int N, const int Dim, const int K, float sampleRate, const int convThreshold, int** B_Index,double** B_Dist, ofstream& logFile, string distanceMetric, float distanceV1, float distanceV2, string filePathOptionalArray);
