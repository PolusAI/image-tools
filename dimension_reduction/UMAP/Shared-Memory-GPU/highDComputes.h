#include <string>
#include <string>
#include <boost/iostreams/stream.hpp> 

using namespace std;

void findMin(int** B_Index,double** B_Dist, int N,int K,int* B_Index_Min,double* B_Dist_Min);

void findSigma(double ** B_Dist, double * B_Dist_Min, double * SigmaValues, int N, int K);

/**
 * Read the output of linux command execution 
 * @param  cmd  is the linux command to be executed
 * @return the output from the execution of the linux command
 */
inline std::string exec(const char* cmd) {
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


