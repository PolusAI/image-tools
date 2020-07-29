#include <Eigen/Sparse>
using namespace Eigen;

void Initialization (bool randominitializing, double** locationLowSpace, ofstream& logFile, int N, SparseMatrix<float> & graphSM, float MaxWeight, int DimLowSpace, int n_epochs);
