// This section of program is the Levenberg-Marquardt solution to estimate 2 parameters of a and b.
//and was modified from this source: https://github.com/SarvagyaVaish/Eigen-Levenberg-Marquardt-Optimization

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <Eigen/Eigen>

#include <unsupported/Eigen/NonLinearOptimization>
using namespace std;

struct LMFunctor
{
	// 'm' pairs of (x, f(x))
	Eigen::MatrixXf measuredValues;

	// Compute 'm' errors, one for each data point, for the given parameter values in 'x'
	int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
	{
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fvec' has dimensions m x 1
		// It will contain the error for each data point.

		float aParam = x(0);
		float bParam = x(1);

		for (int i = 0; i < values(); i++) {
			float xValue = measuredValues(i, 0);
			float yValue = measuredValues(i, 1);

			fvec(i) = yValue - (1.0 / (1.0+ aParam * pow(xValue, 2*bParam)) );
		}
		return 0;
	}

	// Compute the jacobian of the errors
	int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const
	{
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fjac' has dimensions m x n
		// It will contain the jacobian of the errors, calculated numerically in this case.

		float epsilon;
		epsilon = 1e-5f;

		for (int i = 0; i < x.size(); i++) {
			Eigen::VectorXf xPlus(x);
			xPlus(i) += epsilon;
			Eigen::VectorXf xMinus(x);
			xMinus(i) -= epsilon;

			Eigen::VectorXf fvecPlus(values());
			operator()(xPlus, fvecPlus);

			Eigen::VectorXf fvecMinus(values());
			operator()(xMinus, fvecMinus);

			Eigen::VectorXf fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

			fjac.block(0, i, values(), 1) = fvecDiff;
		}

		return 0;
	}

	// Number of data points, i.e. values.
	int m;

	// Returns 'm', the number of values.
	int values() const { return m; }

	// The number of parameters, i.e. inputs.
	int n;

	// Returns 'n', the number of inputs.
	int inputs() const { return n; }

};



//
// Goal
//
// Given a non-linear equation: f(x) = 1.0/(1.0+a*pow(x,2*b))
// and 'm' data points (x1, f(x1)), (x2, f(x2)), ..., (xm, f(xm))
// our goal is to estimate 'n' parameters (2 in this case: a, b)
// using LM optimization.
//
void estimateParameters(float &a, float &b, float min_dist, float spread, ofstream& logFile)
{

	std::vector<float> x_values;
	std::vector<float> y_values;

	/**
	 * The interval used for data fitting 
	 * The values were adopted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1138
	 */
	const float minInterval=0;
	const float maxInterval=3*spread;
	const int intervalCounts=300;


	for (int i = 0; i<intervalCounts; ++i){

		float tmp=minInterval+float(i)/float(intervalCounts)*(maxInterval-minInterval);    
		x_values.push_back(tmp);

		if (tmp <= min_dist) y_values.push_back(1.0);
		else if (tmp > min_dist) y_values.push_back(exp((min_dist-tmp)/spread));
		else {
			logFile<< "Error: Negative x_values during Parameter Estimation"<<endl;
			cout<< "Error: Negative x_values during Parameter Estimation"<<endl;
		}
	}

	// 'm' is the number of data points.
	int m = x_values.size();

	// Move the data into an Eigen Matrix.
	// The first column has the input values, x. The second column is the f(x) values.
	Eigen::MatrixXf measuredValues(m, 2);
	for (int i = 0; i < m; i++) {
		measuredValues(i, 0) = x_values[i];
		measuredValues(i, 1) = y_values[i];
	}

	// 'n' is the number of parameters (a and b) in the function.
	int n = 2;

	// 'x' is vector of length 'n' containing the initial values for the parameters.
	// The parameters 'x' are also referred to as the 'inputs' in the context of LM optimization.
	// The LM optimization inputs should not be confused with the x input values.
	Eigen::VectorXf x(n);
	x(0) = 1.8;             // initial value for 'a'
	x(1) = 0.7;             // initial value for 'b'

	//
	// Run the LM optimization
	// Create a LevenbergMarquardt object and pass it the functor.
	//

	LMFunctor functor;
	functor.measuredValues = measuredValues;
	functor.m = m;
	functor.n = n;

	Eigen::LevenbergMarquardt<LMFunctor, float> lm(functor);
	int status = lm.minimize(x);
	logFile << "LM optimization status: " << status << std::endl;
	cout << "LM optimization status: " << status << std::endl;
	//
	// Results
	// The 'x' vector also contains the results of the optimization.
	//
	logFile << "Optimization results" << std::endl;
	logFile << "\ta: " << x(0) << std::endl;
	logFile << "\tb: " << x(1) << std::endl;
	cout << "Optimization results" << std::endl;
	cout << "\ta: " << x(0) << std::endl;
	cout << "\tb: " << x(1) << std::endl;

	a=x(0);
	b=x(1);

}
