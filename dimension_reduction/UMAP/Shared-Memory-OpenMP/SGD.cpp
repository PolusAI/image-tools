#include <math.h> 
using namespace std;


/**
 * Standard clamping of a value into a fixed range (in this case -4.0 to 4.0)
 * This function is used in SGD solver
 */    
double clip(double value){

	const double clipLowVal=-4.0;
	const double clipHighVal=4.0;
	double returnValue;

	if (value < clipLowVal) {returnValue=clipLowVal;}
	else if ( value > clipHighVal) {returnValue=clipHighVal;}
	else {returnValue=value;}

	return returnValue;
}


/**
 * The squared distance between 2 points in the Low-D (embedded) space
 */ 
double rdist(double ** embedding, int Dim, int index1, int index2){

	double dist_squared=0;

	for (int j = 0; j < Dim; ++j) {  
		dist_squared += pow(embedding[index1][j]-embedding[index2][j],2);
	} 

	return dist_squared;			
}
