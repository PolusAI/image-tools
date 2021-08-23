
#define epsilonGPU 1e-6
__constant__ int epsilon=epsilonGPU;


__device__ double piGPU() { return atan(1)*4;}

template<typename T>
__device__ double approx_log_GammaGPU(T x){
	if (x - 1 < epsilon) return 0;
	return x*log(x) - x + 0.5*log(2.0*piGPU()/x) + 1.0/(x*12.0);
}

__device__ double log_betaGPU(double x, double y){
	double a = min(x, y);
	double b = max(x, y);

	if (b < 5){
		double value = -log(b);
		for (int i = 1; i < int(a); ++i) value += log(i) - log(b + i);
		return value;
	}    
	else return approx_log_GammaGPU(x) + approx_log_GammaGPU(y) - approx_log_GammaGPU(x + y);
}

__device__ double log_single_betaGPU(double x){
	return log(2.0) * (-2.0 * x + 0.5) + 0.5 * log(2.0 * piGPU() / x) + 0.125 / x;
}

__device__ double distanceCompute(int Dim, double * device_dataPointsGPU, int par1, int par2, int metricID, float distanceV1, float distanceV2, float * v0, float * v1){

	/**
	 *  zero approximation in float precision for Metric computations
	 */
	double epsilon=1e-6;

	if (metricID==1) { //euclidean Metric
		double output;
		for (int i=0; i<Dim; ++i){
			double tmp = device_dataPointsGPU[par1*Dim+i] - device_dataPointsGPU[par2*Dim+i];
			output += tmp * tmp; 
		}
		return sqrt(output);
	}
	else if (metricID ==2) { // manhattan
		double tmp = 0;
		for (int i = 0; i < Dim; ++i) {
			tmp += abs(device_dataPointsGPU[par1*Dim+i] - device_dataPointsGPU[par2*Dim+i]);
		}
		return tmp;
	}
	else if (metricID ==3) { //minkowski
		double tmp = 0;
		for (int i = 0; i < Dim; ++i) {
			tmp += pow(abs(device_dataPointsGPU[par1*Dim+i] - device_dataPointsGPU[par2*Dim+i]), distanceV1);
		}   
		return pow(tmp, (1.0 / distanceV1) );  
	}
	else if (metricID ==4) { //cosine
		double result = 0.0, norm_x = 0.0, norm_y = 0.0;
		for (int i = 0; i < Dim; ++i) {
			result += device_dataPointsGPU[par1*Dim+i] * device_dataPointsGPU[par2*Dim+i];
			norm_x += device_dataPointsGPU[par1*Dim+i] * device_dataPointsGPU[par1*Dim+i];
			norm_y += device_dataPointsGPU[par2*Dim+i] * device_dataPointsGPU[par2*Dim+i];
		}
		if (norm_x < epsilon && norm_y < epsilon) return 0.0;
		else if (norm_x < epsilon || norm_y < epsilon) return 1.0;
		else return 1.0 - (result / sqrt(norm_x * norm_y)); 
	}
	else if (metricID ==5) {  //correlation
		double mu_x=0.0, norm_x=0.0;
		double mu_y=0.0, norm_y=0.0;
		double dot_product = 0.0;

		for (int i = 0; i < Dim; ++i) {
			mu_x += device_dataPointsGPU[par1*Dim+i];
			mu_y += device_dataPointsGPU[par2*Dim+i];	  
		}
		mu_x /=Dim;
		mu_y /=Dim;

		double shifted_x,shifted_y; 
		for (int i = 0; i < Dim; ++i) {
			shifted_x = device_dataPointsGPU[par1*Dim+i] - mu_x;
			shifted_y = device_dataPointsGPU[par2*Dim+i] - mu_y;
			norm_x += shifted_x * shifted_x;
			norm_y += shifted_y * shifted_y;
			dot_product += shifted_x * shifted_y;	  	  	  
		}

		if (norm_x < epsilon && norm_y < epsilon) return 0.0;
		else if (dot_product < epsilon)  return 1.0;
		else  return 1.0 - (dot_product / sqrt(norm_x * norm_y));    
	}
	else if (metricID ==6) {  //bray_curtis
		double numerator = 0.0, denominator = 0.0;

		for (int i = 0; i < Dim; ++i) {
			numerator += abs(device_dataPointsGPU[par1*Dim+i] - device_dataPointsGPU[par2*Dim+i]);
			denominator += abs(device_dataPointsGPU[par1*Dim+i] + device_dataPointsGPU[par2*Dim+i]);
		}

		if (denominator > 0.0) return numerator/denominator;
		else return 0.0;
	}
	else if (metricID ==7) { //ll_dirichlet
		double n1,n2;
		for (int i = 0; i < Dim; ++i) {
			n1 +=device_dataPointsGPU[par1*Dim+i];
			n2 +=device_dataPointsGPU[par2*Dim+i];	
		}
		double log_b = 0.0, self_denom1 = 0.0, self_denom2 = 0.0;

		for (int i = 0; i < Dim; ++i) {
			if (device_dataPointsGPU[par1*Dim+i] * device_dataPointsGPU[par2*Dim+i] > 0.9){
				log_b += log_betaGPU(device_dataPointsGPU[par1*Dim+i], device_dataPointsGPU[par2*Dim+i]);
				self_denom1 += log_single_betaGPU(device_dataPointsGPU[par1*Dim+i]);
				self_denom2 += log_single_betaGPU(device_dataPointsGPU[par2*Dim+i]);
			}
			else {
				if (device_dataPointsGPU[par1*Dim+i] > 0.9) self_denom1 += log_single_betaGPU(device_dataPointsGPU[par1*Dim+i]);
				if (device_dataPointsGPU[par2*Dim+i] > 0.9) self_denom2 += log_single_betaGPU(device_dataPointsGPU[par2*Dim+i]);	  
			}  	    
		}

		return sqrt(1.0 / n2 * (log_b - log_betaGPU(n1, n2) - (self_denom2 - log_single_betaGPU(n2)))
				+ 1.0 / n1 * (log_b - log_betaGPU(n2, n1) - (self_denom1 - log_single_betaGPU(n1))) );
	}
	else if (metricID ==8) { //jaccard
		int x_true, y_true, num_non_zero=0, num_equal=0; 

		for (int i = 0; i < Dim; ++i) {
			if ( device_dataPointsGPU[par1*Dim+i] < epsilon) x_true=0;
			else x_true=1;

			if ( device_dataPointsGPU[par2*Dim+i] < epsilon) y_true=0;
			else y_true=1;

			if (x_true==1 || y_true==1) ++num_non_zero;
			if (x_true==1 && y_true==1) ++num_equal;    
		}

		if (num_non_zero == 0) return 0.0;
		else return double (num_non_zero - num_equal) / num_non_zero;
	}

	else if (metricID ==9) { //dice
		int num_true_true=0, num_not_equal=0,x_true, y_true;
		for (int i = 0; i < Dim; ++i) {
			if ( device_dataPointsGPU[par1*Dim+i] < epsilon) x_true=0;
			else x_true=1;

			if ( device_dataPointsGPU[par2*Dim+i] < epsilon) y_true=0;
			else y_true=1;

			if (x_true==1 && y_true==1) ++num_true_true;
			if (x_true != y_true) ++num_not_equal;    
		}

		if (num_not_equal==0) return 0.0;
		else return double(num_not_equal) / (2.0 * num_true_true + num_not_equal);
	}

	else if (metricID ==10) { //categorical_distance
		if (device_dataPointsGPU[par1*Dim] == device_dataPointsGPU[par2*Dim]) return 0.0;
		else return 1.0;
	}

	else if (metricID ==11) { //ordinal_distance
		return abs(device_dataPointsGPU[par1*Dim] - device_dataPointsGPU[par2*Dim]) / distanceV1;
	}

	else if (metricID ==12) { //count_distance
		double poisson_lambda=distanceV1; //default 1.0
		double normalisation=distanceV2; //default 1.0
		double log_k_factorial;

		double lo=int(min(device_dataPointsGPU[par1*Dim], device_dataPointsGPU[par2*Dim]));
		double hi=int(max(device_dataPointsGPU[par1*Dim], device_dataPointsGPU[par2*Dim]));

		double log_lambda = log(poisson_lambda);

		if (lo < 2) log_k_factorial = 0.0;
		else if (lo < 10) {
			log_k_factorial = 0.0;
			for (int k=2; k<lo; ++k) log_k_factorial += log(k);
		}
		else log_k_factorial = approx_log_GammaGPU(lo + 1);

		double result = 0.0;
		for (int k = lo; k < hi; ++k) {
			result += k * log_lambda - poisson_lambda - log_k_factorial;
			log_k_factorial += log(k);
		}    
		return result/normalisation;
	}

	else if (metricID ==13) { //levenshtein
		float normalisation = distanceV1;  //default 1.0
		int max_distance = int(distanceV2); //default 20
		int x_len=Dim, y_len=Dim;
		//float v0[y_len + 1], v1[y_len + 1];

		if (abs(x_len - y_len) > max_distance) return float(abs(x_len - y_len))/normalisation;

		for (int i=0; i<y_len+1; ++i){
			v0[i]=i;
			v1[i]=0;
		}

		int substitution_cost;
		float minVal,deletion_cost,insertion_cost;
		for (int i=0; i<x_len; ++i){
			v1[i] = i + 1;
			for (int j=0; j<y_len; ++j){      
				deletion_cost = v0[j + 1] + 1;
				insertion_cost = v1[j] + 1;

				if (device_dataPointsGPU[par1*Dim+i] == device_dataPointsGPU[par2*Dim+i]) substitution_cost=1;
				else  substitution_cost=0;

				v1[j + 1] = min(deletion_cost, min(insertion_cost, float(substitution_cost))  );
			}

			for (int k=0; k<y_len+1; ++k) {
				v0[k]=v1[k];

				if (k==0) minVal= v0[k];
				else { if (v0[k] < minVal) minVal= v0[k];}
			}

			if (minVal> max_distance) return float(max_distance)/normalisation;

		}
		return v0[y_len] / normalisation; 
	}
	else {
		printf("Wrong input for GPU metric name!");
	}
return -1;
} 

