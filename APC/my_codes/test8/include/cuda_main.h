#ifndef ABAS_H
#define ABAS_H


//#include <armadillo>
//using namespace arma; 


// define our functions
void gpuKernel(double* y_n_real,
			   double* y_n_imag,
			   double* X_real,
			   double* X_imag,
			   double* R_real,
			   double* R_imag,
			   double* Ss_real,
			   double* Ss_imag,
			   double* s_real,
			   double* s_imag,
			   double* alpha_real,
			   double* output_real,
			   double* output_imag
			   ); 
//void fill(float* data, int size);



#endif // ABAS_H
