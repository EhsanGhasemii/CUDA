#ifndef ABAS_H
#define ABAS_H


//#include <armadillo>
//using namespace arma; 

// define our functions
void gpuKernel(float* y_n_real,
			   float* y_n_imag,
			   double* X_real,
			   double* X_imag,
			   float* R_real,
			   float* R_imag,
			   float* Ss_real,
			   float* Ss_imag,
			   float* s_real,
			   float* s_imag,
			   float* alpha_real,
			   double* output_real,
			   double* output_imag,

			   int batch_size, 
			   int data_num, 
			   int y_n_size, 
			   int X_size, 
			   int R_row, 
			   int Ss_size, 
			   int s_size, 
			   int alpha_size, 
			   int num_iter
			   ); 

#endif // ABAS_H
