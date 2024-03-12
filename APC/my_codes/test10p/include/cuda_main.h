#ifndef ABAS_H
#define ABAS_H


//#include <armadillo>
//using namespace arma; 


// define our functions
void gpuKernel(double* y_n_real,
			   double* y_n_imag,
			   double* X_real,
			   double* X_imag,
			   double* rho_real, 
			   double* rho_imag, 
			   double* R_real,
			   double* R_imag,
			   double* Ss_real,
			   double* Ss_imag,
			   double* s_real,
			   double* s_imag,
			   double* alpha_real,
			   double* output_real,
			   double* output_imag,
			   double* test, 

			   int data_num, 
			   int y_n_size, 
			   int X_size, 
			   int R_row, 
			   int Ss_size, 
			   int s_size
			   ); 
//void fill(float* data, int size);



#endif // ABAS_H
