#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H

#include <armadillo>
#include <vector>
using namespace arma; 

int abas(int in); 


// define our functions
void fun1(cx_mat s,
		std::vector<cx_mat> my_vec, 
	//	cx_mat y_noisy, 
		double N, 
		mat alpha, 
		double sigma,
		float** y_n_real,
		float** y_n_imag,
		double** X_real, 
		double** X_imag,
	//	double** rho_real, 
	//	double** rho_imag, 
		float** R_real, 
		float** R_imag,
		float** Ss_real,
		float** Ss_imag,
		float** s_real,
		float** s_imag,
		double** alpha_real,
		double** output_real,
		double** output_imag,
//		double** test_real,
//		double** test_imag, 

		int &batch_size, 
		int data_num, 
		int &y_n_size, 
		int &X_size, 
		int &R_row, 
		int &Ss_size, 
		int &s_size,
		int &alpha_size
		); 



#endif // GPUFUNCTIONS_H
