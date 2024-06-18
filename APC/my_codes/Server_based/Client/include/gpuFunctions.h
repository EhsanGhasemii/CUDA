#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H

#include <armadillo>
#include <vector>
using namespace arma; 

double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size);

// define our functions
void fun1(cx_mat s,
		std::vector<cx_mat> my_vec, 
		double N, 
		mat alpha, 
		double sigma,
		float** y_n_real,
		float** y_n_imag,
		double** X_real, 
		double** X_imag,
		float** R_real, 
		float** R_imag,
		float** Ss_real,
		float** Ss_imag,
		float** s_real,
		float** s_imag,
		float** alpha_real,
		double** output_real,
		double** output_imag,

		int &batch_size, 
		int data_num, 
		int &y_n_size, 
		int &X_size, 
		int &R_row, 
		int &Ss_size, 
		int &s_size,
		int &alpha_size, 
		int &num_iter
		); 



#endif // GPUFUNCTIONS_H
