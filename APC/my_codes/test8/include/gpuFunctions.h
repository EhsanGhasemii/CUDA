#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H


#include <armadillo>
using namespace arma; 


// define our functions
void fun1(cx_mat s, 
		cx_mat  y_noisy, 
		double N, 
		mat alpha, 
		double sigma,
		double** X_real, 
		double** X_imag, 
		double** R_real, 
		double** R_imag,
		double** Ss_real,
		double** Ss_imag,
		double** s_real,
		double** s_imag,
		double** alpha_real,
		double** output
		); 



#endif // GPUFUNCTIONS_H
