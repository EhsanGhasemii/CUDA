#include "../include/gpuFunctions.h"

#include <vector>
#include <armadillo>

using namespace arma; 

// function2 for MSE calculation between an array and a cx_mat
double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size) {
	/* We suppose that size of data1 is n * m and
	 * sise of data 2 is 1 * m_prime and we just 
	 * compaire first col_size of (row)th row of 
	 * first data with second data. 
	 */
	double mse = 0.0; 
	for (int i = 0; i < col_size; ++i) {
		mse += (data1(row, i).real() - data2(0, i).real()) * (data1(row, i).real() - data2(0, i).real());
		mse += (data1(row, i).imag() - data2(0, i).imag()) * (data1(row, i).imag() - data2(0, i).imag());

	}
	return mse; 
}


void fun1(cx_mat s, 
		  std::vector<cx_mat> my_vec, 
//		  cx_mat  y_noisy, 
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
//		  double** test_real,
//		  double** test_imag, 

		  int &batch_size, 
		  int data_num, 
		  int &y_n_size, 
		  int &X_size, 
		  int &R_row, 
		  int &Ss_size, 
		  int &s_size, 
		  int &alpha_size, 
		  int &num_iter
		  ) {


	cx_mat y_noisy;
	for (int count0 = 0; count0 < my_vec.size(); ++count0) {
		y_noisy = my_vec[count0]; 

		// store each data in rows
		y_noisy = y_noisy.st();

		// zero padding of y_noisy
		double M = alpha.size();													// input y_noisy data size is 250 * 1
		y_noisy = join_rows(zeros<cx_mat>(y_noisy.n_rows, M*(N-1)), y_noisy);
		y_noisy = join_rows(y_noisy, zeros<cx_mat>(y_noisy.n_rows, M*(N-1)));		// output y_noisy data size is 1 * 274
	 
		cx_mat S = zeros<cx_mat>(N,2*N-1);

		cx_mat temp = s;
		for (int var = N-1; var < 2 * N - 1; ++var) {
			S.col(var) = temp;
			temp = join_cols(zeros<cx_mat>(1,1), temp.submat(0,0,N-2,0));
		}

		temp = s;
		for (int var = N-2; var >=0; --var) {
			temp = join_cols(temp.submat(1,0,N-1,0),zeros<cx_mat>(1,1));
			S.col(var) = temp;
		}

		std::vector<arma::cx_mat> Ss(25);
		for(int k = 0; k < 2*N-1; k++) {
			Ss[k] = S.col(k) * S.col(k).t(); 
		}

		cx_mat sum_s = zeros<cx_mat>(N,N);
		for (int var = 0; var < 2*N-1; ++var) {
			sum_s = sum_s + S.col(var) * S.col(var).t();
		}

		cx_mat W_int = inv<cx_mat>(sum_s)*s;
		cx_mat temp_X = y_noisy.submat(0, 0, y_noisy.n_rows-1, N-1) * W_int;
		
		for (int var = 1; var < y_noisy.n_cols-(N-1) - 1; ++var) {
			temp_X = join_rows(temp_X, y_noisy.submat(0, var, y_noisy.n_rows-1, var+N-1) * W_int);
		}

		cx_mat R = eye<cx_mat>(N,N) * sigma;
		cx_mat X = temp_X;

		// define our variabels
		y_n_size = y_noisy.n_cols;							// 274, 250 
		X_size = X.n_cols;									// 262, 286	
		R_row = R.n_rows;									// suppose R is a 13 * 13 square mat
		Ss_size = Ss.size();								// 25, 25
		s_size = s.size();									// 13, 13
		alpha_size = alpha.size();							// 1, 2, 3 ..

		// allocate memory in CPU for calculation
		if (count0 == 0) {
			*y_n_real = (float*)malloc(batch_size * data_num * y_n_size * sizeof(float));		// size: batch_size * data_num * y_n_size
			*y_n_imag = (float*)malloc(batch_size * data_num * y_n_size * sizeof(float));		// size: batch_size * data_num * y_n_size
			*X_real = (double*)malloc(batch_size * data_num * X_size * sizeof(double));			// size: batch_size * data_num * X_size
			*X_imag = (double*)malloc(batch_size * data_num * X_size * sizeof(double));			// size: batch_size * data_num * X_size
			*R_real = (float*)malloc(R_row * R_row * sizeof(float));							// size: R_row * R_row
			*R_imag = (float*)malloc(R_row * R_row * sizeof(float));							// size: R_row * R_row
			*Ss_real = (float*)malloc(Ss_size * s_size * s_size * sizeof(float));				// size: Ss_size * s_size * s_size 
			*Ss_imag = (float*)malloc(Ss_size * s_size * s_size * sizeof(float));				// size: Ss_size * s_size * s_size
			*s_real = (float*)malloc(s_size * sizeof(float));									// size: s_size
			*s_imag = (float*)malloc(s_size * sizeof(float));									// size: s_size
			*alpha_real = (float*)malloc(alpha_size * sizeof(float));							// size: alpha_size
			*output_real = (double*)malloc(batch_size * data_num * X_size * sizeof(double));	// size: batch_size * data_num * X_size
			*output_imag = (double*)malloc(batch_size * data_num * X_size * sizeof(double));	// size: batch_size * data_num * X_size

			// check size of the variables
			std::cout << "num_iter: " << num_iter << std::endl; 
			std::cout << "batch_size: " << batch_size << std::endl; 
			std::cout << "data_num: " << data_num - 1 << std::endl; 
			std::cout << "y_n_size: " << y_n_size << std::endl; 
			std::cout << "X_size: " << X_size << std::endl; 
			std::cout << "R_row: " << R_row << std::endl; 
			std::cout << "Ss_size: " << Ss_size << std::endl; 
			std::cout << "s_size: " << s_size << std::endl; 
		}

	// transfering data from armadillo to ordinary arrays to use in GPU kernels
		for(int i=0; i<y_noisy.n_rows; ++i){
			for(int j=0; j<y_noisy.n_cols; ++j){
				(*y_n_real)[count0 * y_noisy.n_rows * X.n_cols + i * X.n_cols + j] = y_noisy(i, j).real();       // flattening the 2D data 
				(*y_n_imag)[count0 * y_noisy.n_rows * X.n_cols + i * X.n_cols + j] = y_noisy(i, j).imag();       // flattening the 2D data
			}
		}
	
		for(int i=0; i<X.n_rows; ++i){
			for(int j=0; j<X.n_cols; ++j){
				(*X_real)[count0 * X.n_rows * X.n_cols + i * X.n_cols + j] = X(i, j).real();       // flattening the 2D data 
				(*X_imag)[count0 * X.n_rows * X.n_cols + i * X.n_cols + j] = X(i, j).imag();       // flattening the 2D data
			}
		}


		if (count0 == 0){
			for(int i=0; i<R.n_rows; ++i){
				for(int j=0; j<R.n_cols; ++j){
					(*R_real)[i * R.n_cols + j] = R(i, j).real();       // flattening the 2D data 
					(*R_imag)[i * R.n_cols + j] = R(i, j).imag();       // flattening the 2D data
				}
			}

			int index = 0;
			for (const auto& mat : Ss) {
				for (int i = 0; i < mat.n_rows; ++i) {
					for (int j = 0; j < mat.n_cols; ++j) {
						(*Ss_real)[index] = mat(i, j).real();           // flattening the 3D data
						(*Ss_imag)[index] = mat(i, j).imag();           // flattening the 3D data
						++index;
					}
				}
			}

			for(int i=0; i<s.n_rows; ++i){
				for(int j=0; j<s.n_cols; ++j){
					(*s_real)[i * s.n_cols + j] = s(i, j).real();       // flattening the 2D data 
					(*s_imag)[i * s.n_cols + j] = s(i, j).imag();       // flattening the 2D data
				}
			}

			// copy alpha to alpha real. It convert from mat to array. 
			std::copy(alpha.memptr(), alpha.memptr() + alpha.size(), (*alpha_real));
		} // End of (count0 == 0)

	} // End of loop over batch size

}

