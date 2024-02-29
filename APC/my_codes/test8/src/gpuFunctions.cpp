#include "../include/gpuFunctions.h"

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
		  ) {

	// zero padding of y_noisy
	// input y_noisy is 250 * 1 
    double M = alpha.size();
    y_noisy = join_cols(zeros<cx_mat>(M*(N-1),1), y_noisy);
    y_noisy = join_cols(y_noisy, zeros<cx_mat>(M*(N-1),1));
	// output of y_noisy is 298 * 1


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

	// modifying ====================================================
	std::vector<arma::cx_mat> Ss(25);
	for(int k = 0; k < 2*N-1; k++) {
        Ss[k] = S.col(k) * S.col(k).t(); 
    }
	// =============================================================

    cx_mat sum_s = zeros<cx_mat>(N,N);
    for (int var = 0; var < 2*N-1; ++var) {
        sum_s = sum_s + S.col(var) * S.col(var).t();
    }

    cx_mat W_int = inv<cx_mat>(sum_s)*s;
    cx_mat temp_X = zeros<cx_mat>(y_noisy.size()-(N-1),1);

    for (int var = 0; var < y_noisy.size()-(N-1) - 1; ++var) {
        temp_X.row(var) = W_int.t() * y_noisy.submat(var,0,var+N-1,0);
    }

    cx_mat R = eye<cx_mat>(N,N) * sigma;
    cx_mat X = temp_X;



	// allocate memory in CPU for calculation
	*X_real  = (double*)malloc(X.n_rows * X.n_cols * sizeof(double));
	*X_imag  = (double*)malloc(X.n_rows * X.n_cols * sizeof(double));
	*R_real  = (double*)malloc(R.n_rows * R.n_cols * sizeof(double)); 
	*R_imag  = (double*)malloc(R.n_rows * R.n_cols * sizeof(double)); 
	*Ss_real = (double*)malloc(Ss.size() * s.size() * s.size() * sizeof(double)); 
	*Ss_imag = (double*)malloc(Ss.size() * s.size() * s.size() * sizeof(double));
	*s_real  = (double*)malloc(s.n_rows * s.n_cols * sizeof(double)); 
	*s_imag  = (double*)malloc(s.n_rows * s.n_cols * sizeof(double));
	*alpha_real = (double*)malloc(alpha.size() * sizeof(double)); 
	*output  = (double*)malloc(262 * 13 * 13 * sizeof(double)); // hard_code: 262, change it in future


	// check size of the variables
	std::cout << "X size: " << X.n_rows * X.n_cols << std::endl; 
	std::cout << "X rows: " << X.n_rows << std::endl; 
	std::cout << "X cols: " << X.n_cols << std::endl; 
	std::cout << "R size: " << R.n_rows * R.n_cols << std::endl; 
	std::cout << "Ss size: " << Ss.size() * s.size() * s.size() << std::endl; 
	std::cout << "s size: " << s.n_rows * s.n_cols << std::endl; 


	// transfering data from armadillo to ordinary arrays to use in GPU kernels
	for(int i=0; i<X.n_rows; ++i){
		for(int j=0; j<X.n_cols; ++j){
			(*X_real)[i * X.n_cols + j] = X(i, j).real();       // flattening the 2D data 
			(*X_imag)[i * X.n_cols + j] = X(i, j).imag();       // flattening the 2D data
		}
	}

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


	// report our state
	std::cout << "$$$$$$$$$$$$$$" << std::endl; 
	std::cout << "man fun1 hastam .." << std::endl; 

	/*cx_mat Sss = Ss[12]; 
	std::cout << "Sss : " << std::endl;						// size: 13 * 13
	for(int i=0; i<Sss.n_rows; i++){
		for(int j=0; j<Sss.n_cols; j++){
			std::cout << "Sss(" << i << ", " << j << "): ";
			std::cout << Sss(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	Sss = Ss[13]; 
	std::cout << "Sss : " << std::endl;						// size: 13 * 13
	for(int i=0; i<Sss.n_rows; i++){
		for(int j=0; j<Sss.n_cols; j++){
			std::cout << "Sss(" << i << ", " << j << "): ";
			std::cout << Sss(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	Sss = R; 
	std::cout << "Sss : " << std::endl;						// size: 13 * 13
	for(int i=0; i<Sss.n_rows; i++){
		for(int j=0; j<Sss.n_cols; j++){
			std::cout << "Sss(" << i << ", " << j << "): ";
			std::cout << Sss(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/



	std::cout << "$$$$$$$$$$$$$$" << std::endl; 


}

