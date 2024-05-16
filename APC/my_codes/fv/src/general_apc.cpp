#include "../include/general_apc.h"

// additional libraries
#include <chrono>
#include <vector>

General_APC::General_APC()
{

}

cx_mat General_APC::algorithm(cx_mat s, cx_mat  y_noisy, double N, mat alpha, double sigma)
{
	
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

    cx_mat sum_s = zeros<cx_mat>(N,N);
    for (int var = 0; var < 2*N-1; ++var) {
        sum_s = sum_s + S.col(var) * S.col(var).t();
    }

    cx_mat W_int = inv<cx_mat>(sum_s)*s;
    cx_mat temp_X = zeros<cx_mat>(y_noisy.size()-(N-1),1);

    for (int var = 0; var < y_noisy.size()-(N-1) - 1; ++var) {
        temp_X.row(var) = W_int.st() * y_noisy.submat(var,0,var+N-1,0);
    }

    cx_mat R = eye<cx_mat>(N,N) * sigma;
    cx_mat X = temp_X;

    // Get the starting timepoint
	auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < M; ++j) {									// 2x
        temp_X = zeros<cx_mat>(1,X.size()-2*N+2);
        //cx_mat rho = elementwisePow(X, alpha(j,0));
		arma::mat rho = arma::pow(arma::abs(X), alpha(j, 0)); 

        for (int i = N-1; i < X.size()-N+1; ++i) {					// 262x and 238x
			arma::mat temp_rho = rho.submat(i-N+1,0,i+N-1,0);
            cx_mat C = zeros<cx_mat>(N,N);


			auto time1 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < 2*N-1; ++k) {						// 25x
                C = temp_rho(k,0)*S.col(k)*S.col(k).t() + C;
            }

			auto time2 = std::chrono::high_resolution_clock::now();
            cx_mat W = inv(C+R)*s*rho(i,0);


			auto time3 = std::chrono::high_resolution_clock::now();
            cx_mat t = W.t() * y_noisy.submat(i+(N-1)*(j),0,(j)*(N-1)+i+N-1,0);


			auto time4 = std::chrono::high_resolution_clock::now();
            temp_X(0, i-N+1) = t(0,0);

        }
        X = temp_X.st();
    }

    return X;
}





// modifying =========================================================================================
cx_mat General_APC::algorithm2(cx_mat s, cx_mat  y_noisy, double N, mat alpha, double sigma)
{


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
	std::vector<arma::cx_mat> Ss(25);                   // size: 25 * 13 * 13
	for(int k = 0; k < 2*N-1; k++) {
        Ss[k] = S.col(k) * S.col(k).t(); 
    }
	// =============================================================


    cx_mat sum_s = zeros<cx_mat>(N,N);                  // size: 13 * 13
    for (int var = 0; var < 2*N-1; ++var) {
        sum_s = sum_s + S.col(var) * S.col(var).t();
    }

    cx_mat W_int = inv<cx_mat>(sum_s)*s;                // size: 13 * 1
    cx_mat temp_X = zeros<cx_mat>(y_noisy.size()-(N-1),1);

    for (int var = 0; var < y_noisy.size()-(N-1) - 1; ++var) {
        temp_X.row(var) = W_int.t() * y_noisy.submat(var,0,var+N-1,0);
    }

    cx_mat R = eye<cx_mat>(N,N) * sigma;
    cx_mat X = temp_X;



	int sample_test = 12 + 0; 


    // Get the starting timepoint
	auto start = std::chrono::high_resolution_clock::now();


	// modifying =============
	cx_mat apc_test; 
	// =======================

    for (int j = 0; j < M; ++j) {									// 2x
        
		// std::cout << "X size: " << X.size() << std::endl; 
		temp_X = zeros<cx_mat>(1,X.size()-2*N+2);
		cx_mat rho = elementwisePow(X, alpha(j,0));
		//apc_test = rho; 


        for (int i = N-1; i < X.size()-N+1; ++i) {  // upper: X.size()-N+1  // 262x and 238x
            //cx_mat temp_rho = rho.submat(i-N+1,0,i+N-1,0);
            cx_mat C = zeros<cx_mat>(N,N);

			// time1
			auto time1 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < 2*N-1; ++k) {						// 2N-1  25x
                C = rho(i-N+1+k,0)*Ss[k] + C;
            }

			// time2
			auto time2 = std::chrono::high_resolution_clock::now();


            cx_mat W = inv(C+R)*s*rho(i,0);



        }
        X = temp_X.st(); 
    }
    return X;
}
// =====================================================================================================





cx_mat General_APC::elementwisePow(cx_mat input, double p)
{
    cx_mat out = input;
    for (int i = 0; i < input.n_rows; ++i) {
        for (int j = 0; j < input.n_cols; ++j) {
            out(i,j) = pow(input(i,j), p);
        }
    }
    return out;
}
