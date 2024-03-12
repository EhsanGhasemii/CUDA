#include "../include/general_apc.h"
#include <QDebug>

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


	// modifying ================================= 
	//std::cout << " ------ " << std::endl; 
	//std::cout << " Now we are in general APC: " << std::endl; 
	/*std::cout << "y noisy : " << std::endl; 
	for(int i=0; i<y_noisy.n_rows; i++){
		for(int j=0; j<y_noisy.n_cols; j++){
			std::cout << "y_noisy(" << i << ", " << j << "): ";
			std::cout << y_noisy(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================



    cx_mat S = zeros<cx_mat>(N,2*N-1);

    cx_mat temp = s;
    for (int var = N-1; var < 2 * N - 1; ++var) {
        S.col(var) = temp;
        temp = join_cols(zeros<cx_mat>(1,1), temp.submat(0,0,N-2,0));
    }



	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "M : " << M << std::endl;
	std::cout << "alpha size : " << alpha.size() << std::endl; */
	// ===========================================



	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "S : " << std::endl; 
	for(int i=0; i<S.n_rows; i++){
		for(int j=0; j<S.n_cols; j++){
			std::cout << "S(" << i << ", " << j << "): ";
			std::cout << S(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================





    temp = s;
    for (int var = N-2; var >=0; --var) {
        temp = join_cols(temp.submat(1,0,N-1,0),zeros<cx_mat>(1,1));
        S.col(var) = temp;
    }



	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "S : " << std::endl; 
	for(int i=0; i<S.n_rows; i++){
		for(int j=0; j<S.n_cols; j++){
			std::cout << "S(" << i << ", " << j << "): ";
			std::cout << S(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================


    cx_mat sum_s = zeros<cx_mat>(N,N);
    for (int var = 0; var < 2*N-1; ++var) {
        sum_s = sum_s + S.col(var) * S.col(var).t();
    }


	// modifying ================================
    /*cx_mat abas = {
		{std::complex<double>(1), std::complex<double>(1)},
		{std::complex<double>(2), std::complex<double>(2)},
		{std::complex<double>(3), std::complex<double>(3)}
	};
	cx_mat abas2 = {
		{std::complex<double>(5), std::complex<double>(5), std::complex<double>(4)}, 
		{std::complex<double>(3), std::complex<double>(2), std::complex<double>(1)}
	};*/
	//cx_mat abas3 = zeros<cx_mat>(1, 1); 
	//cx_mat abas3 = inv<cx_mat>(abas) * abas2; 
	//cx_mat abas1 = zeros<cx_mat>(2, 2);
	/*for (int var = 0; var < 3; ++var) {
		abas1 = abas1 + abas.col(var) * abas.col(var).t();
	}*/
	// =========================================





	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "abas : " << std::endl;						// size: 13 * 13
	for(int i=0; i<abas.n_rows; i++){
		for(int j=0; j<abas.n_cols; j++){
			std::cout << "abas(" << i << ", " << j << "): ";
			std::cout << abas(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================





    cx_mat W_int = inv<cx_mat>(sum_s)*s;
    cx_mat temp_X = zeros<cx_mat>(y_noisy.size()-(N-1),1);

    for (int var = 0; var < y_noisy.size()-(N-1) - 1; ++var) {
        temp_X.row(var) = W_int.t() * y_noisy.submat(var,0,var+N-1,0);
    }

    cx_mat R = eye<cx_mat>(N,N) * sigma;
    cx_mat X = temp_X;



	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "W_int : " << std::endl;						// size: 13 * 1
	for(int i=0; i<W_int.n_rows; i++){
		for(int j=0; j<W_int.n_cols; j++){
			std::cout << "W_int(" << i << ", " << j << "): ";
			std::cout << W_int(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << " ------ " << std::endl; 
	std::cout << "temp_X : " << std::endl;						// size 286 * 1		
	for(int i=0; i<temp_X.n_rows; i++){
		for(int j=0; j<temp_X.n_cols; j++){
			std::cout << "temp_X(" << i << ", " << j << "): ";
			std::cout << temp_X(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << " ------ " << std::endl; 
	std::cout << "R : " << std::endl;							// size: 13 * 13
	for(int i=0; i<R.n_rows; i++){
		for(int j=0; j<R.n_cols; j++){
			std::cout << "R(" << i << ", " << j << "): ";
			std::cout << R(i,j) << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << " ------ " << std::endl; 
	std::cout << "X : " << std::endl;							// size: 286 * 1
	for(int i=0; i<X.n_rows; i++){
		for(int j=0; j<X.n_cols; j++){
			std::cout << "X(" << i << ", " << j << "): ";
			std::cout << X(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================


	
    // Get the starting timepoint
	auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < M; ++j) {									// 2x
        temp_X = zeros<cx_mat>(1,X.size()-2*N+2);
        cx_mat rho = elementwisePow(X, alpha(j,0));
		
        for (int i = N-1; i < X.size()-N+1; ++i) {					// 262x and 238x
            cx_mat temp_rho = rho.submat(i-N+1,0,i+N-1,0);
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


			/*auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
			std::cout << "Time1: " << duration1.count() << " microseconds" << std::endl;
			auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2);
			std::cout << "Time2: " << duration2.count() << " microseconds" << std::endl;
			auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3);
			std::cout << "Time3: " << duration3.count() << " microseconds" << std::endl;*/







        }
        X = temp_X.t(); 
    }

	
	// Get the ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();

	// calculate time processing
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Time taken by algorithm1: " << duration.count() << " microseconds" << std::endl;




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



	// modifying ================================= 
	/*std::cout << " ------ " << std::endl; 
	std::cout << "X : " << std::endl; 
	for(int i=0; i<X.n_rows; i++){
		for(int j=0; j<X.n_cols; j++){
			std::cout << "X(" << i << ", " << j << "): ";
			std::cout << X(i,j) << "\t";
		}
		std::cout << std::endl;
	}*/
	// ===========================================


	int sample_test = 12 + 1; 


    // Get the starting timepoint
	auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < M; ++j) {									// 2x
        temp_X = zeros<cx_mat>(1,X.size()-2*N+2);
		cx_mat rho = elementwisePow(X, alpha(j,0));


        for (int i = N-1; i < X.size()-N+1; ++i) {  // upper: X.size()-N+1  // 262x and 238x
            //cx_mat temp_rho = rho.submat(i-N+1,0,i+N-1,0);
            cx_mat C = zeros<cx_mat>(N,N);

			// time1
			auto time1 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < 2*N-1; ++k) {						// 25x
                C = rho(i-N+1+k,0)*Ss[k] + C;
            }

			// time2
			auto time2 = std::chrono::high_resolution_clock::now();
            cx_mat W = inv(C+R)*s*rho(i,0);


			// modifying ================================= 
			/*if (i == sample_test) {
				//cx_mat CR = y_noisy.submat(i+(N-1)*(j), 0, (j)*(N-1)+i+N-1, 0);
				cx_mat CR = inv(C+R); 
				std::cout << " ------ " << std::endl; 
				std::cout << "C+R: " << std::endl; 
				for(int i=0; i<CR.n_rows; i++){
					for(int j=0; j<CR.n_cols; j++){
						std::cout << "CR(" << i << ", " << j << "): ";
						std::cout << CR(i,j) << "\t";
					}
					std::cout << std::endl;
				}
			}*/
			// ===========================================


			// time3 
			auto time3 = std::chrono::high_resolution_clock::now();
            cx_mat t = W.t() * y_noisy.submat(i+(N-1)*(j),0,(j)*(N-1)+i+N-1,0);

			// modifying =================================
			if (i < X.size()) {
				std::cout << "i: " << i-12 << " - Finall Result: " << t << std::endl;
			}
			// ===========================================

			// time4 
			auto time4 = std::chrono::high_resolution_clock::now();
            temp_X(0, i-N+1) = t(0,0);


/*
			auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1);
			std::cout << "Time1: " << duration1.count() << " microseconds" << std::endl;
			auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2);
			std::cout << "Time2: " << duration2.count() << " microseconds" << std::endl;
			auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3);
			std::cout << "Time3: " << duration3.count() << " microseconds" << std::endl;
*/




        }
        X = temp_X.t(); 
    }

	// Get the ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();

	// calculate time processing
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Time taken by algorithm2: " << duration.count() << " microseconds" << std::endl;



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
