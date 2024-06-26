#include "../include/mainwindow.h"
#include <iostream>
#include <armadillo>




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




int main() {

    // Load the matrix from the CSV file
    arma::cx_mat y_noisy2;
	y_noisy2.load("./data/HighSNR_Target_Mask_20dBWeakerTarget_Speed_10mPerS_MF_in_Q.csv", arma::csv_ascii);
    std::cout << "y_noisy2 shape:\n" << y_noisy2.n_rows << " " << y_noisy2.n_cols << std::endl;
	arma::mat realPart = arma::real(y_noisy2);
	y_noisy2.set_imag(realPart);
	y_noisy2 = y_noisy2.st(); 

	// start coding CUDA =============================

    int temp=0;
    s.resize(13,1);
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;

	////
	alpha = mat(2, 1); 
	alpha(0,0) = 1.9;
	alpha(1,0) = 1.8;
	//alpha(2,0) = 1.7; 



	// allocate memory in CPU for calculation
	double* y_n_real;
	double* y_n_imag; 
	double* X_real; 
	double* X_imag;
	//double* rho_real; 
	//double* rho_imag; 
	double* R_real; 
	double* R_imag;
	double* Ss_real; 
	double* Ss_imag; 
	double* s_real; 
	double* s_imag;
	double* alpha_real; 

	double* output_real;
	double* output_imag;

	double* test_real;
	double* test_imag; 

	// define our variables
	batch_size = 0; 
	data_num = y_noisy2.n_cols; 
	int y_n_size; 
	int X_size; 
	int R_row; 
	int Ss_size;
	int s_size;
	int alpha_size; 
	int print_flag = 1;						// print report from GPU function 





	// prepare our GPU format data
	// convert cx_mat of armadillo library to ordinary arrays.
	



	fun1(s, 
		 y_noisy2, 
		 13, 
		 alpha, 
		 1e-5, 
		 &y_n_real, 
		 &y_n_imag, 
		 &X_real, 
		 &X_imag, 
	//	 &rho_real,
	//	 &rho_imag, 
		 &R_real, 
		 &R_imag,
		 &Ss_real,
		 &Ss_imag,
		 &s_real,
		 &s_imag,
		 &alpha_real,
		 &output_real,
		 &output_imag, 
		 &test_real,
		 &test_imag, 

		 batch_size, 
		 data_num,
		 y_n_size,
		 X_size, 
		 R_row, 
		 Ss_size, 
		 s_size, 
		 alpha_size
		 );


	// Get the starting timepoint
    auto start22 = std::chrono::high_resolution_clock::now();

	// main GPU kernel
	gpuKernel(y_n_real,
			  y_n_imag,
			  X_real,
			  X_imag,
//			  rho_real, 
//			  rho_imag, 
			  R_real,
			  R_imag,
			  Ss_real,
			  Ss_imag,
			  s_real,
			  s_imag, 
			  alpha_real,
			  output_real,
			  output_imag, 
			  test_real, 
			  test_imag,

			  batch_size, 
			  data_num, 
			  y_n_size, 
			  X_size, 
			  R_row, 
			  Ss_size, 
			  s_size, 
			  alpha_size
			  );


	
	// Get the ending timepoint
    auto stop22 = std::chrono::high_resolution_clock::now();

	// calculate time processing
	auto du2 = std::chrono::duration_cast<std::chrono::microseconds>(stop22 - start22);
	std::cout << "GPUUUU Time: " << du2.count() / 1000.0 << " miliseconds" << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl; 


	// store result of GPU in an armadillo cx_mat variable
	arma::mat real_part(output_real, X_size, data_num * batch_size); 
	arma::mat imag_part(output_imag, X_size, data_num * batch_size); 
	gpu_apc = arma::cx_mat(real_part, imag_part);
	gpu_apc = gpu_apc.st();


	// modifying =========================================
	arma::mat test_real_part(test_real, R_row * R_row, X_size * data_num * batch_size);
	arma::mat test_imag_part(test_imag, R_row * R_row, X_size * data_num * batch_size);
	gpu_test_apc = arma::cx_mat(test_real_part, test_imag_part); 
	gpu_test_apc = gpu_test_apc.st(); 
	// ===================================================


	// free the memory that we use
	free(y_n_real);
	free(y_n_imag); 
	free(X_real); 
	free(X_imag);
	free(R_real); 
	free(R_imag);
	free(Ss_real); 
	free(Ss_imag); 
	free(s_real); 
	free(s_imag); 
	free(alpha_real); 
	free(output_real); 
	free(output_imag); 





	// Get the ending timepoint
    auto start = std::chrono::high_resolution_clock::now();
	clock_t t1 = clock(); 


	// modifying ==========================================
	while (my_indx < y_noisy2.n_cols) {
	// ====================================================



	std::cout << "indx: " << my_indx << std::endl;
	//std::cout << "y_noisy size: " << y_noisy2.size() << std::endl; 
	// exclude our target y_noisy
	y_noisy = y_noisy2.col(my_indx); 


   


				// modifying ====================================
				/*if (my_indx == 0) {	
				std::cout << " ------ " << std::endl; 
				std::cout << "y noisy : " << std::endl; 
				for(int i=0; i<1; i++){
					for(int j=0; j<gpu_apc.n_cols; j++){
						std::cout << "y_noisy(" << i << ", " << j << "): ";
						std::cout << gpu_apc(i,j) << "\n";
					}
					std::cout << std::endl;
				}
				}*/
				// =============================================
				



                General_APC apc;

                cx_mat result_apc = apc.algorithm(s, y_noisy, 13, alpha, 1e-5);

                cx_mat apc_test = apc.algorithm2(s, y_noisy, 13, alpha, 1e-5);

				for (int k=0; k<batch_size; ++k) {
					double mse = calc_mse(gpu_apc, result_apc.st(), my_indx + k * data_num, 238);



				/*	if (my_indx == 0 && k == 0) {
						std::cout << "acp_test : " << std::endl; 
						for(int i=0; i<apc_test.n_rows; i++){
							for(int j=0; j<apc_test.n_cols; j++){
								std::cout << "apc_test(" << i << ", " << j << "): ";
								std::cout << apc_test(i,j) << "\n";
							}
							std::cout << std::endl;
						}
					}*/


					/*if (my_indx == 3) {
					double test_mse = 0.0;
					double test_mse_acc = 0.0; 
					for (int count0 = 0; count0 < 238; ++count0) {
						test_mse = calc_mse(gpu_test_apc, apc_test.col(count0).st(), count0 + (my_indx + k * data_num) * 285, 169); 

						test_mse_acc += calc_mse(gpu_test_apc, apc_test.col(count0).st(), count0 + (my_indx + k * data_num) * 285, 169); 
				
						std::cout << "test MSE(" << count0 << "): " << test_mse << std::endl; 
						std::cout << "test ACC MSE(" << count0 << "): " << test_mse_acc << std::endl;
						std::cout << "-------------" << std::endl; 
					}
					}*/

					std::cout << "MSE(" << k << "): " << mse << std::endl;
				}
				/*std::cout << "abs: " << apc_test.n_cols << " " << apc_test.n_rows << std::endl;
				std::cout << "hsn: " << gpu_test_apc.n_cols << " " << gpu_test_apc.n_rows << std::endl;
				std::cout << "gpu: " << gpu_apc.n_cols << " " << gpu_apc.n_rows << std::endl; 
				std::cout << "res: " << result_apc.n_cols << " " << result_apc.n_rows << std::endl; 
				std::cout << "-------------" << std::endl; */



				/*std::cout << "gpu_test_apc : " << std::endl; 
				for(int i=0; i<1; i++){
					for(int j=0; j<gpu_test_apc.n_cols; j++){
						std::cout << "gpu_test_apc(" << i << ", " << j << "): ";
						std::cout << gpu_test_apc(i,j) << "\n";
					}
					std::cout << std::endl;
				}*/


				/*std::cout << "apc_test : " << std::endl; 
				for(int i=0; i<apc_test.n_rows; i++){
					for(int j=1; j<2; j++){
						std::cout << "apc_test(" << i << ", " << j << "): ";
						std::cout << apc_test(i,j) << "\n";
					}
					std::cout << std::endl;
				}*/




				// Check if the two matrices are equal
				/*if(arma::approx_equal(result_apc, result_apc2, "absdiff", 0.0001)) {
					std::cout << "The matrices are equal." << std::endl;
				} else {
					std::cout << "The matrices are not equal." << std::endl;
				}*/





				// modifying ==================================
				/*if (my_indx == 3) {
				std::cout << "result apc : " << std::endl; 
				for(int i=0; i<result_apc.n_rows; i++){
					for(int j=0; j<result_apc.n_cols; j++){
						std::cout << "result_apc(" << i << ", " << j << "): ";
						std::cout << result_apc(i,j) << "\n";
					}
					std::cout << std::endl;
				}
				}*/
				//std::cout << "=========================" << std::endl;
				// ============================================






	// modifying ====================================
	my_indx ++; 
	}
	// ==============================================
	
	// Get the ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();
	clock_t t2 = clock(); 


	// calculate time processing
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "CPU: " << duration.count() << " microseconds" << std::endl;
	printf("CPU: %g\n", (t2-t1)/1000.0); 
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl; 
	















    return 0;
}




