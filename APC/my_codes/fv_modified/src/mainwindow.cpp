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

//		std::cout << data1(row, i).real() << " " << data1(row, i).imag() << std::endl; 
//		std::cout << data2(0, i).real() << " " << data2(0, i).imag() << std::endl; 
//		std::cout << "mse: " << mse << std::endl; 
//		std::cout << "------------------------" << std::endl; 

	}
	return mse; 
}


int main() { 

	// define batch size 
	batch_size = 1;				 // batch size is for previous versions.
	int offset = 0;              // 0 in cpp is equal to 1 in matlab 
	int element_len = 98;
	int out_len = element_len - (13 - 1); 
	double sigma = 1e-5;

	// load the matrix from the CSV files
	std::vector<arma::cx_mat> my_vec;
	//arma::cx_mat y_noisy2;
	arma::mat y_noisy2_real;
	arma::mat y_noisy2_imag; 


	std::string name;
	std::string name_i; 
	std::string name_q; 

	name = "HighSNR_Target_Mask_20dBWeakerTarget_Speed_1mPerS"; 

	name_i = "./data_in/" + name + "_I.csv"; 
	y_noisy2_real.load(name_i, arma::csv_ascii); 

	name_q = "./data_in/" + name + "_Q.csv"; 
	y_noisy2_imag.load(name_q, arma::csv_ascii); 

	y_noisy2 = arma::cx_mat(y_noisy2_real, y_noisy2_imag) / std::pow(2.0, 10);
	y_noisy2 = y_noisy2.st();


	y_noisy2 = y_noisy2.submat(offset, 0, offset + element_len, y_noisy2.n_cols - 1); 
	//y_noisy2 = y_noisy2.submat(0, 0, y_noisy2.n_rows-1, 0); 


	my_vec.push_back(y_noisy2); 


	

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
//	alpha(2,0) = 1.7; 




	// allocate memory in CPU for calculation
	float* y_n_real;
	float* y_n_imag; 
	double* X_real; 
	double* X_imag;
	//double* rho_real; 
	//double* rho_imag; 
	float* R_real; 
	float* R_imag;
	float* Ss_real; 
	float* Ss_imag; 
	float* s_real; 
	float* s_imag;
	float* alpha_real; 

	double* output_real;
	double* output_imag;

//	double* test_real;
//	double* test_imag; 



	// define our variables
	data_num = y_noisy2.n_cols; 
	int y_n_size; 
	int X_size; 
	int R_row; 
	int Ss_size;
	int s_size;
	int alpha_size; 
	int print_flag = 1;						// print report from GPU function
	int num_iter = 1000; 





	fun1(s, 
		 my_vec, 
		 13, 
		 alpha, 
		 1e-5, 
		 &y_n_real, 
		 &y_n_imag, 
		 &X_real, 
		 &X_imag, 
		 &R_real, 
		 &R_imag,
		 &Ss_real,
		 &Ss_imag,
		 &s_real,
		 &s_imag,
		 &alpha_real,
		 &output_real,
		 &output_imag, 

		 batch_size, 
		 data_num + 1,
		 y_n_size,
		 X_size, 
		 R_row, 
		 Ss_size, 
		 s_size, 
		 alpha_size, 
		 num_iter
		 );


	// main GPU kernel
	gpuKernel(y_n_real,
			  y_n_imag,
			  X_real,
			  X_imag,
			  R_real,
			  R_imag,
			  Ss_real,
			  Ss_imag,
			  s_real,
			  s_imag, 
			  alpha_real,
			  output_real,
			  output_imag, 

			  batch_size, 
			  data_num + 1, 
			  y_n_size, 
			  X_size, 
			  R_row, 
			  Ss_size, 
			  s_size, 
			  alpha_size, 
			  num_iter
			  );


	// store result of GPU in an armadillo cx_mat variable
	arma::mat real_part(output_real, X_size, data_num * batch_size); 
	arma::mat imag_part(output_imag, X_size, data_num * batch_size); 
	gpu_apc = arma::cx_mat(real_part, imag_part);
	gpu_apc = gpu_apc.st();



	// load output result of Matlab calculation to check our result
/*	arma::cx_mat gpu_apc; 
	arma::mat gpu_apc_real; 
	arma::mat gpu_apc_imag; 

	name_i = "./data_result/Result" + name + "_I.csv"; 
	gpu_apc_real.load(name_i, arma::csv_ascii); 

	name_q = "./data_result/Result" + name + "_Q.csv"; 
	gpu_apc_imag.load(name_q, arma::csv_ascii); 

	gpu_apc = arma::cx_mat(gpu_apc_real, gpu_apc_imag);  // gpu apc shape: data_num * out_size: (99 * 87)
*/



	// =======================================
	while (my_indx < y_noisy2.n_cols) {  
	
		// print output
		/*std::cout << "y_noisy : " << std::endl; 
		for(int i=0; i<y_noisy2.n_rows; i++){
			for(int j=0; j<y_noisy2.n_cols; j++){
				std::cout << "out(" << i << ", " << j << "): ";
				std::cout << y_noisy2(i,j) << "\n";
			}
			std::cout << std::endl;
		}*/


		General_APC apc;

		y_noisy = y_noisy2.col(my_indx); 
		cx_mat result_apc = apc.algorithm(s, y_noisy, 13, alpha, sigma);

		std::cout << "my_indx: " << my_indx << std::endl; 

		double mse = calc_mse(gpu_apc, result_apc.st(), my_indx , out_len);

		std::cout << "MSE: " << mse << std::endl;
		std::cout << "----------" << std::endl; 


/*		result_apc = result_apc.st(); 
		std::cout << "result_apc : " << std::endl; 
		for(int i=0; i<result_apc.n_rows; i++){
			for(int j=0; j<result_apc.n_cols; j++){
				std::cout << "out(" << i << ", " << j << "): ";
				std::cout << result_apc(i,j) << "\n";
			}
			std::cout << std::endl;
		}*/

		my_indx ++; // go to the next data row
	}


	return 0; 
}

