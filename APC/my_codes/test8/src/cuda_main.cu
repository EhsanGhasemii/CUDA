#include "../include/cuda_main.h"
#include "../include/gpuerrors.h"
#include "../include/gputimer.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>





// define our functions ======================================
void fill(float* data, int size);
dim3 getDimGrid(const int n); 
dim3 getDimBlock(const int n); 
__global__ void kernelFunc(double* X_reald, 
						   double* X_imagd,
						   double* R_reald,
						   double* R_imagd,
						   double* Ss_reald,
						   double* Ss_imagd,
						   double* s_reald,
						   double* s_imagd,
						   double* alpha_reald,
						   double* rho_reald, 
						   double* rho_imagd,
						   double* output_reald,
						   double* output_imagd, 

						   int X_size, 
						   int N, 
						   double* testd
						   );
__global__ void matrixInversion(double* inputd,
								double* outputd, 
								const int n, 
								const int m
								);	

__global__ void complexMatrixInversion(double* input_reald,		// input data is "inputd"
									   double* input_imagd,
									   //double* output_reald,    // output data is "outputd"
									   //double* output_imagd,
							           const int n,				// number of matrices is n
									   const int m,		     	// size of each matrix is m*m

									   double* y_n_reald, 
									   double* y_n_imagd, 
									   double* s_reald, 
									   double* s_imagd, 
									   double* rho_reald, 
									   double* rho_imagd, 
									   double* W_reald, 
									   double* W_imagd, 
									   double* W_reald_shr2, 
									   double* W_imagd_shr2, 
									   double* W_reald_shr, 
									   double* W_imagd_shr
									   );			     		// we suppose input data is squre matrix


__global__ void kernelFunc3(double* y_n_reald, 
							double* y_n_imagd,
							double* CRinv_reald,	
							double* CRinv_imagd,
							double* s_reald,   
							double* s_imagd,
							double* rho_reald, 
							double* rho_imagd, 
							double* W_reald, 
							double* W_imagd,
							const int n,			
							const int m		     	
							);		     		

void print_matrix(char* name, double* data, int size, int d_shift);
// ==========================================================


// main body
void gpuKernel(double* y_n_real,
			   double* y_n_imag,
			   double* X_real, 
			   double* X_imag, 
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
			   ) {
	
	// print name of device
	struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);


	// define our variabels
	int N = 13; 
	int print_flag = 1; 

	// allocate memory in CPU for calculation

	// define our needed variables in GPU
	double* y_n_reald; 
	double* y_n_imagd; 
	double* X_reald;
	double* X_imagd; 
	double* R_reald;
	double* R_imagd; 
	double* Ss_reald; 
	double* Ss_imagd; 
	double* s_reald; 
	double* s_imagd;
	double* alpha_reald;

	double* rho_reald; 
	double* rho_imagd; 
	double* output_reald;
	double* output_imagd;
	double* W_reald;
	double* W_imagd;
	double* W_reald_shr2; 
	double* W_imagd_shr2; 
	double* W_reald_shr; 
	double* W_imagd_shr; 
	double* testd; 


	// allocation memory in GPU
	HANDLE_ERROR(cudaMalloc((void**)&y_n_reald, data_num * y_n_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&y_n_imagd, data_num * y_n_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&X_reald, data_num * X_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&X_imagd, data_num * X_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&R_reald, R_row * R_row * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&R_imagd, R_row * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&Ss_reald, Ss_size * R_row * R_row * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&Ss_imagd, Ss_size * R_row * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&s_reald, s_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&s_imagd, s_size * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&alpha_reald, 1 * sizeof(double))); // hard_code: 1 - ch..

	HANDLE_ERROR(cudaMalloc((void**)&rho_reald, data_num * X_size * sizeof(double))); // hard_code: 262 - ch..
	HANDLE_ERROR(cudaMalloc((void**)&rho_imagd, data_num * X_size * sizeof(double))); // hard_code: 262 - ch..
	HANDLE_ERROR(cudaMalloc((void**)&output_reald, data_num * X_size * R_row * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&output_imagd, data_num * X_size * R_row * R_row * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&W_reald, data_num * X_size * sizeof(double))); // hard_code: 13 - ch..
	HANDLE_ERROR(cudaMalloc((void**)&W_imagd, data_num * X_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&W_reald_shr2, data_num * X_size * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&W_imagd_shr2, data_num * X_size * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&W_reald_shr, data_num * X_size * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&W_imagd_shr, data_num * X_size * R_row * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&testd, data_num * X_size * R_row * R_row * sizeof(double))); 

	// copy array from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(y_n_reald, y_n_real, data_num * y_n_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(y_n_imagd, y_n_imag, data_num * y_n_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(X_reald, X_real, data_num * X_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(X_imagd, X_imag, data_num * X_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(R_reald, R_real, R_row * R_row * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(R_imagd, R_imag, R_row * R_row * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Ss_reald, Ss_real, Ss_size * R_row * R_row * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Ss_imagd, Ss_imag, Ss_size * R_row * R_row * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(s_reald, s_real, s_size * sizeof(double), cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(cudaMemcpy(s_imagd, s_imag, s_size * sizeof(double), cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(cudaMemcpy(alpha_reald, alpha_real, 1 * sizeof(double), cudaMemcpyHostToDevice)); // hard_code: 1 - ch..

	
	// define our threads and blocks dimension
	dim3 dimGrid = getDimGrid(data_num * X_size);
	dim3 dimBlock = getDimBlock(1);

	// transfer processing in CUDA
	double gpu_kernel_time = 0.0;
	GpuTimer timer;
    timer.Start();
	// APC algorithm part1 and part2
	kernelFunc<<< dimGrid,dimBlock >>>(X_reald, 
									   X_imagd, 
									   R_reald, 
									   R_imagd, 
									   Ss_reald, 
									   Ss_imagd, 
									   s_reald, 
									   s_imagd, 
									   alpha_reald, 
									   rho_reald, 
									   rho_imagd, 
									   output_reald, 
									   output_imagd, 

									   X_size, 
									   N, 
									   testd
									   );
	timer.Stop();
	gpu_kernel_time = timer.Elapsed();


	// APC algorithm part3
	dimGrid = getDimGrid(data_num * X_size); 
	dimBlock = getDimBlock(R_row * R_row); 
	complexMatrixInversion<<< dimGrid, dimBlock >>>(output_reald,
													output_imagd,  
													1, 
													13,

													y_n_reald, 
													y_n_imagd, 
													s_reald, 
													s_imagd, 
													rho_reald, 
													rho_imagd, 
													W_reald, 
													W_imagd, 
													W_reald_shr2, 
													W_imagd_shr2, 
													W_reald_shr, 
													W_imagd_shr
													);
													

	// modifying ====================================
	HANDLE_ERROR(cudaMemcpy(test, output_reald, data_num * X_size * R_row * R_row * sizeof(double), cudaMemcpyDeviceToHost));
	// ==============================================

	// copy result from GPU to CPU memory
	HANDLE_ERROR(cudaMemcpy(output_real, W_reald, data_num * X_size * sizeof(double), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_imag, W_imagd, data_num * X_size * sizeof(double), cudaMemcpyDeviceToHost)); 
	

	// print report
	if (print_flag) {
		std::cout << "this is output of GPU: " << std::endl; 
		char name[20] = "a"; 
		//print_matrix(name, a		 , n);

		strcpy(name, "output_real"); 
		print_matrix(name, output_real, X_size, 0 * X_size); 

		strcpy(name, "output_imag");
		print_matrix(name, output_imag, X_size, 0 * X_size); 

		strcpy(name, "test");
		print_matrix(name, test, 13, 1 * 13 * 13); 

		/*strcpy(name, "inv_real");
		print_matrix(name, inv_real, mat_size * mat_size, 0); 

		strcpy(name, "inv_imag");
		print_matrix(name, inv_imag, mat_size * mat_size, 0); */
		
	}




	// complex matrix inversion =========================================================================
	/*double* input_real; 
	double* input_imag;

	double* inv_real;
	double* inv_imag;

	int mat_size = 3; 

	input_real = (double*)malloc(mat_size * mat_size * sizeof(double)); 
	input_imag = (double*)malloc(mat_size * mat_size * sizeof(double));

	inv_real = (double*)malloc(mat_size * mat_size * sizeof(double));
	inv_imag = (double*)malloc(mat_size * mat_sRize * sizeof(double));

	input_real[0] = 1; 
	input_real[1] = 1; 
	input_real[2] = 2; 
	input_real[3] = 1; 
	input_real[4] = 0; 
	input_real[5] = 1; 
	input_real[6] = 0; 
	input_real[7] = 0; 
	input_real[8] = 0;

	input_imag[0] = 1; 
	input_imag[1] = 0; 
	input_imag[2] = 0; 
	input_imag[3] = 1; 
	input_imag[4] = 2; 
	input_imag[5] = 0; 
	input_imag[6] = 0; 
	input_imag[7] = 0; 
	input_imag[8] = 2; 

	double* input_reald;
	double* input_imagd; 

	double* inv_reald; 
	double* inv_imagd;

	HANDLE_ERROR(cudaMalloc((void**)&input_reald, mat_size * mat_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&input_imagd, mat_size * mat_size * sizeof(double)));

	HANDLE_ERROR(cudaMalloc((void**)&inv_reald, mat_size * mat_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&inv_imagd, mat_size * mat_size * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(input_reald, input_real, mat_size * mat_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(input_imagd, input_imag, mat_size * mat_size * sizeof(double), cudaMemcpyHostToDevice));

	dimGrid = getDimGrid(1); 
	dimBlock = getDimBlock(mat_size * mat_size); 

	complexMatrixInversion<<< dimGrid, dimBlock >>>(input_reald, input_imagd, inv_reald, inv_imagd, 1, mat_size); 

	HANDLE_ERROR(cudaMemcpy(inv_real, inv_reald, mat_size * mat_size * sizeof(double), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(inv_imag, inv_imagd, mat_size * mat_size * sizeof(double), cudaMemcpyDeviceToHost));*/
	// ==================================================================================================



	// remove array in GPU
	HANDLE_ERROR(cudaFree(X_reald));
	HANDLE_ERROR(cudaFree(X_imagd));
	HANDLE_ERROR(cudaFree(R_reald));
	HANDLE_ERROR(cudaFree(R_imagd));
	HANDLE_ERROR(cudaFree(Ss_reald));
	HANDLE_ERROR(cudaFree(Ss_imagd));
	HANDLE_ERROR(cudaFree(s_reald));
	HANDLE_ERROR(cudaFree(s_imagd));
	HANDLE_ERROR(cudaFree(alpha_reald));
	HANDLE_ERROR(cudaFree(output_reald));
	HANDLE_ERROR(cudaFree(output_imagd));



	// print a report
	std::cout << "I am in gpuKernel .." << std::endl;


}



// 
dim3 getDimGrid(const int n) {
	dim3 dimGrid(n, 1, 1);

	return dimGrid;
}

//
dim3 getDimBlock(const int n) {
	dim3 dimBlock(n, 1, 1);

	return dimBlock;
}

// kernelFunc
__global__ void kernelFunc(double* X_reald, 
						   double* X_imagd,
						   double* R_reald,
						   double* R_imagd,
						   double* Ss_reald,
						   double* Ss_imagd,
						   double* s_reald,
						   double* s_imagd,
						   double* alpha_reald,
						   double* rho_reald, 
						   double* rho_imagd,
						   double* output_reald,
						   double* output_imagd, 

						   int X_size, 
						   int N, 
						   double* testd
						   ) {
	

	// define our variables
	// !!!!! Please consider that if your input data will be big 
	// you may need to use long format of variables !!!!!!!!!!!!
	//__shared__ double rho_reald_shr[262]; // hard_code: 262 - change it in future
	//__shared__ double rho_imagd_shr[262]; // hard_code: 262 - change it in future

	//__shared__ float Ss_reald_shr[25 * 13 * 13]; 
	//__shared__ float Ss_imagd_shr[25 * 13 * 13]; 

	//double C_reald[262 * 13 * 13]; 
	//double C_imagd[262 * 13 * 13]; 

	// define index of each thread
	long long i;
	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);

	int block_ind = i % X_size; 


	// !!!!!!!!!!!!!!!!!!!!!!!! this will be updated for multiple chunked data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	// create rho varible in shared memory
	if (block_ind <= X_size) {               // hard_code: 262 - change it in future
		double my_angle = atan2(X_imagd[i], X_reald[i]); 
		double my_radius = sqrt(X_imagd[i] * X_imagd[i] + X_reald[i] * X_reald[i]); 
		my_radius = pow(my_radius, alpha_reald[0]); 
		my_angle *= alpha_reald[0]; 

		rho_reald[i] = my_radius * cos(my_angle); 
		rho_imagd[i] = my_radius * sin(my_angle); 
	}


	// transfer Ss matrices in shared memory
	/*if (i < 13 * 13) {
		for (int count1 = 0; count1 < 25; ++count1) {
			Ss_reald_shr[count1 * 169 + i] = Ss_reald[count1 * 169 + i]; 
			Ss_imagd_shr[count1 * 169 + i] = Ss_imagd[count1 * 169 + i]; 
		}
	}*/
	
	// wait till all t172.17.0.0/16he data is ready
	__syncthreads(); 


	if ((blockIdx.x <= 261 - 24) || (blockIdx.x <= 261 + 261 - 24 && blockIdx.x >=261)) {     // blockIdx.x <= X_size - 2 * N + 2
		// first part of the algorithm: 25 * (matrix multilplication and addition)
		for (int count1 = 0; count1 < 25; ++count1) {
			for (int count2 = 0; count2 < 13; ++count2) {
				for (int count3 = 0; count3 < 13; ++count3) {         // hard_code: (i-12) in both lines
					output_reald[blockIdx.x * 169 + count2 * 13 + count3] += (rho_reald[blockIdx.x + count1] * Ss_reald[count1 * 169 + count2 * 13 + count3]
																		    - rho_imagd[blockIdx.x + count1] * Ss_imagd[count1 * 169 + count2 * 13 + count3]); 
					output_imagd[blockIdx.x * 169 + count2 * 13 + count3] += (rho_reald[blockIdx.x + count1] * Ss_imagd[count1 * 169 + count2 * 13 + count3]
																		    + rho_imagd[blockIdx.x + count1] * Ss_reald[count1 * 169 + count2 * 13 + count3]);
				}
			}
		}

		// second part of the algorithm: C += R
		for (int count2 = 0; count2 < 13; ++count2) {
			for (int count3 = 0; count3 < 13; ++count3) {
				output_reald[blockIdx.x * 169 + count2 * 13 + count3] += R_reald[count2 * 13 + count3]; 
				output_imagd[blockIdx.x * 169 + count2 * 13 + count3] += R_imagd[count2 * 13 + count3];
			}
		}


		// save rho to use it in next steps
		//rho_reald[i] = rho_reald_shr[i];
		//rho_imagd[i] = rho_imagd_shr[i]; 

	}


}

__global__ void complexMatrixInversion(double* input_reald,		// input data is "inputd"
									   double* input_imagd,
							           const int n,			    // number of matrices is n
									   const int m,		     	// size of each matrix is m*m

									   double* y_n_reald, 
									   double* y_n_imagd, 
									   double* s_reald, 
									   double* s_imagd, 
									   double* rho_reald, 
									   double* rho_imagd, 
									   double* W_reald, 
									   double* W_imagd, 
									   double* W_reald_shr2, 
									   double* W_imagd_shr2, 
									   double* W_reald_shr, 
									   double* W_imagd_shr


									   ) {			     		// we suppose input data is squre matrix


	// define our variables
	__shared__ double out_real[30 * 30];
	__shared__ double out_imag[30 * 30];

	__shared__ double out_real_shr[30 * 30]; 
	__shared__ double out_imag_shr[30 * 30]; 

	__shared__ double in_real[30 * 30]; 
	__shared__ double in_imag[30 * 30]; 
	

	// define index of each thread
	long long i;
	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);


	// data and thread location
	int mat_num = i / (m * m); 
	int mat_ind = i % (m * m); 
	int mat_row = (i % (m * m)) / m; 
	int mat_col = (i % (m * m)) % m; 





	if (threadIdx.x < 169) {          // i < 169 

		// transfer input data to shared memory
		in_real[mat_ind] = input_reald[i]; 
		in_imag[mat_ind] = input_imagd[i]; 


		// creating eye matrix for gauss jordan elimination
		if (mat_row == mat_col) {	
			out_real[mat_ind] = 1.0; 
			out_imag[mat_ind] = 0.0; 
		}
		else {
			out_real[mat_ind] = 0.0; 
			out_imag[mat_ind] = 0.0; 
		}


		


		// Matrix inversion algorithm main body ======================================== 
		// we use Gauss Jordan Algorithm
		// algorithm: part1 - make the input data upper-triangular
		for (int count1 = 0; count1 < m - 1; ++count1) {
			
			// change current row when its pivot is zero
			if ((input_reald[blockIdx.x * 169 + count1 * m + count1] == 0) && (input_imagd[blockIdx.x * 169 + count1 * m + count1] == 0)) {
				int count2 = count1 + 1; 
				while ((input_reald[blockIdx.x * 169 + count2 * m + count1] == 0) && (input_imagd[blockIdx.x * 169 + count2 * m + count1] == 0) && (count2 < m)) {
					++count2;
				}
				if(mat_row == count1) {
					input_reald[blockIdx.x * 169 + mat_ind] += input_reald[blockIdx.x * 169 + count2 * m + mat_col]; // ch ..
					input_imagd[blockIdx.x * 169 + mat_ind] += input_imagd[blockIdx.x * 169 + count2 * m + mat_col]; // ch ..

					out_real[mat_ind] += out_real[count2 * m + mat_col]; 
					out_imag[mat_ind] += out_imag[count2 * m + mat_col]; 
				}
				__syncthreads(); 	
			}

			if (mat_row > count1) {
				double mul_real = in_real[mat_row * m + count1] * in_real[count1 * m + count1]
								+ in_imag[mat_row * m + count1] * in_imag[count1 * m + count1];
				mul_real /= (in_real[count1 * m + count1] * in_real[count1 * m + count1]
						   + in_imag[count1 * m + count1] * in_imag[count1 * m + count1]); 

				double mul_imag = in_imag[mat_row * m + count1] * in_real[count1 * m + count1]
								- in_real[mat_row * m + count1] * in_imag[count1 * m + count1];
				mul_imag /= (in_real[count1 * m + count1] * in_real[count1 * m + count1]
						   + in_imag[count1 * m + count1] * in_imag[count1 * m + count1]); 

				in_real[mat_ind] -= (mul_real * in_real[count1 * m + mat_col]
								   - mul_imag * in_imag[count1 * m + mat_col]); 
				in_imag[mat_ind] -= (mul_real * in_imag[count1 * m + mat_col]
								   + mul_imag * in_real[count1 * m + mat_col]);

				out_real[mat_ind] -= (mul_real * out_real[count1 * m + mat_col]
									- mul_imag * out_imag[count1 * m + mat_col]);
				out_imag[mat_ind] -= (mul_real * out_imag[count1 * m + mat_col]
									+ mul_imag * out_real[count1 * m + mat_col]);
			}

			// wait till all the data is changed
			__syncthreads(); 
		}


		// algorithm: part2 - make the input data lower-triangular
		for (int count1 = m - 1; count1 > 0; --count1) {
			if (mat_row < count1) {
				double mul_real = in_real[mat_row * m + count1] * in_real[count1 * m + count1]
								+ in_imag[mat_row * m + count1] * in_imag[count1 * m + count1];
				mul_real /= (in_real[count1 * m + count1] * in_real[count1 * m + count1]
						   + in_imag[count1 * m + count1] * in_imag[count1 * m + count1]); 

				double mul_imag = in_imag[mat_row * m + count1] * in_real[count1 * m + count1]
								- in_real[mat_row * m + count1] * in_imag[count1 * m + count1];
				mul_imag /= (in_real[count1 * m + count1] * in_real[count1 * m + count1]
						   + in_imag[count1 * m + count1] * in_imag[count1 * m + count1]); 

				in_real[mat_ind] -= (mul_real * in_real[count1 * m + mat_col]
								   - mul_imag * in_imag[count1 * m + mat_col]); 
				in_imag[mat_ind] -= (mul_real * in_imag[count1 * m + mat_col]
								   + mul_imag * in_real[count1 * m + mat_col]);

				out_real[mat_ind] -= (mul_real * out_real[count1 * m + mat_col]
									- mul_imag * out_imag[count1 * m + mat_col]);
				out_imag[mat_ind] -= (mul_real * out_imag[count1 * m + mat_col]
									+ mul_imag * out_real[count1 * m + mat_col]);
			}

			// wait till all the data is changed
			__syncthreads(); 
		}

		// algorithm: part3 - normalize input data to create matrix inversion
		out_real_shr[mat_ind] = (out_real[mat_ind] * in_real[mat_row * m + mat_row]
							   + out_imag[mat_ind] * in_imag[mat_row * m + mat_row])
							   / (in_real[mat_row * m + mat_row] * in_real[mat_row * m + mat_row]
								+ in_imag[mat_row * m + mat_row] * in_imag[mat_row * m + mat_row]);
		
		out_imag_shr[mat_ind] = (out_imag[mat_ind] * in_real[mat_row * m + mat_row]
							   - out_real[mat_ind] * in_imag[mat_row * m + mat_row])
								/ (in_real[mat_row * m + mat_row] * in_real[mat_row * m + mat_row]
								 + in_imag[mat_row * m + mat_row] * in_imag[mat_row * m + mat_row]);

	} // if (i < 169)



	// ============================================================================

	// define our variables
/*	__shared__ double W_reald_shr[25]; 
	__shared__ double W_imagd_shr[25]; 
	__shared__ double W_reald_shr2[25]; 
	__shared__ double W_imagd_shr2[25];*/


	// =======================================
	//double W_reald_shr[30 * 13]; // ..........
	//double W_imagd_shr[30 * 13]; // ..........
	//double W_reald_shr2[30 * 13]; // ch ......
	//double W_imagd_shr2[30 * 13]; // ch .......
	// =======================================

	mat_num = 0; 


	// initialize shared memroy to zero
	if (threadIdx.x < 13) {        // i < 13
		W_reald_shr[blockIdx.x * 13 + threadIdx.x] = 0.0; // ch ...................
		W_imagd_shr[blockIdx.x * 13 + threadIdx.x] = 0.0; // ch ...................
	}

	__syncthreads(); 

	// APC algorithm pqrt4: inv(C+R) * s 
	if (threadIdx.x < 13) {            // hard_code: 13 - ch..
		for (int count1 = 0; count1 < 13; ++count1) {     // count1 < 13
			W_reald_shr[blockIdx.x * 13 + threadIdx.x] += out_real_shr[threadIdx.x * 13 + count1] * s_reald[count1]; 
			W_reald_shr[blockIdx.x * 13 + threadIdx.x] -= out_imag_shr[threadIdx.x * 13 + count1] * s_imagd[count1]; 

			W_imagd_shr[blockIdx.x * 13 + threadIdx.x] += out_real_shr[threadIdx.x * 13 + count1] * s_imagd[count1]; 
			W_imagd_shr[blockIdx.x * 13 + threadIdx.x] += out_imag_shr[threadIdx.x * 13 + count1] * s_reald[count1]; 
		}

	}

	__syncthreads(); 
	
	// APC algorithm part5: W = inv(C+R) * s * rho
	if (threadIdx.x < 13) {
		W_reald_shr2[blockIdx.x * 13 + threadIdx.x] = W_reald_shr[blockIdx.x * 13 + threadIdx.x] * rho_reald[blockIdx.x + 12]
									   			    - W_imagd_shr[blockIdx.x * 13 + threadIdx.x] * rho_imagd[blockIdx.x + 12];        // handle this for data_num > 1 
		W_imagd_shr2[blockIdx.x * 13 + threadIdx.x] = W_reald_shr[blockIdx.x * 13 + threadIdx.x] * rho_imagd[blockIdx.x + 12]
											  	    + W_imagd_shr[blockIdx.x * 13 + threadIdx.x] * rho_reald[blockIdx.x + 12];
	}

	__syncthreads(); 

	// modifying ====================================
	/*if (threadIdx.x < 13 && blockIdx.x == 1) {
		W_reald[threadIdx.x] = input_reald[i]; 
		W_imagd[threadIdx.x] = in_real[mat_ind]; 
	}*/
	// ==============================================


	// APC algorithm part6: W.t() * y_noisy
	if (threadIdx.x == 0) {			// hard_code: 13 - ch..

		W_reald[blockIdx.x] = 0.0; // 0.0
		W_imagd[blockIdx.x] = 0.0; // 0.0

		for (int count1 = 0; count1 < 13; ++count1) {        // count1 < 13 
			W_reald[blockIdx.x] += W_reald_shr2[blockIdx.x * 13 + count1] * y_n_reald[blockIdx.x + 12 + count1]
						 	     - W_imagd_shr2[blockIdx.x * 13 + count1] * y_n_imagd[blockIdx.x + 12 + count1];
			W_imagd[blockIdx.x] += W_reald_shr2[blockIdx.x * 13 + count1] * y_n_imagd[blockIdx.x + 12 + count1]
							     + W_imagd_shr2[blockIdx.x * 13 + count1] * y_n_reald[blockIdx.x + 12 + count1]; 
		}
	}

}





/*
__global__ void matrixInversion(double* inputd,		// input data is "inputd"
								double* outputd,	// output data is "outputd"
								const int n,				// number of matrices is n
								const int m				// size of each matrix is m*m
								) {					// we suppose input data is squre matrix




	// define our variables
	__shared__ double out[3 * 3];
	__shared__ double in[3 * 3]; 
	

	// define index of each thread
	long long i;
	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);


	// data and thread location
	int mat_num = i / (m * m); 
	int mat_ind = i % (m * m); 
	int mat_row = (i % (m * m)) / m; 
	int mat_col = (i % (m * m)) % m; 

	// transfer input data to shared memory
	in[mat_ind] = inputd[i]; 


	// creating eye matrix for gauss jordan elimination
	if (mat_row == mat_col) {	
		out[mat_ind] = 1.0; 
	}
	else {
		out[mat_ind] = 0.0; 
	}

	// Matrix inversion algorithm main body ======================================== 
	// we use Gauss Jordan Algorithm
	// algorithm: part1 - make the input data upper-triangular
	for (int count1 = 0; count1 < m - 1; ++count1) {
		
		// change current row when its pivot is zero
		if (in[count1 * m + count1] == 0) {
			int count2 = count1 + 1; 
			while ((in[count2 * m + count1] == 0) && (count2 < m)) {
				++count2;
			}
			if(mat_row == count1) {
				in[mat_ind] += in[count2 * m + mat_col];
				out[mat_ind] += out[count2 * m + mat_col]; 
			}
			__syncthreads(); 	
		}


		if (mat_row > count1) {
			double mul = in[mat_row * m + count1] / in[count1 * m + count1]; 
			in[mat_ind] -= mul * in[count1 * m + mat_col];
			out[mat_ind] -= mul * out[count1 * m + mat_col];
		}

		// wait till all the data is changed
		__syncthreads(); 
	}


	// algorithm: part2 - make the input data lower-triangular
	for (int count1 = m - 1; count1 > 0; --count1) {
		if (mat_row < count1) {
			double mul = in[mat_row * m + count1] / in[count1 * m + count1]; 
			in[mat_ind] -= mul * in[count1 * m + mat_col];
			out[mat_ind] -= mul * out[count1 * m + mat_col];
		}

		// wait till all the data is changed
		__syncthreads(); 
	}

	// algorithm: part3 - normalize input data to create matrix inversion
	out[mat_ind] /= in[mat_row * m + mat_row]; 
	// ============================================================================


	outputd[i] = out[mat_ind]; 
}
*/






// print matrix
void print_matrix(char* name, double* data, int size, int d_shift) {
	printf("arr : %s\n", name);
	for (int i=0+d_shift; i<size+d_shift; ++i) {
		printf("%d : %f\n", i, data[i]); 
	}
	printf("--------------------\n"); 
}
