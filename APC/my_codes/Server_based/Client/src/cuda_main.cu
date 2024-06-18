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

__global__ void X_to_rho(double* X_reald, 
						 double* X_imagd, 
						 float* rho_reald, 
						 float* rho_imagd, 
						 float* alpha_reald, 
						 int alpha_indx
						 );


__global__ void complexMatrixInversion(
							           const int n,				// number of matrices is n
									   const int m,		     	// size of each matrix is m*m

									   float* y_n_reald, 
									   float* y_n_imagd, 
									   float* s_reald, 
									   float* s_imagd, 
									   float* rho_reald, 
									   float* rho_imagd, 
									   double* W_reald, 
									   double* W_imagd, 
									   float* Ss_reald, 
									   float* Ss_imagd, 
									   float* R_reald,
									   float* R_imagd, 
									   
									   int alpha_indx, 
									   int N, 
									   int X_size, 
									   int data_num, 
									   int batch_size
									   );			     		// we suppose input data is squre matrix

// ==========================================================

// main body
void gpuKernel(float* y_n_real,
			   float* y_n_imag,
			   double* X_real, 
			   double* X_imag,
			   float* R_real, 
			   float* R_imag, 
			   float* Ss_real, 
			   float* Ss_imag, 
			   float* s_real, 
			   float* s_imag,
			   float* alpha_real,
			   double* output_real,
			   double* output_imag,

			   int batch_size, 
			   int data_num, 
			   int y_n_size, 
			   int X_size, 
			   int R_row, 
			   int Ss_size, 
			   int s_size, 
			   int alpha_size, 
			   int num_iter
			   ) {


	


	// print name of device
	struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

	// define processing time  
	GpuTimer p15;
	double t15[2000]; 

	// define our variabels
	double my_eps = 0.0; 
	if (batch_size % 2 == 1){
		my_eps = 0.000001;
	}
	int N = 13; 
	int print_flag = 0; 

	// define our needed variables in GPU
	float* y_n_reald; 
	float* y_n_imagd; 
	double* X_reald;
	double* X_imagd; 
	float* R_reald;
	float* R_imagd; 
	float* Ss_reald; 
	float* Ss_imagd; 
	float* s_reald; 
	float* s_imagd;
	float* alpha_reald;

	float* rho_reald; 
	float* rho_imagd; 

	// allocation memory in GPU
	HANDLE_ERROR(cudaMalloc((void**)&y_n_reald, batch_size * data_num * y_n_size * sizeof(float)));	// size: batch_size * data_num * y_n_size	space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&y_n_imagd, batch_size * data_num * y_n_size * sizeof(float)));	// size: batch_size * data_num * y_n_size	space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&X_reald, batch_size * data_num * X_size * sizeof(double)));		// size: batch_size * data_num * X_size		space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&X_imagd, batch_size * data_num * X_size * sizeof(double)));		// size: batch_size * data_num * X_size		space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&R_reald, R_row * R_row * sizeof(float)));							// size: R_row * R_row						space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&R_imagd, R_row * R_row * sizeof(float)));							// size: R_row * r_row						space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&Ss_reald, Ss_size * R_row * R_row * sizeof(float)));				// size: Ss_size * R_row * R_row			space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&Ss_imagd, Ss_size * R_row * R_row * sizeof(float)));				// size: Ss_size * R_row * R_row			space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&s_reald, s_size * sizeof(float)));								// size: s_size								space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&s_imagd, s_size * sizeof(float)));								// size: s_size								space: <1MB
	HANDLE_ERROR(cudaMalloc((void**)&alpha_reald, alpha_size * sizeof(float)));						// size: alpha_size							space: <1MB

	HANDLE_ERROR(cudaMalloc((void**)&rho_reald, batch_size * data_num * X_size * sizeof(float)));						// size: batch_size * data_num * X_size
	HANDLE_ERROR(cudaMalloc((void**)&rho_imagd, batch_size * data_num * X_size * sizeof(float)));						// size: batch_size * data_num * X_size


	// lets check speed of our algoritm
	for (int count0 = 0; count0 < num_iter; ++count0) {

		// calculate processing time 
		p15.Start(); 

		// copy array from CPU to GPU
		HANDLE_ERROR(cudaMemcpy(y_n_reald, y_n_real, batch_size * data_num * y_n_size * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(y_n_imagd, y_n_imag, batch_size * data_num * y_n_size * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(X_reald, X_real, batch_size * data_num * X_size * sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(X_imagd, X_imag, batch_size * data_num * X_size * sizeof(double), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(R_reald, R_real, R_row * R_row * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(R_imagd, R_imag, R_row * R_row * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(Ss_reald, Ss_real, Ss_size * R_row * R_row * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(Ss_imagd, Ss_imag, Ss_size * R_row * R_row * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(s_reald, s_real, s_size * sizeof(float), cudaMemcpyHostToDevice)); 
		HANDLE_ERROR(cudaMemcpy(s_imagd, s_imag, s_size * sizeof(float), cudaMemcpyHostToDevice)); 
		HANDLE_ERROR(cudaMemcpy(alpha_reald, alpha_real, alpha_size * sizeof(float), cudaMemcpyHostToDevice));

		
		// define our threads and blocks dimension
		dim3 dimGrid; 
		dim3 dimBlock; 

		for (int count1 = 0; count1 < alpha_size; ++count1) {

			// create rho from X
			dimGrid = getDimGrid(batch_size * data_num * X_size); 
			dimBlock = getDimBlock(1); 
			X_to_rho<<< dimGrid, dimBlock >>>(X_reald, 
											  X_imagd, 
											  rho_reald, 
											  rho_imagd, 
											  alpha_reald, 
											  count1
											  ); 
					


			// APC algorithm part3
			dimGrid = getDimGrid((ceil((batch_size / 2.0) * data_num) - my_eps) * X_size);                // batch_size should be an even number
			dimBlock = getDimBlock(2 * R_row); 
			complexMatrixInversion<<< dimGrid, dimBlock >>>(//output_reald,
															//output_imagd,  
															1, 
															13,

															y_n_reald,		
															y_n_imagd,		
															s_reald,	
															s_imagd,		
															rho_reald,		// depend on alpha step
															rho_imagd,		// depend on alpha step
															X_reald,		// result
															X_imagd,		// result
															Ss_reald, 
															Ss_imagd, 
															R_reald, 
															R_imagd, 

															count1, 
															N, 
															X_size, 
															(ceil((batch_size / 2.0) * data_num) - my_eps), 
															batch_size
															);
			

		} // alpha iterations

		// copy result from GPU to CPU memory
		HANDLE_ERROR(cudaMemcpy(output_real, X_reald, batch_size * data_num * X_size * sizeof(double), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(output_imag, X_imagd, batch_size * data_num * X_size * sizeof(double), cudaMemcpyDeviceToHost));

		p15.Stop(); 
		t15[count0] = p15.Elapsed(); 



	} // end of speed check 


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

	HANDLE_ERROR(cudaFree(y_n_reald));
	HANDLE_ERROR(cudaFree(y_n_imagd));
	HANDLE_ERROR(cudaFree(rho_reald));
	HANDLE_ERROR(cudaFree(rho_imagd));


	// time analysis
	printf("=================================================\n");
	for (int i = 0; i < num_iter; ++i) { 
		std::cout << t15[i] << " "; 
	}
	std::cout << std::endl; 

	// srot the timing array
    for (int i = 0; i < num_iter - 1; ++i) {
        for (int j = 0; j < num_iter - i - 1; ++j) {
            if (t15[j] > t15[j + 1]) {
                double temp = t15[j];
                t15[j] = t15[j + 1];
                t15[j + 1] = temp;
            }
        }
    }
	
	double time_max = t15[num_iter - 1]; 
	double time_min = t15[0]; 
	double time_med = t15[num_iter / 2]; 

	printf("Per batch: \n"); 
	printf("\tmax time: %g ms\n", time_max); 
	printf("\tmedian time: %g ms\n", time_med); 
	printf("\tmin time: %g ms\n", time_min); 


	printf("Per row: \n"); 
	printf("\tmax time: %g ms\n", time_max / data_num); 
	printf("\tmedian time: %g ms\n", time_med / data_num); 
	printf("\tmin time: %g ms\n", time_min / data_num); 
	printf("=================================================\n");

}



// define our functions
dim3 getDimGrid(const int n) {
	dim3 dimGrid(n, 1, 1);

	return dimGrid;
}

dim3 getDimBlock(const int n) {
	dim3 dimBlock(n, 1, 1);

	return dimBlock;
}

__global__ void X_to_rho(double* X_reald, 
						 double* X_imagd, 
						 float* rho_reald, 
						 float* rho_imagd, 
						 float* alpha_reald, 
						 int alpha_indx
						 ){

	double my_angle = atan2(X_imagd[blockIdx.x], X_reald[blockIdx.x]); 
	double my_radius = sqrt(X_imagd[blockIdx.x] * X_imagd[blockIdx.x] + X_reald[blockIdx.x] * X_reald[blockIdx.x]); 
	my_radius = pow(my_radius, alpha_reald[alpha_indx]); 
	my_angle *= alpha_reald[alpha_indx]; 

	// first version of APC algorithm on MATLAB code
	//rho_reald[blockIdx.x] = my_radius * cos(my_angle);
	//rho_imagd[blockIdx.x] = my_radius * sin(my_angle);

	// second version of APC algorithm on MATLAB code 
	rho_reald[blockIdx.x] = my_radius; 
	rho_imagd[blockIdx.x] = 0.0; 
}

__global__ void complexMatrixInversion(const int n,			    // number of matrices is n
									   const int m,		     	// size of each matrix is m*m

									   float* y_n_reald, 
									   float* y_n_imagd, 
									   float* s_reald, 
									   float* s_imagd, 
									   float* rho_reald, 
									   float* rho_imagd, 
									   double* W_reald, 
									   double* W_imagd, 
									   float* Ss_reald, 
									   float* Ss_imagd, 
									   float* R_reald, 
									   float* R_imagd, 
									   
									   int alpha_indx, 
									   int N, 
									   int X_size, 
									   int data_num, 
									   int batch_size

									   ) {			     		// we suppose input data is squre matrix



	// =========================================================================


	// ==========================================================================
	// define our variables
	// thr_batch must be 2 ... .
	__shared__ double out_real[2 * 13 * 13];
	__shared__ double out_imag[2 * 13 * 13];

	__shared__ double in_real[2 * 13 * 13]; 
	__shared__ double in_imag[2 * 13 * 13]; 

	__shared__ double W_real_shr[2 * 13]; 
	__shared__ double W_imag_shr[2 * 13];
 


	// define index of each thread
	long long i;
	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);


	// data and thread location
	int mat_num = i / (m);					// 0, 1, ... data_num * blablabla
	int mat_ind = i % (m);					// 0, 1, ... 13
	int mat_col = i % m;					// 0, 1, ... 13



//	int thr_row = threadIdx.x / 13; 
	int thr_col = threadIdx.x % 13;
	int thr_batch = threadIdx.x / 13; 

	double out_real_temp; 
	double out_imag_temp;

	double W_real_temp;
	double W_imag_temp; 


		// first part of the algorithm: 25 * (matrix multilplication and addition)
		for (int thr_row = 0; thr_row < 13; ++thr_row) {
			in_real[thr_batch * 169 + thr_row * 13 + thr_col] = 0.0; 
			in_imag[thr_batch * 169 + thr_row * 13 + thr_col] = 0.0;
		}
		
		for (int count1 = 0; count1 < 25; ++count1) {         // 25
			for (int thr_row = 0; thr_row < 13; ++thr_row) {
				in_real[thr_batch*169+thr_row*13+thr_col] += (rho_reald[thr_batch*data_num*X_size+blockIdx.x+count1]*Ss_reald[count1*169+thr_row*13+thr_col]
														    - rho_imagd[thr_batch*data_num*X_size+blockIdx.x+count1]*Ss_imagd[count1*169+thr_row*13+thr_col]); 
				in_imag[thr_batch*169+thr_row*13+thr_col] += (rho_reald[thr_batch*data_num*X_size+blockIdx.x+count1]*Ss_imagd[count1*169+thr_row*13+thr_col]
															+ rho_imagd[thr_batch*data_num*X_size+blockIdx.x+count1]*Ss_reald[count1*169+thr_row*13+thr_col]);
			}
		}
	
		// second part of the algorithm: C += R
		for (int thr_row = 0; thr_row < 13; ++thr_row) {
			in_real[thr_batch * 169 + thr_row * 13 + thr_col] += R_reald[thr_row * 13 + thr_col]; 
			in_imag[thr_batch * 169 + thr_row * 13 + thr_col] += R_imagd[thr_row * 13 + thr_col];
		}

	// transfer input data to shared memory
	for (int mat_row = 0; mat_row < 13; ++mat_row) {

		// creating eye matrix for gauss jordan elimination
		if (mat_row == mat_col) {	
			out_real[thr_batch * 169 + mat_row * 13 + thr_col] = 1.0; 
			out_imag[thr_batch * 169 + mat_row * 13 + thr_col] = 0.0; 
		}
		else {
			out_real[thr_batch * 169 + mat_row * 13 + thr_col] = 0.0; 
			out_imag[thr_batch * 169 + mat_row * 13 + thr_col] = 0.0; 
		}
	}




		// Matrix inversion algorithm main body ======================================== 
		// we use Gauss Jordan Algorithm
		for (int count1 = 0; count1 < m - 1; ++count1) {
		
			// change current row when its pivot is zero
			if ((in_real[thr_batch * 169 + count1 * m + count1] == 0) && (in_imag[thr_batch * 169 + count1 * m + count1] == 0)) {
				int count2 = count1 + 1; 
				while ((in_real[thr_batch * 169 + count2 * m + count1] == 0) && (in_imag[thr_batch * 169 + count2 * m + count1] == 0) && (count2 < m)) {
					++count2;
				}	
				in_real[thr_batch * 169 + count1 * 13 + mat_col] += in_real[thr_batch * 169 + count2 * m + mat_col]; // ch ..
				in_imag[thr_batch * 169 + count1 * 13 + mat_col] += in_imag[thr_batch * 169 + count2 * m + mat_col]; // ch ..

				out_real[thr_batch * 169 + count1 * 13 + mat_col] += out_real[thr_batch * 169 + count2 * m + mat_col]; 
				out_imag[thr_batch * 169 + count1 * 13 + mat_col] += out_imag[thr_batch * 169 + count2 * m + mat_col]; 
			}

			// algorithm: part1 - make the input data upper-triangular
			for (int mat_row = count1 + 1; mat_row < 13; ++mat_row) {
				double mul_real = in_real[thr_batch * 169 + mat_row* m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + mat_row* m + count1] *in_imag[thr_batch * 169 + count1 * m + count1];
					 mul_real /= (in_real[thr_batch * 169 + count1 * m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + count1 * m + count1] *in_imag[thr_batch * 169 + count1 * m + count1]);

				double mul_imag = in_imag[thr_batch * 169 + mat_row* m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								- in_real[thr_batch * 169 + mat_row* m + count1] *in_imag[thr_batch * 169 + count1 * m + count1];
					 mul_imag /= (in_real[thr_batch * 169 + count1 * m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + count1 * m + count1] *in_imag[thr_batch * 169 + count1 * m + count1]); 

				in_real[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * in_real[thr_batch * 169 + count1 * m + mat_col]
																	- mul_imag * in_imag[thr_batch * 169 + count1 * m + mat_col]); 
				in_imag[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * in_imag[thr_batch * 169 + count1 * m + mat_col]
																	+ mul_imag * in_real[thr_batch * 169 + count1 * m + mat_col]);

				out_real[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * out_real[thr_batch * 169 + count1 * m + mat_col]
																	 - mul_imag * out_imag[thr_batch * 169 + count1 * m + mat_col]);
				out_imag[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * out_imag[thr_batch * 169 + count1 * m + mat_col]
																	 + mul_imag * out_real[thr_batch * 169 + count1 * m + mat_col]);

			}
		}

		// algorithm: part2 - make the input data lower-triangular
		for (int count1 = m - 1; count1 > 0; --count1) {
			for (int mat_row = count1 - 1; mat_row >= 0; --mat_row) {
				double mul_real = in_real[thr_batch * 169 + mat_row* m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + mat_row* m + count1] *in_imag[thr_batch * 169 + count1 * m + count1];
					 mul_real /= (in_real[thr_batch * 169 + count1 * m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + count1 * m + count1] *in_imag[thr_batch * 169 + count1 * m + count1]); 

				double mul_imag = in_imag[thr_batch * 169 + mat_row* m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								- in_real[thr_batch * 169 + mat_row* m + count1] *in_imag[thr_batch * 169 + count1 * m + count1];
					 mul_imag /= (in_real[thr_batch * 169 + count1 * m + count1] *in_real[thr_batch * 169 + count1 * m + count1]
								+ in_imag[thr_batch * 169 + count1 * m + count1] *in_imag[thr_batch * 169 + count1 * m + count1]); 

				in_real[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * in_real[thr_batch * 169 + count1 * m + mat_col]
																	- mul_imag * in_imag[thr_batch * 169 + count1 * m + mat_col]); 
				in_imag[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * in_imag[thr_batch * 169 + count1 * m + mat_col]
																	+ mul_imag * in_real[thr_batch * 169 + count1 * m + mat_col]);

				out_real[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * out_real[thr_batch * 169 + count1 * m + mat_col]
																	 - mul_imag * out_imag[thr_batch * 169 + count1 * m + mat_col]);
				out_imag[thr_batch * 169 + mat_row * 13 + mat_col] -= (mul_real * out_imag[thr_batch * 169 + count1 * m + mat_col]
																	 + mul_imag * out_real[thr_batch * 169 + count1 * m + mat_col]);
			}
		}



		// algorithm: part3 - normalize input data to create matrix inversion
		for (int mat_row = 0; mat_row < 13; ++mat_row) { 
		out_real_temp = out_real[thr_batch * 169 + mat_row * 13 + mat_col]; 
		out_imag_temp = out_imag[thr_batch * 169 + mat_row * 13 + mat_col]; 
		out_real[thr_batch * 169 + mat_row * 13 + mat_col] = (out_real_temp * in_real[thr_batch * 169 + mat_row * m + mat_row]
															+ out_imag_temp * in_imag[thr_batch * 169 + mat_row * m + mat_row])
						/ (in_real[thr_batch * 169 + mat_row * m + mat_row] * in_real[thr_batch * 169 + mat_row * m + mat_row]
						 + in_imag[thr_batch * 169 + mat_row * m + mat_row] * in_imag[thr_batch * 169 + mat_row * m + mat_row]);
		
		out_imag[thr_batch * 169 + mat_row * 13 + mat_col] = (out_imag_temp * in_real[thr_batch * 169 + mat_row * m + mat_row]
															- out_real_temp * in_imag[thr_batch * 169 + mat_row * m + mat_row])
						/ (in_real[thr_batch * 169 + mat_row * m + mat_row] * in_real[thr_batch * 169 + mat_row * m + mat_row]
						 + in_imag[thr_batch * 169 + mat_row * m + mat_row] * in_imag[thr_batch * 169 + mat_row * m + mat_row]);
		}

	// ============================================================================



	// ============================================================================
	// initialize shared memroy to zero

		W_real_shr[thr_batch * 13 + mat_col] = 0.0; 
		W_imag_shr[thr_batch * 13 + mat_col] = 0.0; 

	// APC algorithm pqrt4: inv(C+R) * s 
		for (int count1 = 0; count1 < 13; ++count1) {     // count1 < 13
			W_real_shr[thr_batch * 13 + mat_col] += out_real[thr_batch * 169 + mat_col * 13 + count1] * s_reald[count1]; 
			W_real_shr[thr_batch * 13 + mat_col] -= out_imag[thr_batch * 169 + mat_col * 13 + count1] * s_imagd[count1]; 

			W_imag_shr[thr_batch * 13 + mat_col] += out_real[thr_batch * 169 + mat_col * 13 + count1] * s_imagd[count1]; 
			W_imag_shr[thr_batch * 13 + mat_col] += out_imag[thr_batch * 169 + mat_col * 13 + count1] * s_reald[count1]; 
		}
			
		W_real_temp = W_real_shr[thr_batch * 13 + mat_col]; 
		W_imag_temp = W_imag_shr[thr_batch * 13 + mat_col]; 
		W_real_shr[thr_batch * 13 + mat_col]= W_real_temp * rho_reald[thr_batch*data_num*X_size+blockIdx.x+12]
											- W_imag_temp * rho_imagd[thr_batch*data_num*X_size+blockIdx.x+12];
		W_imag_shr[thr_batch * 13 + mat_col]= W_real_temp * rho_imagd[thr_batch*data_num*X_size+blockIdx.x+12]
											+ W_imag_temp * rho_reald[thr_batch*data_num*X_size+blockIdx.x+12];

	W_real_temp = 0.0; 
	W_imag_temp = 0.0; 

	// APC algorithm part6: W.t() * y_noisy
	if (threadIdx.x % 13 == 0) {	
		for (int count1 = 0; count1 < 13; ++count1) {       
			W_real_temp += W_real_shr[thr_batch * 13 + count1]*y_n_reald[thr_batch*data_num*X_size+blockIdx.x+12+12*alpha_indx+count1]
						 + W_imag_shr[thr_batch * 13 + count1]*y_n_imagd[thr_batch*data_num*X_size+blockIdx.x+12+12*alpha_indx+count1];		
			W_imag_temp += W_real_shr[thr_batch * 13 + count1]*y_n_imagd[thr_batch*data_num*X_size+blockIdx.x+12+12*alpha_indx+count1]
						 + W_imag_shr[thr_batch * 13 + count1]*y_n_reald[thr_batch*data_num*X_size+blockIdx.x+12+12*alpha_indx+count1];	
		}
		W_imagd[thr_batch * data_num * X_size + blockIdx.x] = W_imag_temp; 
		W_reald[thr_batch * data_num * X_size + blockIdx.x] = W_real_temp; 
	}
}


