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
						   double* outputd
						   );
__global__ void matrixInversion(double* inputd,
								double* outputd, 
								const int n, 
								const int m
								);		
void print_matrix(char* name, double* data, int size, int d_shift);
// ==========================================================


// main body
void gpuKernel(double* X_real, 
			   double* X_imag, 
			   double* R_real, 
			   double* R_imag, 
			   double* Ss_real, 
			   double* Ss_imag, 
			   double* s_real, 
			   double* s_imag,
			   double* alpha_real,
			   double* output
			   ) {
	
	// print name of device
	struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);


	// define our variabels
	int X_size = 286; 
	int R_row = 13; 
	int R_col = 13; 
	int Ss_size = 25; 
	int s_size = 13; 
	int print_flag = 1; 

	// allocate memory in CPU for calculation

	// define our needed variables in GPU
	double* X_reald;
	double* X_imagd; 
	double* R_reald;
	double* R_imagd; 
	double* Ss_reald; 
	double* Ss_imagd; 
	double* s_reald; 
	double* s_imagd;
	double* alpha_reald;

	double* outputd; 


	// allocation memory in GPU
	HANDLE_ERROR(cudaMalloc((void**)&X_reald, X_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&X_imagd, X_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&R_reald, R_row * R_col * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&R_imagd, R_row * R_col * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&Ss_reald, Ss_size * R_row * R_col * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&Ss_imagd, Ss_size * R_row * R_col * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&s_reald, s_size * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&s_imagd, s_size * sizeof(double))); 
	HANDLE_ERROR(cudaMalloc((void**)&alpha_reald, 1 * sizeof(double))); // hard_code: 1 - ch..

	HANDLE_ERROR(cudaMalloc((void**)&outputd, 262 * 13 * 13 * sizeof(double))); // hard_code: 262 - change it in future

	// copy array from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(X_reald, X_real, X_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(X_imagd, X_imag, X_size * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(R_reald, R_real, R_row * R_col * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(R_imagd, R_imag, R_row * R_col * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Ss_reald, Ss_real, Ss_size * R_row * R_col * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(Ss_imagd, Ss_imag, Ss_size * R_row * R_col * sizeof(double), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(s_reald, s_real, s_size * sizeof(double), cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(cudaMemcpy(s_imagd, s_imag, s_size * sizeof(double), cudaMemcpyHostToDevice)); 
	HANDLE_ERROR(cudaMemcpy(alpha_reald, alpha_real, 1 * sizeof(double), cudaMemcpyHostToDevice)); // hard_code: 1 - ch..

	
	// define our threads and blocks dimension
	dim3 dimGrid = getDimGrid(1);
	dim3 dimBlock = getDimBlock(262);

	// transfer processing in CUDA
	double gpu_kernel_time = 0.0;
	GpuTimer timer;
    timer.Start();
	kernelFunc<<< dimGrid,dimBlock >>>(X_reald, X_imagd, R_reald, R_imagd, Ss_reald, Ss_imagd, s_reald, s_imagd, alpha_reald, outputd);
	timer.Stop();
	gpu_kernel_time = timer.Elapsed();

	// copy result from GPU to CPU memory
	HANDLE_ERROR(cudaMemcpy(output, outputd, 262 * 13 * 13 * sizeof(double), cudaMemcpyDeviceToHost));
	
	
	// matrix inversion =================================================================================
	double* input; 
	double* inversion;
	
	input = (double*)malloc(3 * 3 * sizeof(double)); 
	inversion = (double*)malloc(3 * 3 * sizeof(double));

	input[0] = 0; 
	input[1] = 0; 
	input[2] = 1; 
	input[3] = 1; 
	input[4] = 2; 
	input[5] = 3; 
	input[6] = 2; 
	input[7] = 5; 
	input[8] = 9;
	/*input[9] = 1; 
	input[10] = 1; 
	input[11] = 1; 
	input[12] = 2; 
	input[13] = 4; 
	input[14] = 6; 
	input[15] = 3; 
	input[16] = 6; 
	input[17] = 12; */

	double* inputd; 
	double* inversiond; 

	HANDLE_ERROR(cudaMalloc((void**)&inputd, 3 * 3 * sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&inversiond, 3 * 3 * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(inputd, input, 3 * 3 * sizeof(double), cudaMemcpyHostToDevice));

	dimGrid = getDimGrid(2); 
	dimBlock = getDimBlock(3 * 3); 

	matrixInversion<<< dimGrid, dimBlock >>>(inputd, inversiond, 1, 3); 
	HANDLE_ERROR(cudaMemcpy(inversion, inversiond, 3 * 3 * sizeof(double), cudaMemcpyDeviceToHost));
	// ==================================================================================================


	// print report

	if (print_flag) {
		std::cout << "this is output of GPU: " << std::endl; 
		char name[20] = "a"; 
		//print_matrix(name, a		 , n);

		strcpy(name, "input"); 
		print_matrix(name, input, 3 * 3, 0); 

		strcpy(name, "inversion");
		print_matrix(name, inversion, 3 * 3, 0); 
		
	}


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
	HANDLE_ERROR(cudaFree(outputd));



	// print a report
	std::cout << "I am in gpuKernel .." << std::endl;


}



// define our functions
//void fill(float* data, int size) {
//    for (int i=0; i<size; ++i)
//        data[i] = (float) (rand() % 17 - 8);
//}

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
						   double* outputd
						   ) {
	

	// define our variables
	// !!!!! Please consider that if your input data will be big 
	// you may need to use long format of variables !!!!!!!!!!!!
	__shared__ double rho_reald[262]; // hard_code: 262 - change it in future
	__shared__ double rho_imagd[262]; // hard_code: 262 - change it in future

	__shared__ float Ss_reald_shr[25 * 13 * 13]; 
	__shared__ float Ss_imagd_shr[25 * 13 * 13]; 

	double C_reald_shr[262 * 13 * 13]; 
	double C_imagd_shr[262 * 13 * 13]; 

	// define index of each thread
	long long i;
	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);


	// create rho varibel in shared memory
	if (i < 262) {               // hard_code: 262 - change it in future
		double my_angle = atan2(X_imagd[i], X_reald[i]); 
		double my_radius = sqrt(X_imagd[i] * X_imagd[i] + X_reald[i] * X_reald[i]); 
		my_radius = pow(my_radius, alpha_reald[0]); 
		my_angle *= alpha_reald[0]; 

		rho_reald[i] = my_radius * cos(my_angle); 
		rho_imagd[i] = my_radius * sin(my_angle); 
	}

	// transfer Ss matrices in shared memory
	if (i < 13 * 13) {
		for (int count1 = 0; count1 < 25; ++count1) {
			Ss_reald_shr[count1 * 169 + i] = Ss_reald[count1 * 169 + i]; 
			Ss_imagd_shr[count1 * 169 + i] = Ss_imagd[count1 * 169 + i]; 
		}
	}
	
	// wait till all the data is ready
	__syncthreads(); 

	// first part of the algorithm: 25 * (matrix multilplication and addition)
	if ((i >= 12) && (i <= 250)) {          // hard_code: 12 & 250 - ch..
		for (int count1 = 0; count1 < 25; ++count1) {
			for (int count2 = 0; count2 < 13; ++count2) {
				for (int count3 = 0; count3 < 13; ++count3) {         // hard_code: (i-12) in both lines
					C_reald_shr[(i-12) * 169 + count2 * 13 + count3] += (rho_reald[i - 12 + count1] * Ss_reald_shr[count1 * 169 + count2 * 13 + count3]
															           - rho_imagd[i - 12 + count1] * Ss_imagd_shr[count1 * 169 + count2 * 13 + count3]); 
					C_imagd_shr[(i-12) * 169 + count2 * 13 + count3] += (rho_reald[i - 12 + count1] * Ss_imagd_shr[count1 * 169 + count2 * 13 + count3]
																       + rho_imagd[i - 12 + count1] * Ss_reald_shr[count1 * 169 + count2 * 13 + count3]);
				}
			}
		}
	}

}


__global__ void matrixInversion(double* inputd,		// input data is "inputd"
								double* outputd,	// output data is "outputd"
								int n,				// number of matrices is n
								int m				// size of each matrix is m*m
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






// print matrix
void print_matrix(char* name, double* data, int size, int d_shift) {
	printf("arr : %s\n", name);
	for (int i=0+d_shift; i<size+d_shift; ++i) {
		printf("%f\n", data[i]); 
	}
	printf("--------------------\n"); 
}
