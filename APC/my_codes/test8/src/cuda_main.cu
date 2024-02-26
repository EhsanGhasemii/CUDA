#include "../include/cuda_main.h"
#include "../include/gpuerrors.h"
#include "../include/gputimer.h"
#include <iostream>


// TIlEX and TILEY are used to set the number of threads in c CUDA block
#define TILEX 32
#define TILEY 32

// define our functions
dim3 getDimGrid(const int m, const int n); 
dim3 getDimBlock(const int m, const int n); 
__global__ void kernelFunc(float* ad, float* outputd, const int n);
void print_matrix(char* name, float* data, int size);



void gpuKernel() {
	
	// print name of device
	struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	std::cout << "kir va kos e dalaget.. " << std::endl;


	// define our variabels
	int n = 4; 
	int print_flag = 1; 

	// allocate memory in CPU for calculation
	float* a; 
	float* output; 
	a		= (float*)malloc(n * sizeof(float));
	output	= (float*)malloc(n * sizeof(float));

	// fill input arrays with random values
	srand(0); 
	fill(a, n);

	// copy CPU variables in GPU 
	float* ad; 
	float* outputd; 

	// allocation memory in GPU
	HANDLE_ERROR(cudaMalloc((void**)&ad		, n * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&outputd, n * sizeof(float)));

	// copy array from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));

	
	// define our threads and blocks dimension
	dim3 dimGrid = getDimGrid(1,n);
	dim3 dimBlock = getDimBlock(1,n);

	// transfer processing in CUDA
	double gpu_kernel_time = 0.0;
	GpuTimer timer;
    timer.Start();
	kernelFunc<<< dimGrid,dimBlock >>>(ad, outputd, n);
	timer.Stop();
	gpu_kernel_time = timer.Elapsed();

	// copy result from GPU to CPU memory
	HANDLE_ERROR(cudaMemcpy(output, outputd, n * sizeof(float), cudaMemcpyDeviceToHost));


	// print report

	if (print_flag) {
		char name[20] = "a"; 
		print_matrix(name, a		 , n);

		strcpy(name, "output"); 
		print_matrix(name, output	, 8); 
		
	}


	// remove array in GPU
	HANDLE_ERROR(cudaFree(ad	 ));
	HANDLE_ERROR(cudaFree(outputd));


	// print a report
	std::cout << "I am in gpuKernel .." << std::endl;


}



// define our functions
void fill(float* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = (float) (rand() % 17 - 8);
}

// 
dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(1,1,1);
	//grx = ceil(float(n * n) / gry / grz / blx / bly / blz); 
	//dim3 dimGrid(grx, gry, grz); 
	return dimGrid;
}

//
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY,1);
	//dim3 dimBlock(blx, bly, blz);
	return dimBlock;
}

// kernelFunc
__global__ void kernelFunc(float* ad, float* outputd, const int n) {
	
	long long i; 

	i = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + (blockIdx.x);
	i *= blockDim.z * blockDim.y * blockDim.x;
	i += (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x);

	outputd[i] = 1.0; 
}


// print matrix
void print_matrix(char* name, float* data, int size) {
	printf("arr : %s\n", name);
	for (int i=0; i<size; ++i) {
		printf("%f\n", data[i]); 
	}
	printf("--------------------\n"); 
}
