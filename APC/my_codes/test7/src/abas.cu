#include "../include/abas.h"
#include <iostream>

Abas::Abas() {

}

int Abas::fun1() {
	std::cout << "I am abas .." << std::endl;

	struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

	return 1; 
}
