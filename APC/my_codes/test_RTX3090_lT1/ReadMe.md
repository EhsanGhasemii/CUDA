# README

## How to run the program? 
```bash
g++ src/mainwindow.cpp src/gpuFunctions.cpp -o main -larmadillo -llapack -lblas
nvcc src/mainwindow.cpp src/gpuFunctions.cpp src/cuda_main.cu -o main -larmadillo -llapack -lblas
nvcc src/mainwindow.cpp src/gpuFunctions.cpp src/cuda_main.cu src/general_apc.cpp -o main -larmadillo -llapack -lblas
```

## notice
In this directory we decrease number of threads from 13 * 13 to 13 and increase amount of works per each thread. 
