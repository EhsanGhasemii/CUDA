# General APC + CUDA programming

## What are the main parts of this program? 
- src
	- cuda_main.cu(*)
	- general_apc.cpp(modified)
	- gpuFunction.cpp(*)
	- mainwindow.cpp(modified)
	- matchfilter.cpp
- include
	- cuda_main.cu(*)
	- general_apc.h
	- gpuerrors.h(*)
	- gpuFunctions.h(*)
	- gputimer.h(*)
	- mainwindow.h
	- matchfilter.h
	- ui_mainwindow.h
- Plotters
	- jcustomplot.cpp
	- jcustomplot.h
	- qcustomplot.cpp
	- qcustomplot.h
- files
	- mainwindow.ui
- data
	- some datas with .csv format ...
- test3.pro


As the CUDA Programming team, we have added (*) format files and also edited (modified) format files.

## How to run this program?
You need to run the following commands in the main directory of this program, where the above files are located.

```bash
qmake 
make
./output
```

## How to check the timings and accuracy?
The output of your program in command bar will be something as follows: 

```
Reading duration time: 14745 microseconds
batch_size: 1
data_num: 99
y_n_size: 298
X_size: 285
R_row: 13
Ss_size: 25
s_size: 13
Device Name: NVIDIA GeForce GTX 1650
I am in gpuKernel ..
=================================================
TIME OF GPU PARTS:
t12: 1.05341
t23: 96.499
t34: 92.0658
t45: 41.791
t15: 231.41
=================================================
GPUUUU Time: 2024.96 miliseconds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indx: 0
MSE(0): 2.64048e-12
indx: 1
MSE(0): 7.08755e-10
indx: 2
MSE(0): 3.16756e-11
indx: 3
MSE(0): 8.14623e-12
```


We have launched many threads to run multiple batches at the same time. The number of batches is equal to the number of files you plan to run simultaneously. You can reset the number of batches in line 37 of the gpuFunction.cpp file. Also, this program repeats the number of its processes 256 times to show us the most accurate timing. Therefore, the amount of time you see at the end should be divided by this number to understand how much time it takes to process each data. You can see this part on lines 220 and 305 of cuda_main.cu file.  

The amount of time that the GPU spends on its processing is displayed in the following section.  
```
GPUUUU Time: 2024.96 miliseconds
```
You have to divide this number by the 256 I mentioned above. We have included this part of the program in mainwindow.cpp file on line 267. About the accuracy of the program, I must mention the following lines. These errors are related to the difference of each line of data with its corresponding serial calculation. Actually, each file has about 100 rows of data, and we have reported the error of each row separately in these parts.  
```
indx: 3
MSE(0): 8.14623e-12
```


## one important note
Pay attention to test3.pro file. You should modify this file according to your GPU. This file is written for Nvidia Geforce GTX 1650. The name of this file is not important and you can change it according to your project.  
```
###################################################
# CUDA settings -- 
CUDA_DIR = /usr
SYSTEM_TYPE = 64
NVCC_OPTIONS = --use_fast_math
INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcudart
CUDA_SOURCES += src/cuda_main.cu

CUDA_OBJECTS_DIR = ./cuda_objects
system(mkdir -p $$CUDA_OBJECTS_DIR)

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$CUDA_DIR/bin/nvcc -c $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=sm_75 -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda
###################################################
```
You should modify this part: 
```
cuda.commands = $$CUDA_DIR/bin/nvcc -c $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=sm_75 -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
```

## Contact us
Mail: Ehsanghasemi7998@gmail.com
phone: +989904690571


