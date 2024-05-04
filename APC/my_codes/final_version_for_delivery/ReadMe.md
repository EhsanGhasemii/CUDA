# General APC + CUDA programming

## What are the main parts of this program? 
- src
	- cuda_main.cu(*)
	- general_apc.cpp(modified)
	- gpuFunction.cpp(*)
	- mainwindow.cpp(modified)
	- mainwindow2.cpp(*)
	- input.cpp
	- matchfilter.cpp
- include
	- cuda_main.cu(*)
	- general_apc.h
	- gpuerrors.h(*)
	- gpuFunctions.h(*)
	- gputimer.h(*)
	- mainwindow.h
	- mainwindow2.h(*)
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
- config.pro


As the CUDA Programming team, we have added (*) format files and also edited (modified) format files.

## How to run this program?
You need to run the following commands in the main directory of this program, where the above files are located.

```bash
qmake 
make
./output
```
We have also added a debug section to check the performance of the algorithm for more than one batch at the same time. Since in the original code you had to select only one of the data for processing, it was not possible to process several batches at the same time. Therefore, we have provided another mode where the execution of the algorithm can be seen on several batches.
```bash
qmake 
make debug
./output2
```

## How to check the timings and accuracy?
The output of your program in command bar will be something as follows: 

```
Reading duration time: 14722 microseconds
num_iter: 1
batch_size: 1
data_num: 99
y_n_size: 298
X_size: 285
R_row: 13
Ss_size: 25
s_size: 13
Device Name: NVIDIA GeForce GTX 1650
=================================================
Processing time in GPU: 163.381
=================================================
GPUUUU Time: 1930.56 miliseconds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indx: 0
MSE(0): 2.06288e-09
-------------
indx: 1
MSE(0): 2.65317e-08
-------------
indx: 2
MSE(0): 8.30422e-10
-------------
indx: 3
MSE(0): 8.32829e-09
-------------
indx: 4
MSE(0): 1.55028e-11
```
We have launched many threads to run multiple batches at the same time. The number of batches is equal to the number of files you plan to run simultaneously. You can change the number of batches you want to process simultaneously in the 28th line of 'mainwindow2.cpp' file. By default, we read the data that was in the corresponding folder and considered each one as a batch. Therefore, by changing the number of batches, you can process a different number of these files in debug mode. Also, in order to have a more accurate idea of the amount of processing time per batch, we have considered a variable called 'num_iter', which by setting it to a high value, for example, around 1024, you can run this algorithm every 1024 times. Then, by dividing the amount of processing time by this number, you can have a much more accurate idea of the amount of processing time per batch. Our latest results were 24ms per batch on an RTX 3090 GPU.


The amount of time that the GPU spends on its processing is displayed in the following section.  
```
GPUUUU Time: 2024.96 miliseconds
```
I must also say that the displayed output is equivalent to the difference between the results of two algorithms implemented serially and in parallel. We have displayed the output as MSE for each of the hundred rows of each batch. If the number of batches is more than one, we have displayed the MSE for each index according to the number of batches. You can see an example of it below.

```
indx: 3
MSE(0): 8.14623e-12
```


You can consider any value you want for the number of batches, but it is recommended that the number of batches be even for optimal use of resources, and even if possible, choose powers of two for the number of batches. The best numbers for batches are 8, 16 or 32. Keep in mind that by increasing the number of batches, the amount of memory consumed by your GPU will increase, so you must be careful to choose the number of batches in such a way that the amount of data can fit in your GPU memory.



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


