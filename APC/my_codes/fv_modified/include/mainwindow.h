#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "general_apc.h"
#include "cuda_main.h"
#include "gpuFunctions.h"

double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size);

cx_mat s;
cx_mat y_noisy; 
// modifying ======================
cx_mat y_noisy2;
cx_mat gpu_apc;
cx_mat gpu_test_apc; 
int my_indx;
int data_num; 
int batch_size; 
mat alpha; 
// ================================

#endif // MAINWINDOW_H
