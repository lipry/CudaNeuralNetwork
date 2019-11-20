//
// Created by Fabio Lipreri on 2019-11-15.
//

#ifndef NEURALNETWORKCUDA_CUDAMATH_H
#define NEURALNETWORKCUDA_CUDAMATH_H

#include "../utils/common.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

__global__ void add_vect(float *R, float *A, float *B, int x, int y);

__global__ void init_randoms(unsigned int seed, curandState_t* states);

__global__ void randoms(curandState_t* states, float* numbers, float lower, float higher);

__global__ void add_vector_row_wise(float *R, float *A, float *V, int x, int y);

void gpu_add_bias(float *A, float *b, float *Y, int x, int y);

void gpu_blas_mmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, int m, int k, int n);

void gpu_blas_mtmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, int k, int m, int n);


#endif //NEURALNETWORKCUDA_CUDAMATH_H
