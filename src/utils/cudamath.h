//
// Created by Fabio Lipreri on 2019-11-15.
//

#ifndef NEURALNETWORKCUDA_CUDAMATH_H
#define NEURALNETWORKCUDA_CUDAMATH_H

#include "../utils/common.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

//__global__ void add_vect(float *R, float *A, float *B, int x, int y);

__global__ void init_randoms(unsigned int seed, curandState_t* states);

__global__ void randoms(curandState_t* states, float* numbers, float lower, float higher);

__global__ void add_vector_row_wise(float *R, float *A, float *V, int x, int y);

__device__ float sigmoid(float x);

__device__ float sigmoid_derivate(float x, float top_diff);

__global__ void sigmoidForward(float* R, float* V, int x, int y);

__global__ void sigmoidBackward(float* dR, float* V, float *top_diff, int x, int y);

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost);

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY, int x);

__global__ void reluForward(float* R, float* V, int x, int y);

__global__ void reluBackward(float* dZ, float* top_diff, float* V, int x, int y);

void gpu_add_bias(float *A, float *b, float *Y, int x, int y);

void gpu_sigmoid_forward(float *Z, float *Res, int x, int y);

void gpu_sigmoid_backward(float *Z, float *Res, int x, int y);

void gpu_relu_backward(float *Z, float* top_diff, float *Res, int x, int y);

void gpu_relu_forward(float *Z, float *Res, int x, int y);

void gpu_bce_cost(float *cost, float *prediction, float *labels, int x);

void gpu_derivative_bce_cost(float *dY, float* predictions, float* target, int x);

void gpu_blas_mmul(cublasHandle_t &handle, const float *W, cublasOperation_t W_op,
                   const float *A, cublasOperation_t A_op, float *Y,
                   int m, int k, int n, float learning_rate = 1.0f, float batch_size = 1.0, float bet = 0.0f);

//void gpu_blas_mmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, int m, int n, int k);

//void gpu_blas_mtmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, int m, int n, int k);

//void gpu_blas_mmtul(cublasHandle_t &handle, const float *W, const float *A, float *Y, int m, int n, int k, float batchSize, float bet);

void gpu_blas_sum_column(cublasHandle_t &handle, const float *W, float *Y, int m, int n, float batch_size, float bet, float learning_rate);

#endif //NEURALNETWORKCUDA_CUDAMATH_H
