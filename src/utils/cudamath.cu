//
// Created by Fabio Lipreri on 2019-11-15.
//

#include "cudamath.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

// ==============
// =   KERNEL   =
// ==============

using namespace std;

__global__ void add_vect(float *R, float *A, float *B, int x, int y){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < x*y)
        R[idx] = __fadd_rn(A[idx], B[idx]);
}

__global__ void init_randoms(unsigned int seed, curandState_t* states) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    curand_init(seed, index, 0, &states[index]);
}

__global__ void randoms(curandState_t* states, float* numbers, float lower, float higher) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    numbers[index] = lower + (higher - lower) * curand_uniform(&states[index]);
}

__global__ void add_vector_row_wise(float *R, float *A, float *V, int x, int y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = CMIDX(row, col, x);
    if(row < x && col < y) {
        R[index] = A[index] + V[row];
    }
}

// ================
// =   FUNCTION   =
// ================

void gpu_add_bias(float *A, float *b, float *Y, int x, int y){
    dim3 TxB(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks((x*y + TxB.x - 1) / TxB.x, (x*y + TxB.y - 1) / TxB.y);
    add_vector_row_wise<<<num_blocks, TxB>>>(Y, A, b, x, y);
}

// =======================
// =   CUBLAS FUNCTION   =
// =======================

// Multiply the arrays A and B on GPU and save the result in C
// Y(m,n) = W(m,k) * A(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, const int m, const int k, const int n) {
         int lda=m,ldb=k,ldc=m;
         const float alf = 1;
         const float bet = 0;
         const float *alpha = &alf;
         const float *beta = &bet;

         // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, W, lda, A, ldb, beta, Y, ldc);
}