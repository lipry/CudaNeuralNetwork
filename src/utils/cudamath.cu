//
// Created by Fabio Lipreri on 2019-11-15.
//

#include "cudamath.h"
#include <stdio.h>
#include "cublas_v2.h"
#include "Matrix.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdexcept>

// ===============
// =   KERNELS   =
// ===============

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

// ========================
// =   KERNEL FUNCTIONS   =
// ========================

void gpu_add_bias(float *A, float *b, float *Y, int x, int y){
    dim3 TxB(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks((x*y + TxB.x - 1) / TxB.x, (x*y + TxB.y - 1) / TxB.y);
    add_vector_row_wise<<<num_blocks, TxB>>>(Y, A, b, x, y);
}

// ========================
// =   CUBLAS FUNCTIONS   =
// ========================

// Multiply the arrays A and B on GPU and save the result in C
// Y(m,n) = W(m,k) * A(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *W, cublasOperation_t W_op,
        const float *A, cublasOperation_t A_op, float *Y,
        const int m, const int n, const int k, const float batch_size, const float bet) {
    int lda = 0,ldb = 0,ldc = m;
    const float alf = 1.0f / batch_size;
    const float *alpha = &alf;
    const float *beta = &bet;
    if(W_op == CUBLAS_OP_N && A_op == CUBLAS_OP_N) {
        lda = m;
        ldb = k;
    }else if (W_op == CUBLAS_OP_T && A_op == CUBLAS_OP_N){
        lda = k;
        ldb = k;
    }else if (W_op == CUBLAS_OP_N && A_op == CUBLAS_OP_T){
        lda = m;
        ldb = n;
    }else{
        throw std::invalid_argument("Operations not supported in gpu_blas_mmul");
    }

    cublasSgemm(handle, W_op, A_op, m, n, k, alpha, W, lda, A, ldb, beta, Y, ldc);
}

// TODO: implementare eventualemente con parallel reduction
void gpu_blas_sum_column(cublasHandle_t &handle, const float *W, float *Y, const int m, const int n,
        const float batch_size, const float bet){
    int lda = m;
    const float alf = 1.0f / batch_size;
    const float *alpha = &alf;
    const float *beta = &bet;

    //building a dummy 1s vector x
    Matrix x = Matrix(m, 1);
    x.allocate();
    for (int i = 0; i<m; i++)
        x[i] = 1.0f;
    x.cpyHostToDev();

    // Y = W * x
    cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha, W, lda, x.getDevData().get(), 1, beta, Y, 1);
}

/*void gpu_blas_mtmul(cublasHandle_t &handle, const float *W, const float *A, float *Y, const int m,
                    const int n, const int k){
    int lda=k,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, W, lda, A, ldb, beta, Y, ldc);
}

void gpu_blas_mmtul(cublasHandle_t &handle, const float *W, const float *A, float *Y, const int m, const int n,
                    const int k, const float batch_size, const float bet){
    int lda=m,ldb=n,ldc=m;
    const float *alpha = &batch_size;
    const float *beta = &bet;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, W, lda, A, ldb, beta, Y, ldc);
}*/


