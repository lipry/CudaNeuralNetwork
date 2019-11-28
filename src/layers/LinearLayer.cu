#include <utility>
#include <bits/stdc++.h>

//
// Created by Fabio Lipreri on 2019-11-15.
//

#include "LinearLayer.h"
#include "../utils/common.h"
#include "../utils/cudamath.h"

using namespace std;

LinearLayer::LinearLayer(std::string name, size_t x, size_t y) :
W(x, y), b(x, 1)
{
    this->name = name;
    W.allocate();
    b.allocate();
    this->initWeights(false, 0.0f, 1.0f);
    this->initBias();

    // TODO: rimuovere stampe
    /*W.cpyDevToHost();
    cout << "Weights" << endl;
    cout << W << endl;

    b.cpyDevToHost();
    cout << "BIAS" << endl;
    cout << b << endl;*/

}

LinearLayer::~LinearLayer()
{}

Matrix &LinearLayer::forward(cublasHandle_t handle, Matrix &A) {
    this->A = A;

    Y.allocate_size(W.getX(), A.getY());

    // TODO: rimuovi stampa
    W.cpyDevToHost();
    cout << "W " << endl;
    cout << W << endl;

    // TODO: rimuovi stampa
    b.cpyDevToHost();
    cout << "b" << endl;
    cout << b << endl;
    // Y(m,n) = W(m,k) * A(k,n)
    //m, n, k
    gpu_blas_mmul(handle, this->W.getDevData().get(), CUBLAS_OP_N, this->A.getDevData().get(),
                  CUBLAS_OP_N, this->Y.getDevData().get(), this->W.getX(), this->A.getY(), this->W.getY());

    gpu_add_bias(this->Y.getDevData().get(), this->b.getDevData().get(),
            this->Y.getDevData().get(), this->Y.getX(), this->Y.getY());

    Y.cpyDevToHost();
    cout << "Y: " << endl;
    cout << Y << endl;


    return Y;
}

Matrix &LinearLayer::backward(cublasHandle_t handle, Matrix &top_diff, float learning_rate) {
    dA.allocate_size(A.getX(), A.getY());

    top_diff.cpyDevToHost();
    cout << "top_diff: " << endl;
    cout << top_diff << endl;
    //Calculate dA
    //m, n, k
    gpu_blas_mmul(handle, this->W.getDevData().get(), CUBLAS_OP_T, top_diff.getDevData().get(), CUBLAS_OP_N,
                  this->dA.getDevData().get(), this->W.getY(), top_diff.getY(), this->W.getX());

    // TODO: rimuovi stampa
    dA.cpyDevToHost();
    cout << "dA" << endl;
    cout << dA << endl;

    cout << "batch size: " << this->A.getY() << endl;
    // Update Weights
    gpu_blas_mmul(handle, top_diff.getDevData().get(), CUBLAS_OP_N, this->A.getDevData().get(), CUBLAS_OP_T,
                   W.getDevData().get(), top_diff.getX(), this->A.getX(), top_diff.getY(), learning_rate, 4, 1.0f);


    // TODO: rimuovi stampa
    W.cpyDevToHost();
    cout << "dW " << endl;
    cout << W << endl;

    //Update bias

    b.cpyDevToHost();
    cout << "b pre calcolo: " << endl;
    cout << b << endl;

    //gpu_blas_sum_column(cublasHandle_t &handle, const float *W, float *Y, const int m, const int n, float learning_rate,
    //        const float batch_size, const float bet)
    gpu_blas_sum_column(handle, top_diff.getDevData().get(),  b.getDevData().get(), top_diff.getX(),
            top_diff.getY(), learning_rate, this->A.getY(), 1.0f);

    // TODO: rimuovi stampa
    b.cpyDevToHost();
    cout << "b" << endl;
    cout << b << endl;

    return dA;
}

void LinearLayer::initWeights(bool random, float lower, float higher) {
    if(random){

        curandState_t* states;
        int N = this->W.getX()*this->W.getY();

        /* allocate space on the GPU for the random states */
        cudaMalloc((void**) &states, N * sizeof(curandState_t));

        dim3 TxB(BLOCK_SIZE);
        dim3 num_blocks((N + TxB.x - 1) / TxB.x);
        init_randoms<<<num_blocks, TxB>>>(time(0), states);
        randoms<<<num_blocks, TxB>>>(states, this->W.getDevData().get(), lower, higher);

    }else{
        initZeroes(this->W, 0.5f);
        this->W.cpyHostToDev();
    }

}

void LinearLayer::initBias() {
    this->initZeroes(this->b, 1);
    //this->b[0] = 1;
    //this->b[1] = 2;
    //this->b[2] = 3;
    this->b.cpyHostToDev();
}

void LinearLayer::initZeroes(Matrix &x, float n) {
    for(int r = 0; r<x.getX(); r++){
        for(int c = 0; c<x.getY(); c++){
            x[r*x.getY()+c] = n;
        }
    }
}


Matrix &LinearLayer::getWeigths() {
    return W;
}

Matrix &LinearLayer::getBias() {
    return b;
}
