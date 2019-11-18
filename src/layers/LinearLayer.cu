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

    W.cpyDevToHost();
    cout << "Weights" << endl;
    cout << W << endl;

    b.cpyDevToHost();
    cout << "BIAS" << endl;
    cout << b << endl;

}

LinearLayer::~LinearLayer()
{}

Matrix &LinearLayer::forward(cublasHandle_t handle, Matrix &A) {
    this->A = A;

    Y.allocate_size(W.getX(), A.getY());

    // Y(m,n) = W(m,k) * A(k,n)
    gpu_blas_mmul(handle, this->W.getDevData().get(), this->A.getDevData().get(),
            this->Y.getDevData().get(), this->W.getX(), this->W.getY(), this->A.getY());

    gpu_add_bias(this->Y.getDevData().get(), this->b.getDevData().get(),
            this->Y.getDevData().get(), this->Y.getX(), this->Y.getY());

    /*CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, W.getY(),
                             W.getX(), &alpha, W.getDevData().get(), W.getY(),
                             A.getDevData().get(), 1, &beta, Y.getDevData().get(), 1));

    dim3 TxB(BLOCK_SIZE);
    dim3 num_blocks((Y.getY() * Y.getX() + TxB.x - 1) / TxB.x);
    add_vect<<<num_blocks, TxB>>>(Y.getDevData().get(), Y.getDevData().get(), b.getDevData().get(), Y.getX(), Y.getY());
*/



    return Y;
}

Matrix &LinearLayer::backward(cublasHandle_t handle, Matrix &top_diff) {
    return A;
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
        initZeroes(this->W, 3);
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
