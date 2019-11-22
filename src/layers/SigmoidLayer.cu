//
// Created by Fabio Lipreri on 2019-11-20.
//

#include "SigmoidLayer.h"
#include "../utils/cudamath.h"

using namespace std;

SigmoidLayer::SigmoidLayer(std::string name) {
    this->name = name;

}

SigmoidLayer::~SigmoidLayer() {}

Matrix &SigmoidLayer::forward(cublasHandle_t handle, Matrix &A) {
    this->Z = A;
    Res.allocate_size(A.getX(), A.getY());
    gpu_sigmoid_forward(this->Z.getDevData().get(), this->Res.getDevData().get(), this->Res.getX(), this->Res.getY());
    return Res;
}

Matrix &SigmoidLayer::backward(cublasHandle_t handle, Matrix &top_diff) {
    Matrix sigmoid_res;
    sigmoid_res.allocate_size(this->Res.getX(), this->Res.getY());

    gpu_sigmoid_backward(this->Res.getDevData().get(), sigmoid_res.getDevData().get(), sigmoid_res.getX(), sigmoid_res.getY());

    sigmoid_res.cpyDevToHost();
    cout << "sigmoid backward" << endl;
    cout << sigmoid_res << endl;

    this->dZ.allocate_size(top_diff.getX(), this->Res.getY());

    gpu_blas_mmul(handle, top_diff.getDevData().get(), CUBLAS_OP_N, sigmoid_res.getDevData().get(), CUBLAS_OP_N,
                    this->dZ.getDevData().get(), top_diff.getX(), sigmoid_res.getY(), top_diff.getY());

    return dZ;
}

std::string SigmoidLayer::getName() {
    return Layer::getName();
}
