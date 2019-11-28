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

    // TODO: togliere stampa
    Res.cpyDevToHost(); 
    cout << "Y signmoid"<< endl;
    cout << Res << endl;
    return Res;
}

Matrix &SigmoidLayer::backward(cublasHandle_t handle, Matrix &top_diff, float learning_rate) {
    Matrix sigmoid_res;
    sigmoid_res.allocate_size(this->Res.getX(), this->Res.getY());

    top_diff.cpyDevToHost();
    cout << "top_diff" << endl;
    cout << top_diff << endl;

    gpu_sigmoid_backward(this->Res.getDevData().get(), sigmoid_res.getDevData().get(), sigmoid_res.getX(), sigmoid_res.getY());

    this->dZ.allocate_size(top_diff.getX(), this->Res.getY());

    gpu_blas_mmul(handle, top_diff.getDevData().get(), CUBLAS_OP_N, sigmoid_res.getDevData().get(), CUBLAS_OP_N,
                    this->dZ.getDevData().get(), top_diff.getX(), sigmoid_res.getY(), top_diff.getY(), learning_rate);

    return dZ;
}

std::string SigmoidLayer::getName() {
    return Layer::getName();
}
