//
// Created by Fabio Lipreri on 2019-11-28.
//

#include "ReluLayer.h"
#include "../utils/cudamath.h"

using namespace std;

ReluLayer::ReluLayer(std::string name) {
    this->name = name;
}

ReluLayer::~ReluLayer() {}

Matrix& ReluLayer::forward(cublasHandle_t handle, Matrix &A) {
    this->Z = A;
    Res.allocate_size(A.getX(), A.getY());
    gpu_relu_forward(this->Z.getDevData().get(), this->Res.getDevData().get(), this->Res.getX(), this->Res.getY());

    // TODO: togliere stampa
    Res.cpyDevToHost();
    cout << "Y reLU"<< endl;
    cout << Res << endl;
    return Res;
}

Matrix& ReluLayer::backward(cublasHandle_t handle, Matrix &top_diff, float learning_rate) {
    this->dZ.allocate_size(Z.getX(), Z.getY());

    top_diff.cpyDevToHost();
    cout << "top_diff" << endl;
    cout << top_diff << endl;

    // gpu_relu_backward(float *Z, float* top_diff, float *Res, int x, int y)
    gpu_relu_backward(Z.getDevData().get(), top_diff.getDevData().get(), dZ.getDevData().get(), dZ.getX(), dZ.getY());



    // TODO: togliere stampa
    dZ.cpyDevToHost();
    cout << "dZ relu"<< endl;
    cout << dZ << endl;


    return dZ;
}

std::string ReluLayer::getName() {
    return Layer::getName();
}