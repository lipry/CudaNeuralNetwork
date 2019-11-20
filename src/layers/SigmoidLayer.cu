//
// Created by Fabio Lipreri on 2019-11-20.
//

#include "SigmoidLayer.h"
#include "../utils/cudamath.h"

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
    return dZ;
}

std::string SigmoidLayer::getName() {
    return Layer::getName();
}
