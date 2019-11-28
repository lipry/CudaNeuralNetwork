//
// Created by Fabio Lipreri on 2019-11-28.
//

#ifndef NEURALNETWORKCUDA_RELULAYER_H
#define NEURALNETWORKCUDA_RELULAYER_H

#include "../utils/Matrix.h"
#include "Layer.h"

class ReluLayer : public Layer {
private:
    Matrix Z;
    Matrix Res;
    Matrix dZ;
public:
    ReluLayer(std::string name);
    ~ReluLayer();

    Matrix &forward(cublasHandle_t handle, Matrix &A);

    Matrix& backward(cublasHandle_t handle, Matrix &top_diff, float learning_rate);

    std::string getName() override;
};


#endif //NEURALNETWORKCUDA_RELULAYER_H




