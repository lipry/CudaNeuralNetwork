//
// Created by Fabio Lipreri on 2019-11-20.
//

#ifndef NEURALNETWORKCUDA_SIGMOIDLAYER_H
#define NEURALNETWORKCUDA_SIGMOIDLAYER_H


#include "../utils/Matrix.h"
#include "Layer.h"

class SigmoidLayer : public Layer{
private:
    Matrix Z;
    Matrix Res;
    Matrix dZ;
public:

    SigmoidLayer(std::string name);
    ~SigmoidLayer();

    Matrix &forward(cublasHandle_t handle, Matrix &A);

    Matrix &backward(cublasHandle_t handle, Matrix &top_diff);

    std::string getName() override;


};


#endif //NEURALNETWORKCUDA_SIGMOIDLAYER_H
