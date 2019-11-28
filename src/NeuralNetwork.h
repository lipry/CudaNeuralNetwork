//
// Created by Fabio Lipreri on 2019-11-23.
//

#ifndef NEURALNETWORKCUDA_NEURALNETWORK_H
#define NEURALNETWORKCUDA_NEURALNETWORK_H

#include <vector>
#include "layers/Layer.h"
#include "cost_functions/CostFunction.h"


class NeuralNetwork {
private:
    std::vector<Layer*> nn_layers;
    CostFunction* cost;

    Matrix Y;
    Matrix dY;

    float learningRate;
public:
    NeuralNetwork(float lr);

    Matrix forward(cublasHandle_t handle, Matrix X);
    void backprop(cublasHandle_t handle, Matrix predictions, Matrix labels);

    void newLayer(Layer *l);
    void setCostFunction(CostFunction *cf);
};


#endif //NEURALNETWORKCUDA_NEURALNETWORK_H
