//
// Created by Fabio Lipreri on 2019-11-23.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(float lr) : learningRate(lr){}

void NeuralNetwork::newLayer(Layer *l) {
    this->nn_layers.push_back(l);
}

void NeuralNetwork::setCostFunction(CostFunction *cf) {
    this->cost = cf;
}

Matrix NeuralNetwork::forward(cublasHandle_t handle, Matrix X) {
    Matrix tmp = X;
    for(auto it = std::begin(nn_layers); it != std::end(nn_layers); ++it){
        tmp = (*it)->forward(handle, tmp);
    }

    Y = tmp;
    return Y;
}

void NeuralNetwork::backprop(cublasHandle_t handle, Matrix predictions, Matrix labels) {

}