//
// Created by Fabio Lipreri on 2019-11-23.
//

#include "NeuralNetwork.h"

using namespace std;

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
    dY.allocate_size(predictions.getX(), 1);

    Matrix top_diff = this->cost->getDCost(predictions, labels, dY);

    for(auto it = nn_layers.rbegin(); it != nn_layers.rend(); it++ ){
        top_diff = (*it)->backward(handle, top_diff, this->learningRate);
    }

    cudaDeviceSynchronize();
}