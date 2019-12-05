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

        //TODO: rimuovere stampe
        //tmp.cpyDevToHost();
        cout << (*it)->getName() << " FORWARD" << endl;
        //cout << tmp << endl;

        tmp = (*it)->forward(handle, tmp);

        //tmp.cpyDevToHost();
        cout << "OUT FORWARD" << endl;
        //cout << tmp << endl;
    }

    cudaDeviceSynchronize();

    Y = tmp;
    return Y;
}

void NeuralNetwork::backprop(cublasHandle_t handle, Matrix predictions, Matrix labels) {
    dY.allocate_size(predictions.getX(), 1);

    //TODO: rimuovere stampe
    predictions.cpyDevToHost();
    cout << "PREDICTIONS" << endl;
    cout << predictions << endl;

    Matrix top_diff = this->cost->getDCost(predictions, labels, dY);

    //TODO: rimuovere stampe
    top_diff.cpyDevToHost();
    cout << "COST TOP DIFF" << endl;
    cout << top_diff << endl;


    for(auto it = nn_layers.rbegin(); it != nn_layers.rend(); it++){
        cout << (*it)->getName() << " BACKWARD" << endl;
        top_diff = (*it)->backward(handle, top_diff, this->learningRate);
    }

    cudaDeviceSynchronize();
}