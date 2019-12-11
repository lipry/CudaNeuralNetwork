//
// Created by Fabio Lipreri on 2019-11-22.
//

#include "BinaryCrossEntropy.h"
#include "../utils/cudamath.h"

using namespace std;

float BinaryCrossEntropy::getCost(Matrix predictions, Matrix labels) {
    Matrix cost = Matrix(1, 1);
    cost.allocate();
    cost[0] = 0.0f;
    cost.cpyHostToDev();

    // TODO: rimuovi stampa
    /*std::cout<<"cost"<<std::endl;
    std::cout<<cost<<std::endl;

    std::cout<<"pred"<<std::endl;
    std::cout<<predictions<<std::endl;

    std::cout<<"labels"<<std::endl;
    std::cout<<labels<<std::endl;

    std::cout<<"labels X"<<std::endl;
    std::cout<<labels.getX()<<std::endl;

    std::cout<<"-------------------------------"<<std::endl;*/
    gpu_bce_cost(cost.getDevData().get(), predictions.getDevData().get(), labels.getDevData().get(), labels.getX());
    cost.cpyDevToHost();
    return *cost.getHostData().get();
}

Matrix BinaryCrossEntropy::getDCost(Matrix predictions, Matrix labels, Matrix dY) {

    gpu_derivative_bce_cost(dY.getDevData().get(), predictions.getDevData().get(),
            labels.getDevData().get(), predictions.getX());
    return dY;
}

