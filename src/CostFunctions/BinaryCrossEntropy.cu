//
// Created by Fabio Lipreri on 2019-11-22.
//

#include "BinaryCrossEntropy.h"
#include "../utils/cudamath.h"

float BinaryCrossEntropy::getCost(Matrix predictions, Matrix labels) {
    Matrix cost = Matrix(1, 1);
    cost.allocate();
    cost.cpyHostToDev();

    gpu_bce_cost(cost.getDevData().get(), predictions.getDevData().get(), labels.getDevData().get(), labels.getX());

    cost.cpyDevToHost();


    return *cost.getHostData().get();
}

Matrix BinaryCrossEntropy::getDCost(Matrix predictions, Matrix labels, Matrix dY) {

    gpu_derivative_bce_cost(dY.getDevData().get(), predictions.getDevData().get(),
            labels.getDevData().get(), predictions.getX());
    return dY;
}

