//
// Created by Fabio Lipreri on 2019-11-22.
//

#include "BinaryCrossEntropy.h"
#include "../utils/cudamath.h"


Matrix BinaryCrossEntropy::getDCost(Matrix predictions, Matrix label, Matrix dY) {
    return Matrix();
}

float BinaryCrossEntropy::getCost(Matrix predictions, Matrix labels) {

    float cost_value = gpu_bce_cost(predictions.getDevData().get(), labels.getDevData().get(), labels.getX());

    return cost_value;
}

