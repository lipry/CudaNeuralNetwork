//
// Created by Fabio Lipreri on 2019-11-22.
//

#ifndef NEURALNETWORKCUDA_BINARYCROSSENTROPY_H
#define NEURALNETWORKCUDA_BINARYCROSSENTROPY_H


#include "CostFunction.h"

class BinaryCrossEntropy : public CostFunction{
public:
    float getCost(Matrix predictions, Matrix labels);
    Matrix getDCost(Matrix predictions, Matrix labels, Matrix dY);
};


#endif //NEURALNETWORKCUDA_BINARYCROSSENTROPY_H
