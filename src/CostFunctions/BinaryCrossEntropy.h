//
// Created by Fabio Lipreri on 2019-11-22.
//

#ifndef NEURALNETWORKCUDA_BINARYCROSSENTROPY_H
#define NEURALNETWORKCUDA_BINARYCROSSENTROPY_H


#include "CostFunctions.h"

class BinaryCrossEntropy : public CostFunctions{
public:
    float getCost(Matrix predictions, Matrix labels);
    Matrix getDCost(Matrix predictions, Matrix label, Matrix dY);
};


#endif //NEURALNETWORKCUDA_BINARYCROSSENTROPY_H
