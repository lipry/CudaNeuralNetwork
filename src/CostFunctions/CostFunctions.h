//
// Created by Fabio Lipreri on 2019-11-22.
//

#ifndef NEURALNETWORKCUDA_COSTFUNCTIONS_H
#define NEURALNETWORKCUDA_COSTFUNCTIONS_H

#include <string>
#include "../utils/Matrix.h"

class CostFunctions {
protected:
    std::string name;
public:
    virtual float getCost(Matrix predictions, Matrix target) = 0;
    virtual Matrix getDCost(Matrix predictions, Matrix target, Matrix dY) = 0;


    const std::string getName() const {
        return name;
    }

};

#endif //NEURALNETWORKCUDA_COSTFUNCTIONS_H
