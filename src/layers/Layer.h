//
// Created by Fabio Lipreri on 2019-11-15.
//

#ifndef NEURALNETWORKCUDA_LAYER_H
#define NEURALNETWORKCUDA_LAYER_H


#include <string>
#include "../utils/Matrix.h"
#include "cublas_v2.h"

class Layer {
protected:
    std::string name;

public:
    virtual ~Layer() {};

    virtual Matrix& forward(cublasHandle_t handle, Matrix& A) = 0;
    virtual Matrix& backward(cublasHandle_t handle, Matrix& top_diff) = 0;

    virtual std::string getName() {
        return this->name;
    };
};

#endif //NEURALNETWORKCUDA_LAYER_H
