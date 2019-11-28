//
// Created by Fabio Lipreri on 2019-11-15.
//

#ifndef NEURALNETWORKCUDA_LINEARLAYER_H
#define NEURALNETWORKCUDA_LINEARLAYER_H


#include "../utils/Matrix.h"
#include "Layer.h"
#include "cublas_v2.h"

class LinearLayer : public Layer {
    //Y=𝑊𝐴+𝑏
private:
    // parameters
    // TODO: incorporare bias nella matrice W
    Matrix W;
    Matrix b;

    Matrix Y;
    Matrix A;

    //for backward
    Matrix dA;

    void initWeights(bool random, float lower = 0.0f, float higher = 0.0f);
    void initBias();

    void initZeroes(Matrix &x, float n);


public:
    LinearLayer(std::string name, size_t x, size_t y);
    ~LinearLayer();

    Matrix& forward(cublasHandle_t handle, Matrix& A);
    Matrix& backward(cublasHandle_t handle, Matrix& top_diff, float learning_rate);

    Matrix& getWeigths();
    Matrix& getBias();

};


#endif //NEURALNETWORKCUDA_LINEARLAYER_H
