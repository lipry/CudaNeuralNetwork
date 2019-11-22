#include <iostream>
#include "src/utils/Matrix.h"
#include "cublas_v2.h"
#include "src/utils/common.h"
#include "src/layers/LinearLayer.h"
#include "src/layers/SigmoidLayer.h"
#include "src/CostFunctions/BinaryCrossEntropy.h"

using namespace std;


int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Y(m,n) = W(m,k) * A(k,n)
    /*int features = 4;
    int n_entries = 2;
    Matrix A = Matrix(features, n_entries);
    A.allocate();
    int count = 1;
    for(int r = 0; r<features; r++){
        for(int c=0; c<n_entries;c++){
            A[CMIDX(r, c, features)] = count;
            count++;
        }
    }
    A.cpyHostToDev();
    cout << "A:" << endl;
    cout << A << endl;
    Matrix top_diff = Matrix(n_entries, features);
    top_diff.allocate();
    count = 0;
    for(int r = 0; r<n_entries; r++){
        for(int c=0; c<features; c++){
            top_diff[CMIDX(r, c, n_entries)] = count;
            count++;
        }
    }

    top_diff.cpyHostToDev();
    cout << "top_diff:" << endl;
    cout << top_diff << endl;

    SigmoidLayer *l = new SigmoidLayer("Sigmoid");
    //Layer *l = new LinearLayer("prova", 3, features); // W

    Matrix Y = l->forward(handle, A);

    Y.cpyDevToHost();
    cout << "Y: " << endl;
    cout<< Y << endl;

    Matrix b = l->backward(handle, top_diff, 1.0f);

    b.cpyDevToHost();
    cout << "sigmoid backward with mul" << endl;
    cout << b << endl;*/

    Matrix y = Matrix(5, 1);
    y.allocate();

    for(int i = 0; i  < y.getX(); i++){
        y[i] = 0.5;
    }

    y[2] = 0.5;
    y[4] = 0.2;


    Matrix labels = Matrix(5, 1);
    labels.allocate();

    for(int i = 0; i  < labels.getX(); i++){
        labels[i] = 0.5;
    }

    labels[2] = 0.1;
    labels[4] = 0.8;

    y.cpyHostToDev();
    labels.cpyHostToDev();

    BinaryCrossEntropy bce = BinaryCrossEntropy();
    float c = bce.getCost(y, labels);

    cout << "cost: " << c << endl;

    cudaDeviceReset();
    return 0;
}


