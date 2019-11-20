#include <iostream>
#include "src/utils/Matrix.h"
#include "cublas_v2.h"
#include "src/utils/common.h"
#include "src/layers/LinearLayer.h"

using namespace std;

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Y(m,n) = W(m,k) * A(k,n)
    int features = 4;
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
    Matrix top_diff = Matrix(3, n_entries);
    top_diff.allocate();
    count = 0;
    for(int r = 0; r<3; r++){
        for(int c=0; c<n_entries; c++){
            top_diff[CMIDX(r, c, 3)] = count;
            count++;
        }
    }

    top_diff.cpyHostToDev();
    cout << "top_diff:" << endl;
    cout << top_diff << endl;


    Layer *l = new LinearLayer("prova", 3, features); // W

    Matrix Y = l->forward(handle, A);

    Y.cpyDevToHost();
    cout << "Y: " << endl;
    cout<< Y << endl;

    l->backward(handle, top_diff);


    cudaDeviceReset();
    return 0;
}