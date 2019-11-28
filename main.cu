#include <iostream>
#include <cstring>
#include "src/utils/Matrix.h"
#include "cublas_v2.h"
#include "src/utils/common.h"
#include "src/layers/LinearLayer.h"
#include "src/layers/SigmoidLayer.h"
#include "src/cost_functions/BinaryCrossEntropy.h"
#include "src/NeuralNetwork.h"
#include "src/layers/ReluLayer.h"
#include "src/datasets/MNISTParser.h"

using namespace std;

int main() {

    std::string image_file = "/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/data/t10k-images-idx3-ubyte";
    std::string labels_file = "/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/data/t10k-labels-idx1-ubyte";

    MNISTDataset mnist;

    mnist.Parse(image_file.c_str(), labels_file.c_str());

    //cout << mnist.GetImageCount() << endl;
    //mnist.Print();

    //FILE* fimg = nullptr;
    //cout << fopen("/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/data/prova.txt", "r") << endl;
    //printf("Oh dear, something went wrong with read()! %s\n", strerror(errno));

    /*cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int features = 2;
    int n_entries = 4;
    Matrix A = Matrix(features, n_entries);
    A.allocate();
    //int count = 1;
    for(int r = 0; r<features; r++){
        for(int c=0; c<n_entries;c++){
            A[CMIDX(r, c, features)] = 0;
            //count++;
        }
    }
    A[CMIDX(1, 0, features)] = -3;
    A[CMIDX(0, 3, features)] = 1;

    A.cpyHostToDev();

    cout << "A: " << endl;
    cout << A << endl;

    Matrix Y_Labels = Matrix(n_entries, 1);
    Y_Labels.allocate();
    for(int i = 0; i < n_entries; i++){
        Y_Labels[i] = 0.7;
    }

    Y_Labels[1] = 0.3;
    Y_Labels[3] = 0.9;

    Y_Labels.cpyHostToDev();


    NeuralNetwork nn = NeuralNetwork(1.0f);
    nn.newLayer(new LinearLayer("linear_layer1", 3, features));
    nn.newLayer(new ReluLayer("relu1"));
    nn.newLayer(new LinearLayer("linear_layer2", 1, 3));
    nn.newLayer(new SigmoidLayer("sigmoid_out"));

    nn.setCostFunction(new BinaryCrossEntropy());

    Matrix Y;
    for (int e = 0; e < 2; e ++) {
        cout << "EPOCA" << e << endl;
        Y = nn.forward(handle, A);


        Y.cpyDevToHost();
        cout << "Y: " << endl;
        cout << Y << endl;

        nn.backprop(handle, Y, Y_Labels);
    }*/


    // Y(m,n) = W(m,k) * A(k,n)
    /*
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

    /*Matrix y = Matrix(5, 1);
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

    Matrix dY = Matrix(5, 1);
    dY.allocate();

    BinaryCrossEntropy bce = BinaryCrossEntropy();
    float cost = bce.getCost(y, labels);

    //dY.cpyDevToHost();
    cout << "cost: " << cost << endl;*/

    cudaDeviceReset();
    return 0;
}


