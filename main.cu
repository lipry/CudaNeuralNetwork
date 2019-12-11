#include <iostream>
#include <cstring>
#include <fstream>
#include "src/utils/Matrix.h"
#include "cublas_v2.h"
#include "src/utils/common.h"
#include "src/layers/LinearLayer.h"
#include "src/layers/SigmoidLayer.h"
#include "src/cost_functions/BinaryCrossEntropy.h"
#include "src/NeuralNetwork.h"
#include "src/layers/ReluLayer.h"
#include "src/datasets/MNISTParser.h"
#include "src/utils/csv.h"
#include "src/datasets/PulsarParser.h"
#include <chrono>

using namespace std;

float accuracy(const Matrix& predictions, const Matrix& targets);

int main() {
    int batch_size = 1000;
    int n_epochs = 1000;

    std::string logs_dir_path = "/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/logs/";


    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream log_csv;
    log_csv.open(logs_dir_path + "logs_" + std::to_string(batch_size) + "_"+ std::to_string(n_epochs) + ".log", std::ios::app);

    NeuralNetwork nn = NeuralNetwork(0.01f);
    nn.newLayer(new LinearLayer("linear_layer1", 100, 8));
    nn.newLayer(new ReluLayer("relu"));
    nn.newLayer(new LinearLayer("linear_layer2", 1, 100));
    nn.newLayer(new SigmoidLayer("sigmoid_out"));

    nn.setCostFunction(new BinaryCrossEntropy());

    BinaryCrossEntropy cost;

    cout << "BATCH SIZE: " << batch_size << " BLOCK_SIZE: " << BLOCK_SIZE << endl;

    std::string train_path = "/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/data/pulsar_normalized_train.csv";
    PulsarParser pulsar = PulsarParser(batch_size);

    pulsar.Parse(train_path);

    std::vector<Matrix> batches = pulsar.getBatches();
    std::vector<Matrix> labels = pulsar.getLabels();

    Matrix Y;

    float c;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        c = 0.0;

        for (int batch_idx = 0; batch_idx < batches.size() - 1; batch_idx++) {
            if(epoch == 0) {
                batches.at(batch_idx).cpyHostToDev();
                labels.at(batch_idx).cpyHostToDev();
            }

            Y = nn.forward(handle, batches.at(batch_idx));

            nn.backprop(handle, Y, labels.at(batch_idx));
            c += cost.getCost(Y, labels.at(batch_idx));
        }

        if(epoch%100==0) {
            std::cout << "Epoch: " << epoch
                      << ", Cost: " << c / batches.size()
                      << ", Time: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count()
                      << std::endl;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    int time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::string test_path = "/home/studenti/fabio.lipreri/Documents/NeuralNetworkCUDA/data/pulsar_normalized_test.csv";
    PulsarParser pulsar_test = PulsarParser(batch_size);

    pulsar_test.Parse(train_path);

    std::vector<Matrix> batches_test = pulsar_test.getBatches();
    std::vector<Matrix> labels_test = pulsar_test.getLabels();

    float acc = 0.0;
    for (int batch_idx = 0; batch_idx < batches_test.size(); batch_idx++) {
        Y = nn.forward(handle, batches_test.at(batch_idx));
        Y.cpyDevToHost();
        acc += accuracy(Y, labels.at(batch_idx));
    }
    acc = acc / batches_test.size();


    log_csv<< std::to_string(batch_size)+","+std::to_string(BLOCK_SIZE)+","+std::to_string(time)+","+std::to_string(acc)+","+std::to_string(c)+"\n";
    std::cout<< std::to_string(batch_size)+","+std::to_string(BLOCK_SIZE)+","+std::to_string(time)+","+std::to_string(acc)+","+std::to_string(c)+"\n"<<endl;

    log_csv.close();
    cudaDeviceReset();

    return 0;
}

float accuracy(const Matrix& predictions, const Matrix& targets) {
    int m = predictions.getY();
    int correct_predictions = 0;

    for (int i = 0; i < m; i++) {
        float prediction = predictions[i] >= 0.5 ? 1 : 0;
        if (prediction == targets[i]) {
            correct_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / m;
}



