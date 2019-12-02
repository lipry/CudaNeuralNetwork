cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CUDACXX /usr/local/cuda-8.0/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-8.0/bin/nvcc)

project(NeuralNetworkCUDA LANGUAGES C CXX CUDA)

add_executable(NeuralNetworkCUDA main.cu dummy.cpp
        src/utils/Matrix.cu src/utils/Matrix.h
        src/utils/common.h src/layers/Layer.h
        src/layers/LinearLayer.cu
        src/layers/LinearLayer.h
        src/utils/cudamath.cu
        src/utils/cudamath.h
        src/layers/SigmoidLayer.cu
        src/layers/SigmoidLayer.h
        src/cost_functions/CostFunction.h
        src/cost_functions/BinaryCrossEntropy.cu
        src/cost_functions/BinaryCrossEntropy.h
        src/NeuralNetwork.cu src/NeuralNetwork.h
        src/layers/ReluLayer.cu
        src/layers/ReluLayer.h
        src/datasets/MNISTParser.h
        src/datasets/PulsarParser.h
        src/utils/csv.h)


target_compile_features(NeuralNetworkCUDA PUBLIC cxx_std_14)
target_link_libraries(NeuralNetworkCUDA -lcublas -lcurand -lpthread)