//
// Created by Fabio Lipreri on 2019-11-14.
//
#include <iostream>
#include "Matrix.h"
#include "common.h"

using namespace std;

Matrix::Matrix(size_t x, size_t y) : x(x), y(y), host_alloc(false), dev_alloc(false),
                                     dev_data(nullptr), host_data(nullptr)
{}

void Matrix::allocHostMemory() {
    if(!host_alloc){
        host_data = std::shared_ptr<float>(new float[x*y], [&](float *p) {delete[] p; });
        if(host_data)
            host_alloc = true;
    }
}

void Matrix::allocDevMemory() {
    if(!dev_alloc){
        float* tmp = nullptr;
        CHECK(cudaMalloc((void **)&tmp, x*y*sizeof(float)));
        dev_data = std::shared_ptr<float>(tmp, [&](float *p){ cudaFree(p); });
        if(dev_data)
            dev_alloc = true;
    }
}

void Matrix::allocate_size(size_t x, size_t y) {
    if(!dev_alloc && !host_alloc) {
        this->x = x;
        this->y = y;
        allocHostMemory();
        allocDevMemory();
    }
}

void Matrix::allocate() {
    if(!dev_alloc && !host_alloc) {
        allocHostMemory();
        allocDevMemory();
    }
}

void Matrix::cpyHostToDev() {
    if(dev_alloc && host_alloc) {
        CHECK(cudaMemcpy(dev_data.get(), host_data.get(), x * y * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Matrix::cpyDevToHost() {
    if(dev_alloc && host_alloc)
        CHECK(cudaMemcpy(host_data.get(), dev_data.get(), x*y*sizeof(float), cudaMemcpyDeviceToHost));
}

size_t Matrix::getX() const {
    return x;
}

size_t Matrix::getY() const {
    return y;
}

const std::shared_ptr<float> &Matrix::getHostData() const {
    return host_data;
}

const std::shared_ptr<float> &Matrix::getDevData() const {
    return dev_data;
}

float& Matrix::operator[](const int index) {
    return host_data.get()[index];
}

const float& Matrix::operator[](const int index) const {
    return host_data.get()[index];
}

std::ostream& operator<<(std::ostream &strm, const Matrix &m) {
    // TODO: column-major
    strm << "x = " << m.getX() << ", y = " << m.getY() << endl;
    for(int i = 0; i < m.getX(); i++){
        for(int j = 0; j < m.getY(); j++)
            strm << to_string(m.getHostData().get()[CMIDX(i, j, m.getX())]) << " ";
        strm << endl;
    }
    return  strm << endl;
}
