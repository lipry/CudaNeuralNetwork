//
// Created by Fabio Lipreri on 2019-11-14.
//

#ifndef NEURALNETWORKCUDA_MATRIX_H
#define NEURALNETWORKCUDA_MATRIX_H

#include <stdlib.h>
#include <iostream>
#include <memory>

class Matrix {
private:
    size_t x;
    size_t y;

    bool host_alloc;
    bool dev_alloc;

    void allocHostMemory();
    void allocDevMemory();
public:
    Matrix(size_t x, size_t y);
    Matrix() = default;

    std::shared_ptr<float> host_data;
    std::shared_ptr<float> dev_data;

    void allocate();
    void allocate_size(size_t x, size_t y);

    void cpyHostToDev();
    void cpyDevToHost();

    const std::shared_ptr<float> &getHostData() const;
    const std::shared_ptr<float> &getDevData() const;

    size_t getX() const;
    size_t getY() const;

    float& operator[](const int index);
    const float& operator[](const int index) const;
};

std::ostream& operator<<(std::ostream &strm, const Matrix &m);


#endif //NEURALNETWORKCUDA_MATRIX_H
