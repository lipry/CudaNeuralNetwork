#include <utility>

//
// Created by Fabio Lipreri on 2019-12-02.
//

#ifndef NEURALNETWORKCUDA_PULSARPARSER_H
#define NEURALNETWORKCUDA_PULSARPARSER_H
/*
    Copyright 2014 Henry Tan

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http ://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>
#include <byteswap.h>
#include <iostream>
#include <vector>
#include "../utils/Matrix.h"
#include "../utils/common.h"
#include "../utils/csv.h"

using namespace std;

class PulsarParser final
{
public:
    PulsarParser(int batch_size) : batchSize(batch_size){}

    PulsarParser(){
        batchSize = 100;
    }

    ~PulsarParser(){}

    void Print(){}

    int getBatchSize() const {
        return batchSize;
    }

    int Parse(std::string csvPath)
    {
        io::CSVReader<FEATURES+1> in(csvPath);
        in.read_header(io::ignore_no_column,
                "Mean of the integrated profile",
                "Standard deviation of the integrated profile",
                "Excess kurtosis of the integrated profile",
                "Skewness of the integrated profile",
                "Mean of the DM-SNR curve",
                "Standard deviation of the DM-SNR curve",
                "Excess kurtosis of the DM-SNR curve",
                "Skewness of the DM-SNR curve",
                "target_class");
        float items[FEATURES];
        float target_class;

        size_t batchCounter = 0;

        batches.emplace_back(8, batchSize);
        labels.emplace_back(batchSize, 1);

        batches[batches.size()-1].allocate();
        labels[labels.size()-1].allocate();

        while(in.read_row(items[0], items[1], items[2], items[3], items[4],
                          items[5], items[6], items[7], target_class)){
            if(batchCounter == batchSize) {
                batches.emplace_back(8, batchSize);
                labels.emplace_back(batchSize, 1);

                batches[batches.size()-1].allocate();
                labels[labels.size()-1].allocate();

                batchCounter = 0;
            }

            for(int r=0; r < FEATURES; r++) {
                batches[batches.size()-1][CMIDX(r, batchCounter, FEATURES)] = items[r];
            }

            labels[labels.size()-1][batchCounter] = target_class;

            ++batchCounter;
        }

        assert(batches.size() == labels.size());
        
        //cout << batches[1] << endl;
        //cout << labels[1] << endl;
        return 0;
    }

    const vector<Matrix> &getBatches() const {
        return batches;
    }

    const vector<Matrix> &getLabels() const {
        return labels;
    }

private:
    int batchSize;

    std::vector<Matrix> batches;
    std::vector<Matrix> labels;

    static const int FEATURES = 8;
};
#endif //NEURALNETWORKCUDA_PULSARPARSER_H
