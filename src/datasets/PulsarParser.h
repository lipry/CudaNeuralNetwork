#include <utility>

//
// Created by Fabio Lipreri on 2019-12-02.
//

#ifndef NEURALNETWORKCUDA_PULSARPARSER_H
#define NEURALNETWORKCUDA_PULSARPARSER_H

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

    void Print(){
        for(size_t b_id = 0; b_id < batches.size(); b_id++){
            for (int c = 0; c < batches[b_id].getY(); c++){
                for (int r=0; r < batches[b_id].getX(); r++){
                    cout << batches[b_id][CMIDX(r, c, batches[b_id].getX())] << " ";
                }
                cout << " ---> " << labels[b_id][c] << "\n\n";
            }

        }
    }

    int getBatchSize() const {
        return batchSize;
    }

    int Parse(std::string csvPath)
    {
        io::CSVReader<8+1> in(csvPath);
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
        float items[8];
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

            for(int r=0; r < 8; r++) {
                batches[batches.size()-1][CMIDX(r, batchCounter, 8)] = items[r];
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
};
#endif //NEURALNETWORKCUDA_PULSARPARSER_H
