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

using namespace std;

class MNISTDataset final
{
public:
    MNISTDataset(int batch_size)
        : m_count(0),
        m_width(0),
        m_height(0),
        m_imageSize(0),
        batchSize(batch_size)
    {
    }

    MNISTDataset(){}

    ~MNISTDataset(){}

    void Print()
    {
        for(size_t b_id = 0; b_id < batches.size(); b_id++){
            for(size_t img = 0; img < batches[b_id].getY(); img++){
                for (size_t j = 0; j < m_height; ++j) {
                    for (size_t i = 0; i < m_width; ++i) {
                        //CMIDX(r, c, n_righe)
                        printf("%3d ", (uint8_t)batches[b_id][CMIDX(j * m_width + i, img, batches[b_id].getX())]);
                    }
                    printf("\n");
                }
                printf("\n\n\ncat(%u)\n\n", (uint8_t)labels[b_id][img]);
            }
        }
    }

    size_t GetImageWidth() const
    {
        return m_width;
    }

    size_t GetImageHeight() const
    {
        return m_height;
    }

    size_t GetImageCount() const
    {
        return m_count;
    }

    size_t GetImageSize() const
    {
        return m_imageSize;
    }

    const vector<Matrix> &getBatches() const {
        return batches;
    }

    const vector<Matrix> &getLabels() const {
        return labels;
    }

    int Parse(const char* imageFile, const char* labelFile)
    {
        FILE* fimg = nullptr;
        cout << fopen(imageFile, "rb") << endl;
        if ((fimg = fopen(imageFile, "rb")) == nullptr)
        {
            printf("Failed to open %s for reading\n", imageFile);
            return 1;
        }
        
        FILE* flabel = nullptr;
        if ((flabel = fopen(labelFile, "rb")) == nullptr)
        {
            printf("Failed to open %s for reading\n", labelFile);
            return 1;
        }
        std::shared_ptr<void> autofimg(nullptr, [fimg, flabel](void*) {
            if (fimg) fclose(fimg);
            if (flabel) fclose(flabel);
        });

        uint32_t value;

        // Read magic number
        assert(!feof(fimg));
        fread(&value, sizeof(uint32_t), 1, fimg);
        printf("Image Magic        :%0X(%I32u)\n", bswap_32(value), bswap_32(value));
        assert(bswap_32(value) == 0x00000803);

        // Read count
        assert(!feof(fimg));
        fread(&value, sizeof(uint32_t), 1, fimg);
        const uint32_t count = bswap_32(value);
        printf("Image Count        :%0X(%I32u)\n", count, count);
        assert(count > 0);

        // Read rows
        assert(!feof(fimg));
        fread(&value, sizeof(uint32_t), 1, fimg);
        const uint32_t rows = bswap_32(value);
        printf("Image Rows         :%0X(%I32u)\n", rows, rows);
        assert(rows > 0);

        // Read cols
        assert(!feof(fimg));
        fread(&value, sizeof(uint32_t), 1, fimg);
        const uint32_t cols = bswap_32(value);
        printf("Image Columns      :%0X(%I32u)\n", cols, cols);
        assert(cols > 0);

        // Read magic number (label)
        assert(!feof(flabel));
        fread(&value, sizeof(uint32_t), 1, flabel);
        printf("Label Magic        :%0X(%I32u)\n", bswap_32(value), bswap_32(value));
        assert(bswap_32(value) == 0x00000801);

        // Read label count
        assert(!feof(flabel));
        fread(&value, sizeof(uint32_t), 1, flabel);
        printf("Label Count        :%0X(%I32u)\n", bswap_32(value), bswap_32(value));
        // The count of the labels needs to match the count of the image data
        assert(bswap_32(value) == count);

        Initialize(cols, rows, count);

        size_t counter = 0;
        Matrix current_batch = Matrix(m_imageSize, batchSize);
        current_batch.allocate();
        Matrix current_labels = Matrix(batchSize, 1);
        current_labels.allocate();
        size_t batchCounter = 0;

        while (!feof(fimg) && !feof(flabel) && counter < m_count)
        {
            if(batchCounter == batchSize) {
                batches.push_back(current_batch);
                labels.push_back(current_labels);

                current_batch.allocate();
                current_labels.allocate();
                batchCounter = 0;
            }
            for (size_t j = 0; j < m_height; ++j)
            {
                for (size_t i = 0; i < m_width; ++i)
                {
                    uint8_t pixel;
                    fread(&pixel, sizeof(uint8_t), 1, fimg);

                    //CMIDX(r, c, n_righe)
                    current_batch[CMIDX(j * m_width + i, batchCounter, m_imageSize)] = pixel;
                }
            }

            uint8_t cat;
            fread(&cat, sizeof(uint8_t), 1, flabel);
            // assert(cat >= 0 && cat < c_categoryCount);
            current_labels[batchCounter] = cat;

            ++batchCounter;
            ++counter;
        }

        assert(batches.size() == labels.size());

        cout << "Batches: "<< batches.size() << endl;

        return 0;
    }

private:
    void Initialize(const size_t width, const size_t height, const size_t count)
    {
        m_width = width;
        m_height = height;
        m_imageSize = m_width * m_height;
        m_count = count;
    }

    // The total number of images
    size_t m_count;

    // Dimension of the image data
    size_t m_width;
    size_t m_height;
    size_t m_imageSize;

    int batchSize;

    std::vector<Matrix> batches;
    std::vector<Matrix> labels;
};