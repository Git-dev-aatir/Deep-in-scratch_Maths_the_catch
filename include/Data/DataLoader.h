#pragma once

#include "./Dataset.h"
#include <vector>
#include <random>

class DataLoader {
private:
    const Dataset& dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    std::mt19937 rng;

    void reset();

public:
    DataLoader(const Dataset& ds, size_t batch_size, bool shuffle = true);

    class Iterator {
    private:
        const DataLoader& loader;
        size_t cursor;
        
    public:
        Iterator(const DataLoader& loader, size_t cursor);
        
        std::vector<size_t> getCurrentIndices() const;

        Dataset operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
    };

    Iterator begin();
    Iterator end();
};
