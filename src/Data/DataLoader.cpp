#include "../../include/Data/DataLoader.h"

DataLoader::DataLoader(const Dataset& ds, size_t batch_size, bool shuffle)
    : dataset(ds), batch_size(batch_size), shuffle(shuffle), 
      rng(std::random_device{}()) {
    reset();
}

void DataLoader::reset() {
    indices.resize(dataset.rows());
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }
}

DataLoader::Iterator::Iterator(const DataLoader& loader, size_t cursor)
    : loader(loader), cursor(cursor) {}

std::vector<size_t> DataLoader::Iterator::getIndices() const {
    size_t end = std::min(cursor + loader.batch_size, loader.dataset.rows());
    std::vector<size_t> indices;
    for (size_t i = cursor; i < end; i++) {
        indices.push_back(loader.indices[i]);
    }
    return indices;
}

Dataset DataLoader::Iterator::operator*() const {
    size_t end = std::min(cursor + loader.batch_size, loader.dataset.rows());
    std::vector<size_t> batch_indices;
    for (size_t i = cursor; i < end; i++) {
        batch_indices.push_back(loader.indices[i]);
    }
    return loader.dataset.selectRows(batch_indices);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    cursor += loader.batch_size;
    return *this;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return cursor < other.cursor;
}

DataLoader::Iterator DataLoader::begin() {
    return Iterator(*this, 0);
}

DataLoader::Iterator DataLoader::end() {
    return Iterator(*this, dataset.rows());
}
