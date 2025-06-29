#pragma once

#include "./Dataset.h"
#include <vector>
#include <random>

/**
 * @class DataLoader
 * @brief Iterates over a Dataset in batches for efficient training
 * 
 * Provides an iterator interface to access mini-batches of data. Supports:
 * - Configurable batch size
 * - Random shuffling between epochs
 * - Efficient row indexing without data copying
 */
class DataLoader {
private:
    const Dataset& dataset;        ///< Reference to source dataset
    size_t batch_size;             ///< Number of samples per batch
    bool shuffle;                  ///< Whether to shuffle indices each epoch
    std::vector<size_t> indices;   ///< Current epoch's row indices
    std::mt19937 rng;              ///< Mersenne Twister random engine

    /**
     * @brief Reset the data loader for a new epoch
     * 
     * Regenerates row indices and shuffles them if enabled.
     * Automatically called at the start of each epoch.
     */
    void reset();

public:
    /**
     * @brief Construct a new DataLoader object
     * @param ds Reference to the source Dataset
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle data between epochs (default=false)
     */
    DataLoader(const Dataset& ds, size_t batch_size, 
                bool shuffle = false, unsigned int seed = 0);

    /**
     * @class Iterator
     * @brief Bidirectional iterator for batch access
     * 
     * Provides batch-by-batch traversal of the dataset.
     * Satisfies C++ LegacyInputIterator requirements.
     */
    class Iterator {
    private:
        const DataLoader& loader;  ///< Parent DataLoader reference
        size_t cursor;             ///< Current position in epoch
        
    public:
        /**
         * @brief Construct a new Iterator object
         * @param loader Parent DataLoader reference
         * @param cursor Starting position (in samples)
         */
        Iterator(const DataLoader& loader, size_t cursor);
        
        /**
         * @brief Get current batch's row indices
         * @return Vector of dataset row indices in current batch
         */
        std::vector<size_t> getIndices() const;

        /**
         * @brief Dereference operator
         * @return Dataset containing current batch
         * 
         * Creates a new Dataset containing only rows from current batch.
         * No data copying - uses Dataset's row selection constructor.
         */
        Dataset operator*() const;

        /**
         * @brief Prefix increment operator
         * @return Reference to updated iterator
         * 
         * Advances to next batch in epoch. When reaching end:
         * - Automatically resets parent DataLoader
         * - Returns end iterator
         */
        Iterator& operator++();

        /**
         * @brief Inequality comparison
         * @param other Iterator to compare with
         * @return true if iterators are at different positions
         */
        bool operator!=(const Iterator& other) const;
    };

    /**
     * @brief Get iterator to first batch
     * @return Iterator positioned at epoch start
     * 
     * Resets indices and shuffling before returning first batch.
     */
    Iterator begin();

    /**
     * @brief Get end-of-epoch sentinel
     * @return Iterator positioned after last batch
     */
    Iterator end();
};
