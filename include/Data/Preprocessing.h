#pragma once

#include "Dataset.h"
#include <vector>
#include <cstddef>
#include <set>

/**
 * @enum ImputeStrategy
 * @brief Strategy for imputing missing values
 */
enum class ImputeStrategy {
    Mean,   ///< Replace missing values with column mean
    Median, ///< Replace missing values with column median
    Mode    ///< Replace missing values with most frequent value
};

/**
 * @enum OutlierMethod
 * @brief Method for detecting outliers
 */
enum class OutlierMethod {
    ZScore, ///< Use standard deviation-based detection
    IQR     ///< Use interquartile range-based detection
};

/**
 * @namespace Preprocessing
 * @brief Collection of dataset preprocessing operations
 * 
 * Provides common data preprocessing techniques for cleaning,
 * normalizing, and transforming datasets before model training.
 */
namespace Preprocessing {

/**
 * @brief Standardize specified columns to zero mean and unit variance
 * @param dataset Dataset to modify
 * @param columns Column indices to standardize (empty = all columns)
 * 
 * Computes for each value in column j:
 * \( x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j} \)
 */
void standardize(Dataset& dataset, const std::vector<size_t>& columns = {});

/**
 * @brief Scale specified columns to [0,1] range
 * @param dataset Dataset to modify
 * @param columns Column indices to normalize (empty = all columns)
 * 
 * Computes for each value in column j:
 * \( x'_{ij} = \frac{x_{ij} - \min_j}{\max_j - \min_j} \)
 */
void minMaxNormalize(Dataset& dataset, const std::vector<size_t>& columns = {});

/**
 * @brief Print locations of missing values (NaN)
 * @param dataset Dataset to inspect
 * 
 * Output format:
 * [Row 5, Col 2]: NaN
 * [Row 8, Col 7]: NaN
 * Total missing: 2
 */
void printMissingValues(const Dataset& dataset);

/**
 * @brief Remove rows containing any missing values
 * @param dataset Dataset to modify
 * 
 * Deletes entire rows where at least one value is NaN.
 * Preserves original row order of remaining data.
 */
void dropRowsWithMissing(Dataset& dataset);

/**
 * @brief Replace missing values with statistical measures
 * @param dataset Dataset to modify
 * @param strategy Imputation strategy (Mean/Median/Mode)
 * @param columns Column indices to process (empty = all columns)
 * 
 * For each specified column:
 * - Mean: Replace NaNs with column average
 * - Median: Replace NaNs with column median
 * - Mode: Replace NaNs with most frequent value
 */
void imputeMissing(Dataset& dataset, ImputeStrategy strategy, const std::vector<size_t>& columns = {});

/**
 * @brief Replace missing values with a constant
 * @param dataset Dataset to modify
 * @param value Replacement value
 * @param columns Column indices to process (empty = all columns)
 */
void fillMissingWithValue(Dataset& dataset, double value, const std::vector<size_t>& columns = {});

/**
 * @brief Remove outlier rows using statistical methods
 * @param dataset Dataset to modify
 * @param method Outlier detection method
 * @param threshold Detection sensitivity (default=3.0)
 * @param columns Column indices to analyze (empty = all columns)
 * 
 * - ZScore: Removes rows where any |x - μ| > threshold * σ
 * - IQR: Removes rows where any x < Q1 - threshold*IQR or x > Q3 + threshold*IQR
 */
void dropOutliers(Dataset& dataset, OutlierMethod method, double threshold = 3.0, const std::vector<size_t>& columns = {});

/**
 * @brief Remove specified columns from dataset
 * @param dataset Dataset to modify
 * @param columnsToRemove Indices of columns to delete
 * 
 * Columns are removed in-place. Indices should be in descending order
 * to avoid position shifting during deletion.
 */
void dropColumns(Dataset& dataset, const std::vector<size_t>& columnsToRemove);

/**
 * @brief Convert categorical columns to one-hot encoding
 * @param dataset Dataset to modify
 * @param categoricalColumns Indices of categorical columns
 * 
 * Requirements:
 * - Columns must contain integer values
 * - Values should represent category indices
 * 
 * Example: Column [1, 0, 2] with 3 categories becomes:
 * [[0,1,0], [1,0,0], [0,0,1]]
 */
void oneHotEncode(Dataset& dataset, const std::vector<size_t>& categoricalColumns);

/**
 * @brief Randomly shuffle dataset rows
 * @param dataset Dataset to modify
 * 
 * Uses Fisher-Yates shuffle algorithm. Preserves
 * relationships between columns in each row.
 */
void shuffleRows(Dataset& dataset);

} // namespace Preprocessing
