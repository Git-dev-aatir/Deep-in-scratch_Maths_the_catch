#pragma once

#include "Dataset.h"
#include <vector>
#include <cstddef>
#include <set>

enum class ImputeStrategy { Mean, Median, Mode };
enum class OutlierMethod { ZScore, IQR };

namespace Preprocessing {

// Standardize specified columns (or all if empty)
void standardize(Dataset& dataset, const std::vector<size_t>& columns = {});

// Min-max normalize specified columns (or all if empty)
void minMaxNormalize(Dataset& dataset, const std::vector<size_t>& columns = {});

// Print locations of missing (NaN) values
void printMissingValues(const Dataset& dataset);

// Remove all rows with any missing value
void dropRowsWithMissing(Dataset& dataset);

// Impute missing values with mean/median/mode for specified columns
void imputeMissing(Dataset& dataset, ImputeStrategy strategy, const std::vector<size_t>& columns = {});

// Impute missing values with a custom value for specified columns
void fillMissingWithValue(Dataset& dataset, double value, const std::vector<size_t>& columns = {});

// Remove outliers using Z-score or IQR for specified columns
void dropOutliers(Dataset& dataset, OutlierMethod method, double threshold = 3.0, const std::vector<size_t>& columns = {});

// Remove specified columns
void dropColumns(Dataset& dataset, const std::vector<size_t>& columnsToRemove);

// One-hot encode specified categorical columns (integer-valued)
void oneHotEncode(Dataset& dataset, const std::vector<size_t>& categoricalColumns);

// Shuffle all rows in the dataset
void shuffleRows(Dataset& dataset);

}
