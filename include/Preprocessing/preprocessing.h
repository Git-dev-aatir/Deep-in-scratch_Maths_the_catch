#pragma once

#include <set>
#include "dataset_utils.h" 
#include "Helper_functions.h"

/**
 * @brief Enumeration for specifying missing value imputation strategy.
 */
enum class ImputeStrategy { MEAN, MEDIAN, MODE };

/**
 * @brief Enumeration for specifying the outlier detection method.
 */
enum class OutlierMethod { Z_SCORE, IQR };

/**
 * @brief Standardizes the specified columns in the dataset.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to standardize.
 * @param columns Optional vector of column indices to standardize. If empty, all columns are standardized.
 */
template<typename T>
void standardize(Dataset<T>& data, const std::vector<size_t>& columns = {});
/**
 * @brief Normalizes the specified columns in the dataset to range [0, 1].
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to standardize.
 * @param columns Optional vector of column indices to standardize. If empty, all columns are normalized.
 */
template<typename T>
void normalize(Dataset<T>& data, const std::vector<size_t>& columns = {});

/**
 * @brief Finds and prints locations of missing values in the dataset.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to inspect.
 */
template<typename T>
void findMissingValues(const Dataset<T>& data);

/**
 * @brief Removes rows from the dataset that contain missing values.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to modify.
 */
template<typename T>
void removeRowsWithMissingValues(Dataset<T>& data);

/**
 * @brief Replaces missing values in the dataset using the specified strategy.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to process.
 * @param strategy Imputation strategy to use.
 * @param columns Optional vector of column indices to process. If empty, all columns are processed.
 */
template<typename T>
void replaceMissingValues(Dataset<T>& data, ImputeStrategy strategy, const std::vector<size_t>& columns = {});

/**
 * @brief Replaces missing values in the dataset with a custom value.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to process.
 * @param custom_value Value to replace missing entries with.
 * @param columns Optional vector of column indices to process. If empty, all columns are processed.
 */
template<typename T>
void replaceMissingWithCustomValue(Dataset<T>& data, const T& custom_value, const std::vector<size_t>& columns = {});

/**
 * @brief Removes outliers from the dataset using the specified method.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to process.
 * @param method Outlier detection method to use (Z-Score or IQR).
 * @param threshold Threshold value for the method (e.g., Z-score value or IQR multiplier).
 * @param columns Optional vector of column indices to process. If empty, all columns are processed.
 */
template<typename T>
void removeOutliers(Dataset<T>& data, OutlierMethod method = OutlierMethod::Z_SCORE,
                    double threshold = 3.0, const std::vector<size_t>& columns = {});

/**
 * @brief Removes specified columns from the dataset.
 * 
 * @tparam T The datatype of the dataset elements (int, double, std::string).
 * @param dataset The dataset to process. Each row is modified in-place.
 * @param columns_to_remove A vector of column indices to remove.
 */
template<typename T>
void removeColumns(Dataset<T>& dataset, const std::vector<size_t>& columns_to_remove);
