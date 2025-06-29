#pragma once

#include <vector>
#include <tuple>

/**
 * @brief Computes dimensions of dataset with validation
 * 
 * @tparam T Numeric type
 * @param dataset Input data matrix
 * @return std::tuple<size_t, size_t> (rows, columns)
 * @throws std::invalid_argument for inconsistent row sizes
 */
template<typename T>
std::tuple<size_t, size_t> getShape(const std::vector<std::vector<T>>& dataset);

/**
 * @brief Computes covariance matrix for dataset
 * 
 * @tparam T Numeric type
 * @param dataset Input data matrix
 * @return std::vector<std::vector<double>> Covariance matrix
 */
template<typename T>
std::vector<std::vector<double>> computeCovarianceMatrix(
    const std::vector<std::vector<T>>& dataset);

/**
 * @brief Computes Pearson Correlation Matrix
 * 
 * @tparam T Numeric type
 * @param dataset Input data matrix
 * @return std::vector<std::vector<double>> Correlation matrix
 */
template<typename T>
std::vector<std::vector<double>> computeCorrelationMatrix(
    const std::vector<std::vector<T>>& dataset);

/**
 * @brief Computes Pearson correlation of all columns with a specified target column
 * 
 * @tparam T Numeric type
 * @param dataset Input data matrix
 * @param target_col Index of target column
 * @return std::vector<double> Correlation values
 */
template<typename T>
std::vector<double> computeCorrelationWithAttribute(
    const std::vector<std::vector<T>>& dataset, 
    int target_col = -1
);

/**
 * @brief Computes Pearson correlation of all columns with a specified target vector
 * 
 * @tparam T Numeric type
 * @param dataset Input data matrix
 * @param target Target vector
 * @return std::vector<double> Correlation values
 */
template<typename T>
std::vector<double> computeCorrelationWithTarget(
    const std::vector<std::vector<T>>& dataset, 
    const std::vector<T>& target
);

/**
 * @brief Prints correlations sorted by absolute value
 * 
 * @param correlations Correlation values
 * @param ascending Sort order (default descending)
 */
void printSortedCorrelations(
    const std::vector<double>& correlations, 
    bool ascending = false
);

/**
 * @brief Prints highly correlated feature pairs
 * 
 * @param correlationMatrix Correlation matrix
 * @param threshold Correlation threshold (default 0.8)
 */
void printHighlyCorrelatedFeatures(
    const std::vector<std::vector<double>>& correlationMatrix, 
    double threshold = 0.8
);
