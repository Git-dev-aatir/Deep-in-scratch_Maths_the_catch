#pragma once

#include <vector>

/**
 * @brief Computes the Pearson Correlation Matrix for a numeric dataset.
 * 
 * @tparam T Numeric data type (int, double).
 * @param dataset The dataset represented as a vector of DataRow<T>.
 * @return A 2D vector (matrix) of doubles representing the correlation coefficients.
 */
template<typename T>
std::vector<std::vector<double>> computeCorrelationMatrix(
    const std::vector<std::vector<T>>& dataset);

/**
 * @brief Computes the Pearson correlation of all columns with a specified target column.
 * 
 * @tparam T Numeric data type (int, double).
 * @param dataset The dataset as a vector of DataRow<T>.
 * @param target_col The index of the target column.
 * @return A vector of doubles where each value is the correlation with the target column.
 */
template<typename T>
std::vector<double> computeCorrelationWithAttribute(
    const std::vector<std::vector<T>>& dataset, 
    int target_col = -1
);

/**
 * @brief Computes the Pearson correlation of all columns with a specified target vector.
 * 
 * @tparam T Numeric data type (int, double).
 * @param dataset The dataset as a vector of DataRow<T>.
 * @param target The vector of target values or labels.
 * @return A vector of doubles where each value is the correlation with the target vector.
 */
template<typename T>
std::vector<double> computeCorrelationWithTarget(
    const std::vector<std::vector<T>>& dataset, 
    const std::vector<T>& target
);

/**
 * @brief Prints correlations sorted by their absolute values in ascending or descending order.
 * 
 * @param correlations Vector of correlation values.
 * @param ascending If true, sorts in ascending order; otherwise, descending.
 */
void printSortedCorrelations(
    const std::vector<double>& correlations, 
    bool ascending = false
);

/**
 * @brief Prints all pairs of features whose absolute correlation is above a given threshold,
 * sorted in descending order of their absolute correlation value.
 * 
 * @param correlationMatrix The square correlation matrix [n x n].
 * @param threshold Minimum absolute correlation to consider as "highly correlated". Default: 0.8.
 */
void printHighlyCorrelatedFeatures(
    const std::vector<std::vector<double>>& correlationMatrix, 
    double threshold = 0.8
);
