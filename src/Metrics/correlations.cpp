#include "../../include/Metrics/Correlations.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>

using namespace std;

template<typename T>
std::vector<std::vector<double>> computeCorrelationMatrix(const std::vector<std::vector<T>>& dataset) {
    size_t numRows = dataset.size();
    size_t numCols = dataset[0].size();
    vector<double> means(numCols, 0.0);

    for (const auto& row : dataset)
        for (size_t j = 0; j < numCols; ++j)
            means[j] += row[j];
    for (auto& mean : means) mean /= numRows;

    vector<double> stdDevs(numCols, 0.0);
    for (const auto& row : dataset)
        for (size_t j = 0; j < numCols; ++j)
            stdDevs[j] += pow(row[j] - means[j], 2);
    for (auto& stdDev : stdDevs)
        stdDev = sqrt(stdDev / (numRows - 1));

    std::vector<std::vector<double>> corrMatrix(numCols, std::vector<double>(numCols, 0.0));
    for (size_t i = 0; i < numCols; ++i)
        for (size_t j = 0; j < numCols; ++j) {
            double cov = 0.0;
            for (size_t k = 0; k < numRows; ++k)
                cov += (dataset[k][i] - means[i]) * (dataset[k][j] - means[j]);
            corrMatrix[i][j] = cov / ((numRows - 1) * stdDevs[i] * stdDevs[j]);
        }
    return corrMatrix;
}

template<typename T>
vector<double> computeCorrelationWithAttribute(const std::vector<std::vector<T>>& dataset, 
                                               int target_col) {
    size_t numRows = dataset.size();
    size_t numCols = dataset[0].size();
    if (target_col == -1) target_col = numCols - 1;

    vector<double> means(numCols, 0.0);
    for (const auto& row : dataset)
        for (size_t j = 0; j < numCols; ++j)
            means[j] += row[j];
    for (auto& mean : means) mean /= numRows;

    vector<double> stdDevs(numCols, 0.0);
    for (const auto& row : dataset)
        for (size_t j = 0; j < numCols; ++j)
            stdDevs[j] += pow(row[j] - means[j], 2);
    for (auto& stdDev : stdDevs)
        stdDev = sqrt(stdDev / (numRows - 1));

    vector<double> corr(numCols, 0.0);
    for (size_t j = 0; j < numCols; ++j) {
        double cov = 0.0;
        for (size_t i = 0; i < numRows; ++i)
            cov += (dataset[i][j] - means[j]) * (dataset[i][target_col] - means[target_col]);
        corr[j] = cov / ((numRows - 1) * stdDevs[j] * stdDevs[target_col]);
    }
    return corr;
}

template<typename T>
vector<double> computeCorrelationWithTarget(const std::vector<std::vector<T>>& dataset, 
                                            const vector<T>& target) {
    size_t numRows = dataset.size();
    size_t numCols = dataset[0].size();

    vector<double> means(numCols, 0.0);
    double target_mean = accumulate(target.begin(), target.end(), 0.0) / numRows;

    for (const auto& row : dataset)
        for (size_t j = 0; j < numCols; ++j)
            means[j] += row[j];
    for (auto& mean : means) mean /= numRows;

    vector<double> stdDevs(numCols, 0.0);
    double target_stdDev = 0.0;
    for (size_t i = 0; i < numRows; ++i) {
        target_stdDev += pow(target[i] - target_mean, 2);
        for (size_t j = 0; j < numCols; ++j)
            stdDevs[j] += pow(dataset[i][j] - means[j], 2);
    }
    target_stdDev = sqrt(target_stdDev / (numRows - 1));
    for (auto& stdDev : stdDevs) stdDev = sqrt(stdDev / (numRows - 1));

    vector<double> corr(numCols, 0.0);
    for (size_t j = 0; j < numCols; ++j) {
        double cov = 0.0;
        for (size_t i = 0; i < numRows; ++i)
            cov += (dataset[i][j] - means[j]) * (target[i] - target_mean);
        corr[j] = cov / ((numRows - 1) * stdDevs[j] * target_stdDev);
    }
    return corr;
}

void printSortedCorrelations(const vector<double>& correlations, bool ascending) {
    vector<pair<size_t, double>> indexedCorrelations;
    for (size_t i = 0; i < correlations.size(); ++i)
        indexedCorrelations.emplace_back(i, correlations[i]);
    sort(indexedCorrelations.begin(), indexedCorrelations.end(),
         [ascending](const auto& a, const auto& b) {
             return ascending ? abs(a.second) < abs(b.second) : abs(a.second) > abs(b.second);
         });

    for (const auto& idx_and_corr : indexedCorrelations)
        cout << idx_and_corr.first << " : " << idx_and_corr.second << endl;
}

void printHighlyCorrelatedFeatures(const std::vector<std::vector<double>>& matrix, double threshold) {
    size_t n = matrix.size();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (abs(matrix[i][j]) >= threshold)
                cout << i << " - " << j << " : " << matrix[i][j] << endl;
}

// Explicit template instantiation
template std::vector<std::vector<double>> computeCorrelationMatrix<int>(
    const std::vector<std::vector<int>>&
);
template std::vector<std::vector<double>> computeCorrelationMatrix<double>(
    const std::vector<std::vector<double>>&
);
template std::vector<double> computeCorrelationWithAttribute<int>(
    const std::vector<std::vector<int>>&, 
    int
);
template std::vector<double> computeCorrelationWithAttribute<double>(
    const std::vector<std::vector<double>>&, 
    int
);
template std::vector<double> computeCorrelationWithTarget<int>(
    const std::vector<std::vector<int>>&, 
    const vector<int>&
);
template std::vector<double> computeCorrelationWithTarget<double>(
    const std::vector<std::vector<double>>&, 
    const vector<double>&
);
