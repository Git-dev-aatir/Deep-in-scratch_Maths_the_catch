#include "Metrics/Correlation.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <tuple>

using namespace std;

// Helper function to compute column means
template<typename T>
vector<double> computeMeans(const vector<vector<T>>& dataset, size_t numRows, size_t numCols) {
    vector<double> means(numCols, 0.0);
    for (const auto& row : dataset) {
        for (size_t j = 0; j < numCols; ++j) {
            means[j] += static_cast<double>(row[j]);
        }
    }
    for (auto& mean : means) mean /= numRows;
    return means;
}

// Shape computation function
template<typename T>
tuple<size_t, size_t> getShape(const vector<vector<T>>& dataset) {
    if (dataset.empty()) return {0, 0};
    const size_t numCols = dataset[0].size();
    for (const auto& row : dataset) {
        if (row.size() != numCols) {
            throw invalid_argument("All rows must have same number of columns");
        }
    }
    return {dataset.size(), numCols};
}

// Covariance matrix implementation
template<typename T>
vector<vector<double>> computeCovarianceMatrix(const vector<vector<T>>& dataset) {
    const auto res = getShape(dataset);
    const size_t numRows = get<0>(res);
    const size_t numCols = get<1>(res);
    if (numRows < 2) 
        return vector<vector<double>>(numCols, vector<double>(numCols, 0.0));

    const auto means = computeMeans(dataset, numRows, numCols);
    vector<vector<double>> covMatrix(numCols, vector<double>(numCols, 0.0));

    for (const auto& row : dataset) {
        vector<double> centered(numCols);
        for (size_t j = 0; j < numCols; ++j) {
            centered[j] = static_cast<double>(row[j]) - means[j];
        }
        
        for (size_t i = 0; i < numCols; ++i) {
            for (size_t j = i; j < numCols; ++j) {
                covMatrix[i][j] += centered[i] * centered[j];
            }
        }
    }

    const double normFactor = 1.0 / (numRows - 1);
    for (size_t i = 0; i < numCols; ++i) {
        for (size_t j = i; j < numCols; ++j) {
            covMatrix[i][j] *= normFactor;
            if (i != j) covMatrix[j][i] = covMatrix[i][j];
        }
    }
    return covMatrix;
}

// Correlation matrix using covariance matrix
template<typename T>
vector<vector<double>> computeCorrelationMatrix(const vector<vector<T>>& dataset) {
    auto covMatrix = computeCovarianceMatrix(dataset);
    const size_t numCols = covMatrix.size();
    if (numCols == 0) return {};

    vector<vector<double>> corrMatrix(numCols, vector<double>(numCols, 0.0));
    for (size_t i = 0; i < numCols; ++i) {
        const double std_i = sqrt(covMatrix[i][i]);
        for (size_t j = 0; j < numCols; ++j) {
            const double std_j = sqrt(covMatrix[j][j]);
            const double denom = std_i * std_j;
            corrMatrix[i][j] = (denom < 1e-10) ? 0.0 : covMatrix[i][j] / denom;
        }
    }
    return corrMatrix;
}

// Correlation with target column
template<typename T>
vector<double> computeCorrelationWithAttribute(
    const vector<vector<T>>& dataset, 
    int target_col
) {
    auto res = getShape(dataset);
    const size_t numRows = get<0>(res);
    const size_t numCols = get<1>(res);
    if (numRows < 2) return vector<double>(numCols, 0.0);
    
    if (target_col == -1) target_col = numCols - 1;
    if (target_col < 0 || static_cast<size_t>(target_col) >= numCols) {
        throw out_of_range("Invalid target column index");
    }

    const auto means = computeMeans(dataset, numRows, numCols);
    vector<double> cov_target(numCols, 0.0);
    vector<double> var(numCols, 0.0);
    double var_target = 0.0;

    for (const auto& row : dataset) {
        const double centered_target = row[target_col] - means[target_col];
        var_target += centered_target * centered_target;
        
        for (size_t j = 0; j < numCols; ++j) {
            const double centered_val = row[j] - means[j];
            cov_target[j] += centered_val * centered_target;
            var[j] += centered_val * centered_val;
        }
    }

    vector<double> correlations(numCols);
    for (size_t j = 0; j < numCols; ++j) {
        const double cov = cov_target[j] / (numRows - 1);
        const double var_j = var[j] / (numRows - 1);
        correlations[j] = cov / sqrt(var_j * var_target / (numRows - 1));
    }
    return correlations;
}

// Correlation with external target vector
template<typename T>
vector<double> computeCorrelationWithTarget(
    const vector<vector<T>>& dataset, 
    const vector<T>& target
) {
    const auto res = getShape(dataset);
    const size_t numRows = get<0>(res);
    const size_t numCols = get<1>(res);
    if (numRows != target.size()) {
        throw invalid_argument("Target size must match dataset row count");
    }
    if (numRows < 2) return vector<double>(numCols, 0.0);

    const auto means = computeMeans(dataset, numRows, numCols);
    const double target_mean = accumulate(target.begin(), target.end(), 0.0) / numRows;
    
    vector<double> cov_target(numCols, 0.0);
    vector<double> var(numCols, 0.0);
    double var_target = 0.0;

    for (size_t i = 0; i < numRows; ++i) {
        const double centered_target = target[i] - target_mean;
        var_target += centered_target * centered_target;
        
        for (size_t j = 0; j < numCols; ++j) {
            const double centered_val = dataset[i][j] - means[j];
            cov_target[j] += centered_val * centered_target;
            var[j] += centered_val * centered_val;
        }
    }

    vector<double> correlations(numCols);
    for (size_t j = 0; j < numCols; ++j) {
        const double cov = cov_target[j] / (numRows - 1);
        const double var_j = var[j] / (numRows - 1);
        correlations[j] = cov / sqrt(var_j * var_target / (numRows - 1));
    }
    return correlations;
}

// Print sorted correlations
void printSortedCorrelations(const vector<double>& correlations, bool ascending) {
    vector<pair<size_t, double>> indexedCorrelations;
    for (size_t i = 0; i < correlations.size(); ++i)
        indexedCorrelations.emplace_back(i, correlations[i]);
    
    sort(indexedCorrelations.begin(), indexedCorrelations.end(),
         [ascending](const auto& a, const auto& b) {
             return ascending ? 
                 abs(a.second) < abs(b.second) :
                 abs(a.second) > abs(b.second);
         });

    for (const auto& element : indexedCorrelations)
        cout << "Feature " << element.first << ": " << element.second << "\n";
}

// Print highly correlated features
void printHighlyCorrelatedFeatures(const vector<vector<double>>& matrix, double threshold) {
    size_t n = matrix.size();
    vector<tuple<size_t, size_t, double>> highCorrPairs;
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double absCorr = abs(matrix[i][j]);
            if (absCorr >= threshold) {
                highCorrPairs.emplace_back(i, j, matrix[i][j]);
            }
        }
    }
    
    sort(highCorrPairs.begin(), highCorrPairs.end(),
         [](const auto& a, const auto& b) {
             return abs(get<2>(a)) > abs(get<2>(b));
         });
    
    for (const auto& pair : highCorrPairs) {
        cout << "Features " << get<0>(pair) << " - " << get<1>(pair) 
             << ": " << get<2>(pair) << " (abs=" << abs(get<2>(pair)) << ")\n";
    }
}

// Explicit template instantiations
template tuple<size_t, size_t> getShape<int>(const vector<vector<int>>&);
template tuple<size_t, size_t> getShape<double>(const vector<vector<double>>&);
template vector<vector<double>> computeCovarianceMatrix<int>(const vector<vector<int>>&);
template vector<vector<double>> computeCovarianceMatrix<double>(const vector<vector<double>>&);
template vector<vector<double>> computeCorrelationMatrix<int>(const vector<vector<int>>&);
template vector<vector<double>> computeCorrelationMatrix<double>(const vector<vector<double>>&);
template vector<double> computeCorrelationWithAttribute<int>(const vector<vector<int>>&, int);
template vector<double> computeCorrelationWithAttribute<double>(const vector<vector<double>>&, int);
template vector<double> computeCorrelationWithTarget<int>(const vector<vector<int>>&, const vector<int>&);
template vector<double> computeCorrelationWithTarget<double>(const vector<vector<double>>&, const vector<double>&);
