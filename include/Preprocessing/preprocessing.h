#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <set>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <numeric>
#include <iomanip>
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include "dataset_utils.h" // Assuming Dataset, DataRow are defined here

using namespace std;

/**
 * @brief Get the median of a vector within a specified range.
 * 
 * @tparam T Data type.
 * @param v Input vector.
 * @param start Start index.
 * @param end End index.
 * @return T Median value.
 */
template<typename T>
inline T get_median (const vector<T>& v, size_t start, size_t end) {
    size_t len = end - start + 1;
    if (len == 0) return T(); // return default-constructed T if empty

    size_t mid = start + len / 2;
    if (len & 1)
        return v[mid];
    else
        return static_cast<T>((static_cast<double>(v[mid - 1] + v[mid])) / 2.0);
};

/**
 * @brief Checks for missing values in floating point types.
 * 
 * @tparam T Floating point type.
 * @param value Value to check.
 * @return true if missing (NaN), false otherwise.
 */
template<typename T>
typename enable_if<is_floating_point<T>::value, bool>::type
isMissing(const T& value) {
    return isnan(value);
}

/**
 * @brief Checks for missing values in integral types.
 * 
 * @tparam T Integral type.
 * @param value Value to check.
 * @return true if missing (equal to min limit), false otherwise.
 */
template<typename T>
typename enable_if<is_integral<T>::value, bool>::type
isMissing(const T& value) {
    return value == numeric_limits<T>::min();
}


/**
 * @brief Checks for missing values in string type.
 * 
 * @param value String to check.
 * @return true if missing (empty string), false otherwise.
 */
inline bool isMissing(const string& value) {
    return value.empty();
}


/**
 * @brief Standardizes the specified columns in the dataset.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to standardize.
 * @param columns Optional vector of column indices to standardize. If empty, all columns are standardized.
 */
template<typename T>
void standardize(Dataset<T>& data, const vector<size_t>& columns = {}) {
    static_assert(is_arithmetic<T>::value, "Standardize only works with numeric types.");

    size_t n_cols = data[0].size();
    if (data.empty()) return;

    vector<size_t> target_cols;
    if (columns.empty()) {
        target_cols.resize(n_cols);
        iota(target_cols.begin(), target_cols.end(), 0);
    } else {
        target_cols = columns;
    }

    for (size_t col : target_cols) {
        if (col >= n_cols) {
            cerr << "Warning: Column index " << col << " is out of range." << endl;
            continue;
        }
        vector<T> col_vals;

        for (const auto& row : data)
            if (!isMissing(row[col]))
                col_vals.push_back(row[col]);

        if (col_vals.empty()) continue;

        double mean = accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();
        double sq_sum = inner_product(col_vals.begin(), col_vals.end(), col_vals.begin(), 0.0);
        double std_dev = sqrt(sq_sum / col_vals.size() - mean * mean);

        if (std_dev == 0) continue;  
        // Skip standardization if std_dev is zero (constant column)

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>((row[col] - mean) / std_dev);
    }
}

/**
 * @brief Normalizes the specified columns in the dataset to range [0, 1].
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to standardize.
 * @param columns Optional vector of column indices to standardize. If empty, all columns are normalized.
 */
template<typename T>
void normalize(Dataset<T>& data, const vector<size_t>& columns = {}) {
    static_assert(is_arithmetic<T>::value, "Normalize only works with numeric types.");

    size_t n_cols = data[0].size();
    if (data.empty()) return;

    vector<size_t> target_cols;
    if (columns.empty()) {
        target_cols.resize(n_cols);
        iota(target_cols.begin(), target_cols.end(), 0);
    } else {
        target_cols = columns;
    }

    for (size_t col : target_cols) {
        T min_val = numeric_limits<T>::max();
        T max_val = numeric_limits<T>::lowest();

        for (const auto& row : data)
            if (!isMissing(row[col])) {
                min_val = min(min_val, row[col]);
                max_val = max(max_val, row[col]);
            }
        
        if (min_val == max_val) continue; 
        // Skip normalization if min_val and max_val are same (constant column)

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>(
                    (static_cast<double>(row[col]) - static_cast<double>(min_val)) /
                    (static_cast<double>(max_val) - static_cast<double>(min_val))
                );
    }
}

/**
 * @brief Finds and prints locations of missing values in the dataset.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to inspect.
 */
template<typename T>
void findMissingValues(const Dataset<T>& data) {
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            if (isMissing(data[i][j]))
                cout << "Missing at Data Point: " << i << ", Attribute: " << j << endl;
}


/**
 * @brief Removes rows from the dataset that contain missing values.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to modify.
 */
template<typename T>
void removeRowsWithMissingValues(Dataset<T>& data) {
    data.erase(remove_if(data.begin(), data.end(), [](const DataRow<T>& row) {
        return any_of(row.begin(), row.end(), [](const T& val) { return isMissing(val); });
    }), data.end());
}


/**
 * @brief Enumeration for specifying missing value imputation strategy.
 */
enum class ImputeStrategy { MEAN, MEDIAN, MODE };

/**
 * @brief Replaces missing values in the dataset using the specified strategy.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to process.
 * @param strategy Imputation strategy to use.
 * @param columns Optional vector of column indices to process. If empty, all columns are processed.
 */
template<typename T>
void replaceMissingValues(Dataset<T>& data, ImputeStrategy strategy, const vector<size_t>& columns = {}) {
    size_t n_cols = data[0].size();
    if (data.empty()) return;

    vector<size_t> target_cols;
    if (columns.empty()) {
        target_cols.resize(n_cols);
        iota(target_cols.begin(), target_cols.end(), 0);
    } else {
        target_cols = columns;
    }

    for (size_t col : target_cols) {
        vector<T> col_vals;

        for (const auto& row : data)
            if (!isMissing(row[col]))
                col_vals.push_back(row[col]);

        if (col_vals.empty()) continue;

        T replacement;

        switch (strategy) {
            case ImputeStrategy::MEAN:
                replacement = static_cast<T>(accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size());
                break;
            case ImputeStrategy::MEDIAN:
                sort(col_vals.begin(), col_vals.end());
                replacement = get_median<T>(col_vals, 0, col_vals.size()-1);
                break;
            case ImputeStrategy::MODE: {
                unordered_map<T, int> freq;
                for (const auto& val : col_vals) freq[val]++;
                replacement = max_element(freq.begin(), freq.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; })->first;
                break;
            }
        }

        for (auto& row : data)
            if (isMissing(row[col]))
                row[col] = replacement;
    }
}

/**
 * @brief Replaces missing values in the dataset with a custom value.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to process.
 * @param custom_value Value to replace missing entries with.
 * @param columns Optional vector of column indices to process. If empty, all columns are processed.
 */
template<typename T>
void replaceMissingWithCustomValue(Dataset<T>& data, const T& custom_value, const vector<size_t>& columns = {}) {
    size_t n_cols = data[0].size();
    if (data.empty()) return;

    vector<size_t> target_cols;
    if (columns.empty()) {
        target_cols.resize(n_cols);
        iota(target_cols.begin(), target_cols.end(), 0);
    } else {
        target_cols = columns;
    }

    for (auto& row : data)
        for (size_t col : target_cols)
            if (isMissing(row[col]))
                row[col] = custom_value;
}


/**
 * @brief Displays basic statistical descriptions of the dataset columns, including min, max, mean, median, and quartiles.
 * 
 * @tparam T Data type of dataset entries.
 * @param data Dataset to describe.
 */
template<typename T>
void describeDataset(const Dataset<T>& data) {
    static_assert(is_arithmetic<T>::value, "Describe only works with numeric types.");

    size_t n_cols = data[0].size();
    if (data.empty()) return;

    // Header
    cout << endl << string(102, '-') << endl;
    cout << left << setw(10) << "Column" 
         << setw(15) << "Mean"
         << setw(15) << "StdDev"  
         << setw(10) << "Min" 
         << setw(15) << "25%" 
         << setw(15) << "Median" 
         << setw(15) << "75%" 
         << setw(10) << "Max" 
         << endl;
    cout << string(102, '-') << endl;

    for (size_t col = 0; col < n_cols; ++col) {
        vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col]))
                col_vals.push_back(row[col]);

        if (col_vals.empty()) continue;

        sort(col_vals.begin(), col_vals.end());
        double min_val = static_cast<double>(col_vals.front());
        double max_val = static_cast<double>(col_vals.back());
        double mean = accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();

        // Standard Deviation
        size_t n = col_vals.size();
        double sq_sum = 0.0;
        for (const auto& val : col_vals) {
            double diff = val - mean;
            sq_sum += diff * diff;
        }
        double stddev = sqrt(sq_sum / n);

        double median = get_median<double>(col_vals, 0, n - 1);
        size_t mid = n / 2;
        double q1 = get_median<double>(col_vals, 0, mid - 1);
        double q3 = (n & 1) ? get_median<double>(col_vals, mid+1, n - 1)
                                : get_median<double>(col_vals, mid, n - 1);

        if (col != 0) cout << endl;
        cout << left << setw(10) << col
             << setw(15) << fixed << setprecision(2) << mean
             << setw(15) << fixed << setprecision(2) << stddev
             << setw(10) << fixed << setprecision(2) << min_val
             << setw(15) << fixed << setprecision(2) << q1
             << setw(15) << fixed << setprecision(2) << median
             << setw(15) << fixed << setprecision(2) << q3
             << setw(10) << fixed << setprecision(2) << max_val
             << endl;
    }
    cout << string(102, '-') << endl << endl;
}

/**
 * @brief Enumeration for specifying the outlier detection method.
 */
enum class OutlierMethod { Z_SCORE, IQR };

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
                    double threshold = 3.0, const vector<size_t>& columns = {}) {
    static_assert(is_arithmetic<T>::value, "Outlier removal only works with numeric types.");
    size_t n_cols = data[0].size();
    if (data.empty()) return;

    vector<size_t> target_cols;
    if (columns.empty()) {
        target_cols.resize(n_cols);
        iota(target_cols.begin(), target_cols.end(), 0);
    } else {
        target_cols = columns;
    }

    vector<bool> to_remove(data.size(), false); // flag rows for removal

    for (size_t col : target_cols) {
        vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col]))
                col_vals.push_back(row[col]);

        if (col_vals.size() < 2) continue;

        if (method == OutlierMethod::Z_SCORE) {
            double mean = accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();
            double sq_sum = inner_product(col_vals.begin(), col_vals.end(), col_vals.begin(), 0.0);
            double std_dev = sqrt(sq_sum / col_vals.size() - mean * mean);
            if (std_dev == 0) continue;

            for (size_t i = 0; i < data.size(); ++i) {
                if (!isMissing(data[i][col])) {
                    double z = (data[i][col] - mean) / std_dev;
                    if (abs(z) > threshold) to_remove[i] = true;
                }
            }
        } else if (method == OutlierMethod::IQR) {
            sort(col_vals.begin(), col_vals.end());
            size_t n = col_vals.size();
            double q1 = get_median<double>(col_vals, 0, n/2 - 1);
            double q3 = get_median<double>(col_vals, (n+1)/2, n - 1);
            double iqr = q3 - q1;
            T lower = static_cast<T> (q1 - threshold * iqr);
            T upper = static_cast<T> (q3 + threshold * iqr);

            for (size_t i = 0; i < data.size(); ++i) {
                if (!isMissing(data[i][col])) {
                    if (data[i][col] < lower || data[i][col] > upper) to_remove[i] = true;
                }
            }
        }
    }

    // Actually remove marked rows
    Dataset<T> filtered;
    for (size_t i = 0; i < data.size(); ++i)
        if (!to_remove[i])
            filtered.push_back(data[i]);
        else {
            cout << "Removed data point {" << i << "} : ";
            for (size_t j=0; j<data[i].size(); ++j)
                cout << data[i][j] << " | ";
        }
    
    data = filtered;
}


/**
 * @brief Computes the Pearson Correlation Matrix for a numeric dataset.
 * 
 * @tparam T Numeric data type (int, double).
 * @param dataset The dataset represented as a vector of DataRow<T>.
 * @return A 2D vector (matrix) of doubles representing the correlation coefficients.
 */
template<typename T>
Dataset<double> computeCorrelationMatrix(const Dataset<T> &dataset) {
    if (dataset.empty() || dataset[0].empty()) {
        throw invalid_argument("Dataset is empty or malformed.");
    }

    size_t numRows = dataset.size();
    size_t numCols = dataset[0].size();

    // Compute mean for each column
    vector<double> means(numCols, 0.0);
    for (const auto &row : dataset) {
        for (size_t j = 0; j < numCols; ++j) {
            means[j] += static_cast<double>(row[j]);
        }
    }
    for (auto &mean : means) {
        mean /= static_cast<double>(numRows);
    }

    // Compute standard deviation for each column
    vector<double> stdDevs(numCols, 0.0);
    for (const auto &row : dataset) {
        for (size_t j = 0; j < numCols; ++j) {
            double diff = static_cast<double>(row[j]) - means[j];
            stdDevs[j] += diff * diff;
        }
    }
    for (auto &stdDev : stdDevs) {
        stdDev = sqrt(stdDev / static_cast<double>(numRows - 1));
        if (stdDev == 0.0) stdDev = 1e-8; // avoid divide by zero
    }

    // Compute correlation matrix
    Dataset<double> corrMatrix(numCols, DataRow<double>(numCols, 0.0));

    for (size_t i = 0; i < numCols; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            double covariance = 0.0;
            for (size_t k = 0; k < numRows; ++k) {
                double val_i = static_cast<double>(dataset[k][i]);
                double val_j = static_cast<double>(dataset[k][j]);
                covariance += (val_i - means[i]) * (val_j - means[j]);
            }
            covariance /= static_cast<double>(numRows - 1);
            corrMatrix[i][j] = covariance / (stdDevs[i] * stdDevs[j]);
        }
    }

    return corrMatrix;
}

/**
 * @brief Computes the Pearson correlation of all columns with a specified target column.
 * 
 * @tparam T Numeric data type (int, double).
 * @param dataset The dataset as a vector of DataRow<T>.
 * @param target_col The index of the target column.
 * @return A vector of doubles where each value is the correlation with the target column.
 */
template<typename T>
vector<double> computeCorrelationWithTarget(const Dataset<T> &dataset, size_t target_col=-1) {
    if (dataset.empty() || dataset[0].empty()) {
        throw invalid_argument("Dataset is empty or malformed.");
    }

    size_t numRows = dataset.size();
    size_t numCols = dataset[0].size();

    if (target_col == -1) target_col = numCols-1;
    if (target_col >= numCols) {
        throw out_of_range("Target column index out of bounds.");
    }

    // Compute means
    vector<double> means(numCols, 0.0);
    for (const auto &row : dataset) {
        for (size_t j = 0; j < numCols; ++j) {
            means[j] += static_cast<double>(row[j]);
        }
    }
    for (auto &mean : means) {
        mean /= static_cast<double>(numRows);
    }

    // Compute standard deviations
    vector<double> stdDevs(numCols, 0.0);
    for (const auto &row : dataset) {
        for (size_t j = 0; j < numCols; ++j) {
            double diff = static_cast<double>(row[j]) - means[j];
            stdDevs[j] += diff * diff;
        }
    }
    for (auto &stdDev : stdDevs) {
        stdDev = sqrt(stdDev / static_cast<double>(numRows - 1));
        if (stdDev == 0.0) stdDev = 1e-8; // avoid divide by zero
    }

    // Compute correlation with target column
    vector<double> correlations(numCols, 0.0);

    for (size_t j = 0; j < numCols; ++j) {
        double covariance = 0.0;
        for (size_t i = 0; i < numRows; ++i) {
            double val_j = static_cast<double>(dataset[i][j]);
            double val_target = static_cast<double>(dataset[i][target_col]);
            covariance += (val_j - means[j]) * (val_target - means[target_col]);
        }
        covariance /= static_cast<double>(numRows - 1);
        correlations[j] = covariance / (stdDevs[j] * stdDevs[target_col]);
    }

    return correlations;
}

/**
 * @brief Prints correlations sorted by their absolute values in ascending or descending order.
 * 
 * @param correlations Vector of correlation values.
 * @param ascending If true, sorts in ascending order; otherwise, descending.
 */
void printSortedCorrelations(const vector<double> &correlations, bool ascending = false) {
    // Create vector of (index, correlation) pairs
    vector<pair<size_t, double>> indexedCorrelations;
    for (size_t i = 0; i < correlations.size(); ++i) {
        indexedCorrelations.emplace_back(i, correlations[i]);
    }

    // Sort based on absolute value of correlation
    sort(indexedCorrelations.begin(), indexedCorrelations.end(),
        [ascending](const pair<size_t, double>& a, const pair<size_t, double>& b) {
            if (ascending)
                return abs(a.second) < abs(b.second);
            else
                return abs(a.second) > abs(b.second);
        });

    // Print sorted correlations
    cout << (ascending ? "Correlations sorted by ascending absolute value:\n"
                            : "Correlations sorted by descending absolute value:\n");
    cout << string(24, '-') << endl;
    cout << "Column\t|   Correlation\n";
    cout << string(24, '-') << endl;
    for (const auto& element : indexedCorrelations) {
        cout << element.first << "\t|   " << fixed << setprecision(4) << element.second << endl;
    }
    cout << string(24, '-') << endl;
}

/**
 * @brief Prints all pairs of features whose absolute correlation is above a given threshold,
 * sorted in descending order of their absolute correlation value.
 * 
 * @param correlationMatrix The square correlation matrix [n x n].
 * @param threshold Minimum absolute correlation to consider as "highly correlated". Default: 0.8.
 */
void printHighlyCorrelatedFeatures(const Dataset<double> &correlationMatrix, double threshold = 0.8) {
    size_t n_features = correlationMatrix.size();

    // Vector to store pairs: (feature1, feature2, correlation)
    vector<tuple<size_t, size_t, double>> correlatedPairs;

    // Iterate only upper triangular matrix to avoid duplicates and self-correlation
    for (size_t i = 0; i < n_features; ++i) {
        for (size_t j = i + 1; j < n_features; ++j) {
            double corr = correlationMatrix[i][j];
            if (abs(corr) >= threshold) {
                correlatedPairs.push_back(make_tuple(i, j, corr));
            }
        }
    }

    // Sort pairs by absolute correlation in descending order
    sort(correlatedPairs.begin(), correlatedPairs.end(),
        [](const tuple<size_t, size_t, double>& a, const tuple<size_t, size_t, double>& b) {
            return abs(get<2>(a)) > abs(get<2>(b));
        });

    // Print results
    cout << "Highly Correlated Feature Pairs (|correlation| >= " << threshold << "):\n";
    cout << string(44, '-') << endl;
    cout << "Feature1\tFeature2\tCorrelation\n";
    cout << string(44, '-') << endl;
    for (const auto& t : correlatedPairs) {
        cout << get<0>(t) << "\t\t" << get<1>(t) << "\t\t"
                  << fixed << setprecision(4) << get<2>(t) << "\n";
    }
    cout << string(44, '-') << endl << endl;
}

/**
 * @brief Removes specified columns from the dataset.
 * 
 * @tparam T The datatype of the dataset elements (int, double, std::string).
 * @param dataset The dataset to process. Each row is modified in-place.
 * @param columns_to_remove A vector of column indices to remove.
 */
template<typename T>
void removeColumns(Dataset<T>& dataset, const vector<size_t>& columns_to_remove) {
    if (dataset.empty() || columns_to_remove.empty()) return;

    // Create a sorted and unique set of columns to remove for quick lookup
    set<size_t> columnsSet(columns_to_remove.begin(), columns_to_remove.end());

    size_t n_cols = dataset[0].size();

    // Check if all indices are valid
    for (size_t col : columnsSet) {
        if (col >= n_cols) {
            cerr << "Error: Column index " << col << " out of bounds!\n";
            return;
        }
    }

    // Remove specified columns from each row
    for (auto& row : dataset) {
        DataRow<T> newRow;
        for (size_t i = 0; i < row.size(); ++i) {
            if (columnsSet.find(i) == columnsSet.end()) {
                newRow.push_back(row[i]);
            }
        }
        row = newRow;
    }
}


#endif // PREPROCESSING_H
