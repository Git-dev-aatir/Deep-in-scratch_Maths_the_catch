#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <numeric>
#include <iomanip>
#include <type_traits>
#include "dataset_utils.h" // Assuming Dataset, DataRow are defined here

using namespace std;

// helper function to get median
template<typename T>
inline T get_median (const vector<T>& v, size_t start, size_t end) {
    size_t len = end - start + 1;
    if (len == 0) return T(); // return default-constructed T if empty

    size_t mid = start + len / 2;
    if (len & 1)
        return v[mid];
    else
        return static_cast<T>((v[mid - 1] + v[mid]) / 2.0);
};

// Utility to check NaN
// For float/double types
template<typename T>
typename enable_if<is_floating_point<T>::value, bool>::type
isMissing(const T& value) {
    return isnan(value);
}

// For int types
template<typename T>
typename enable_if<is_integral<T>::value, bool>::type
isMissing(const T& value) {
    return value == numeric_limits<T>::min();
}

// For string type
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

        if (std_dev == 0) continue;   // skip division by zero 
        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>((row[col] - mean) / std_dev);
    }
}

// Normalize (min-max scaling to [0, 1])
template<typename T>
void normalize(Dataset<T>& data, const vector<size_t>& columns = {}) {
    static_assert(is_arithmetic<T>::value, "Normalize only works with numeric types.");

    size_t n_cols = data[0].size();
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
        
        if (min_val == max_val) continue; // Skip normalization for flat columns
        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>(
                    (static_cast<double>(row[col]) - static_cast<double>(min_val)) /
                    (static_cast<double>(max_val) - static_cast<double>(min_val))
                );
    }
}

// Find and print missing values
template<typename T>
void findMissingValues(const Dataset<T>& data) {
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            if (isMissing(data[i][j]))
                cout << "Missing at Data Point: " << i << ", Attribute: " << j << endl;
}

// Replace missing values with mean/median/mode
enum class ImputeStrategy { MEAN, MEDIAN, MODE };

template<typename T>
void replaceMissingValues(Dataset<T>& data, ImputeStrategy strategy, const vector<size_t>& columns = {}) {
    size_t n_cols = data[0].size();
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

// Describe dataset: min, max, mean, median, 25%, 75%
template<typename T>
void describeDataset(const Dataset<T>& data) {
    static_assert(is_arithmetic<T>::value, "Describe only works with numeric types.");

    size_t n_cols = data[0].size();

    // Header
    cout << left << setw(10) << "Column" 
         << setw(15) << "Mean" 
         << setw(10) << "Min" 
         << setw(15) << "25%" 
         << setw(15) << "Median" 
         << setw(15) << "75%" 
         << setw(10) << "Max" 
         << endl;
    cout << string(80, '-') << endl;

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

        size_t n = col_vals.size();

        double median = get_median<double>(col_vals, 0, n - 1);
        size_t mid = n / 2;
        double q1 = get_median<double>(col_vals, 0, mid - 1);
        double q3 = (n & 1) ? get_median<double>(col_vals, mid+1, n - 1)
                                : get_median<double>(col_vals, mid, n - 1);


        cout << left << setw(10) << col
             << setw(15) << fixed << setprecision(2) << mean
             << setw(10) << fixed << setprecision(2) << min_val
             << setw(15) << fixed << setprecision(2) << q1
             << setw(15) << fixed << setprecision(2) << median
             << setw(15) << fixed << setprecision(2) << q3
             << setw(10) << fixed << setprecision(2) << max_val
             << endl << endl;
    }
}

#endif // PREPROCESSING_H
