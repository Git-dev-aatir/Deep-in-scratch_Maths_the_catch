#include "../../include/Preprocessing/Preprocessing.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <type_traits>
#include <unordered_map>

using namespace std;

template<typename T>
void standardize(Dataset<T>& data, const std::vector<size_t>& columns) {
    static_assert(is_arithmetic<T>::value, "Standardize only works with numeric types.");
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    vector<size_t> target_cols = columns.empty() ? vector<size_t>(n_cols) : columns;
    if (columns.empty()) iota(target_cols.begin(), target_cols.end(), 0);

    for (size_t col : target_cols) {
        vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col])) col_vals.push_back(row[col]);
        if (col_vals.empty()) continue;

        double mean = accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();
        double sq_sum = inner_product(col_vals.begin(), col_vals.end(), col_vals.begin(), 0.0);
        double std_dev = sqrt(sq_sum / col_vals.size() - mean * mean);
        if (std_dev == 0) continue;

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>((row[col] - mean) / std_dev);
    }
}

template<typename T>
void normalize(Dataset<T>& data, const vector<size_t>& columns) {
    static_assert(is_arithmetic<T>::value, "Normalize only works with numeric types.");
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    vector<size_t> target_cols = columns.empty() ? vector<size_t>(n_cols) : columns;
    if (columns.empty()) iota(target_cols.begin(), target_cols.end(), 0);

    for (size_t col : target_cols) {
        T min_val = numeric_limits<T>::max();
        T max_val = numeric_limits<T>::lowest();
        for (const auto& row : data)
            if (!isMissing(row[col])) {
                min_val = min(min_val, row[col]);
                max_val = max(max_val, row[col]);
            }
        if (min_val == max_val) continue;

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = static_cast<T>(
                    (static_cast<double>(row[col]) - static_cast<double>(min_val)) /
                    (static_cast<double>(max_val) - static_cast<double>(min_val))
                );
    }
}

template<typename T>
void findMissingValues(const Dataset<T>& data) {
    bool flag = false;
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            if (isMissing(data[i][j])) {
                cout << "Missing at Data Point: " << i << ", Attribute: " << j << endl;
                flag = true;
            }
    if (!flag) cout << "No Missing Values !\n";
}

template<typename T>
void removeRowsWithMissingValues(Dataset<T>& data) {
    data.erase(remove_if(data.begin(), data.end(), [](const DataRow<T>& row) {
        return any_of(row.begin(), row.end(), [](const T& val) { return isMissing(val); });
    }), data.end());
}

template<typename T>
void replaceMissingValues(Dataset<T>& data, ImputeStrategy strategy, const vector<size_t>& columns) {
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    vector<size_t> target_cols = columns.empty() ? vector<size_t>(n_cols) : columns;
    if (columns.empty()) iota(target_cols.begin(), target_cols.end(), 0);

    for (size_t col : target_cols) {
        vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col])) col_vals.push_back(row[col]);
        if (col_vals.empty()) continue;

        T replacement;
        switch (strategy) {
            case ImputeStrategy::MEAN:
                replacement = static_cast<T>(accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size());
                break;
            case ImputeStrategy::MEDIAN:
                sort(col_vals.begin(), col_vals.end());
                replacement = get_median<T>(col_vals, 0, col_vals.size() - 1);
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
            if (isMissing(row[col])) row[col] = replacement;
    }
}

template<typename T>
void replaceMissingWithCustomValue(Dataset<T>& data, const T& custom_value, const vector<size_t>& columns) {
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    vector<size_t> target_cols = columns.empty() ? vector<size_t>(n_cols) : columns;
    if (columns.empty()) iota(target_cols.begin(), target_cols.end(), 0);

    for (auto& row : data)
        for (size_t col : target_cols)
            if (isMissing(row[col])) row[col] = custom_value;
}

template<typename T>
void removeOutliers(Dataset<T>& data, OutlierMethod method, double threshold, const vector<size_t>& columns) {
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    vector<size_t> target_cols = columns.empty() ? vector<size_t>(n_cols) : columns;
    if (columns.empty()) iota(target_cols.begin(), target_cols.end(), 0);

    vector<bool> to_remove(data.size(), false);
    for (size_t col : target_cols) {
        vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col])) col_vals.push_back(row[col]);
        if (col_vals.size() < 2) continue;

        if (method == OutlierMethod::Z_SCORE) {
            double mean = accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();
            double sq_sum = inner_product(col_vals.begin(), col_vals.end(), col_vals.begin(), 0.0);
            double std_dev = sqrt(sq_sum / col_vals.size() - mean * mean);
            if (std_dev == 0) continue;

            for (size_t i = 0; i < data.size(); ++i)
                if (!isMissing(data[i][col])) {
                    double z = (data[i][col] - mean) / std_dev;
                    if (abs(z) > threshold) to_remove[i] = true;
                }
        } else if (method == OutlierMethod::IQR) {
            sort(col_vals.begin(), col_vals.end());
            size_t n = col_vals.size();
            double q1 = get_median(col_vals, 0, n / 2 - 1);
            double q3 = get_median(col_vals, (n + 1) / 2, n - 1);
            double iqr = q3 - q1;
            T lower = static_cast<T>(q1 - threshold * iqr);
            T upper = static_cast<T>(q3 + threshold * iqr);

            for (size_t i = 0; i < data.size(); ++i)
                if (!isMissing(data[i][col]) && (data[i][col] < lower || data[i][col] > upper)) to_remove[i] = true;
        }
    }

    Dataset<T> filtered;
    for (size_t i = 0; i < data.size(); ++i)
        if (!to_remove[i]) filtered.push_back(data[i]);

    data = filtered;
}

template<typename T>
void removeColumns(Dataset<T>& dataset, const vector<size_t>& columns_to_remove) {
    if (dataset.empty() || columns_to_remove.empty()) return;
    set<size_t> columnsSet(columns_to_remove.begin(), columns_to_remove.end());

    for (auto& row : dataset) {
        DataRow<T> newRow;
        for (size_t i = 0; i < row.size(); ++i)
            if (columnsSet.find(i) == columnsSet.end()) newRow.push_back(row[i]);
        row = newRow;
    }
}

// Explicit template instantiation for int, double
template void standardize<int>(Dataset<int>&, const vector<size_t>&);
template void standardize<double>(Dataset<double>&, const vector<size_t>&);
template void normalize<int>(Dataset<int>&, const vector<size_t>&);
template void normalize<double>(Dataset<double>&, const vector<size_t>&);
template void findMissingValues<int>(const Dataset<int>&);
template void findMissingValues<double>(const Dataset<double>&);
template void removeRowsWithMissingValues<int>(Dataset<int>&);
template void removeRowsWithMissingValues<double>(Dataset<double>&);
template void replaceMissingValues<int>(Dataset<int>&, ImputeStrategy, const vector<size_t>&);
template void replaceMissingValues<double>(Dataset<double>&, ImputeStrategy, const vector<size_t>&);
template void replaceMissingWithCustomValue<int>(Dataset<int>&, const int&, const vector<size_t>&);
template void replaceMissingWithCustomValue<double>(Dataset<double>&, const double&, const vector<size_t>&);
template void removeOutliers<int>(Dataset<int>&, OutlierMethod, double, const vector<size_t>&);
template void removeOutliers<double>(Dataset<double>&, OutlierMethod, double, const vector<size_t>&);
template void removeColumns<int>(Dataset<int>&, const vector<size_t>&);
template void removeColumns<double>(Dataset<double>&, const vector<size_t>&);
