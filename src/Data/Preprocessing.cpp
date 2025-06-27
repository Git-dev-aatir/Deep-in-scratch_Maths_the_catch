#include "Data/Preprocessing.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <limits>
#include <random>

namespace {

inline bool isMissing(double val) {
    return std::isnan(val);
}

double median(const std::vector<double>& vals) {
    if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> copy = vals;  // Create a copy for sorting
    std::sort(copy.begin(), copy.end());
    size_t n = copy.size();
    if (n % 2 == 1)
        return copy[n / 2];
    else
        return (copy[n / 2 - 1] + copy[n / 2]) / 2.0;
}


}

namespace Preprocessing {

void standardize(Dataset& dataset, const std::vector<size_t>& columns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    std::vector<size_t> targetCols = columns.empty() ? std::vector<size_t>(n_cols) : columns;
    if (columns.empty()) std::iota(targetCols.begin(), targetCols.end(), 0);

    for (size_t col : targetCols) {
        std::vector<double> colVals;
        for (const auto& row : data)
            if (!isMissing(row[col])) colVals.push_back(row[col]);
        if (colVals.empty()) continue;

        double mean = std::accumulate(colVals.begin(), colVals.end(), 0.0) / colVals.size();
        double sq_sum = std::inner_product(colVals.begin(), colVals.end(), colVals.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / colVals.size() - mean * mean);
        if (stddev == 0) continue;

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = (row[col] - mean) / stddev;
    }
}

void minMaxNormalize(Dataset& dataset, const std::vector<size_t>& columns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    std::vector<size_t> targetCols = columns.empty() ? std::vector<size_t>(n_cols) : columns;
    if (columns.empty()) std::iota(targetCols.begin(), targetCols.end(), 0);

    for (size_t col : targetCols) {
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::lowest();
        for (const auto& row : data)
            if (!isMissing(row[col])) {
                minVal = std::min(minVal, row[col]);
                maxVal = std::max(maxVal, row[col]);
            }
        if (minVal == maxVal) continue;

        for (auto& row : data)
            if (!isMissing(row[col]))
                row[col] = (row[col] - minVal) / (maxVal - minVal);
    }
}

void printMissingValues(const Dataset& dataset) {
    const auto& data = dataset.getData();
    bool found = false;
    for (size_t i = 0; i < data.size(); ++i)
        for (size_t j = 0; j < data[i].size(); ++j)
            if (isMissing(data[i][j])) {
                std::cout << "Missing at Row: " << i << ", Col: " << j << std::endl;
                found = true;
            }
    if (!found) std::cout << "No Missing Values!\n";
}

void dropRowsWithMissing(Dataset& dataset) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    data.erase(std::remove_if(data.begin(), data.end(), [](const std::vector<double>& row) {
        return std::any_of(row.begin(), row.end(), isMissing);
    }), data.end());
}

void imputeMissing(Dataset& dataset, ImputeStrategy strategy, const std::vector<size_t>& columns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    std::vector<size_t> targetCols = columns.empty() ? std::vector<size_t>(n_cols) : columns;
    if (columns.empty()) std::iota(targetCols.begin(), targetCols.end(), 0);

    for (size_t col : targetCols) {
        std::vector<double> colVals;
        for (const auto& row : data)
            if (!isMissing(row[col])) colVals.push_back(row[col]);
        if (colVals.empty()) continue;

        double replacement = 0.0;
        switch (strategy) {
            case ImputeStrategy::Mean:
                replacement = std::accumulate(colVals.begin(), colVals.end(), 0.0) / colVals.size();
                break;
            case ImputeStrategy::Median:
                replacement = median(colVals);
                break;
            case ImputeStrategy::Mode: {
                std::unordered_map<double, int> freq;
                for (const auto& val : colVals) freq[val]++;
                replacement = std::max_element(freq.begin(), freq.end(),
                                               [](const auto& a, const auto& b) { return a.second < b.second; })->first;
                break;
            }
        }

        for (auto& row : data)
            if (isMissing(row[col])) row[col] = replacement;
    }
}

void fillMissingWithValue(Dataset& dataset, double value, const std::vector<size_t>& columns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    std::vector<size_t> targetCols = columns.empty() ? std::vector<size_t>(n_cols) : columns;
    if (columns.empty()) std::iota(targetCols.begin(), targetCols.end(), 0);

    for (auto& row : data)
        for (size_t col : targetCols)
            if (isMissing(row[col])) row[col] = value;
}

void dropOutliers(Dataset& dataset, OutlierMethod method, double threshold, const std::vector<size_t>& columns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty()) return;
    size_t n_cols = data[0].size();
    std::vector<size_t> targetCols = columns.empty() ? std::vector<size_t>(n_cols) : columns;
    if (columns.empty()) std::iota(targetCols.begin(), targetCols.end(), 0);

    std::vector<bool> to_remove(data.size(), false);
    for (size_t col : targetCols) {
        std::vector<double> colVals;
        for (const auto& row : data)
            if (!isMissing(row[col])) colVals.push_back(row[col]);
        if (colVals.size() < 2) continue;

        if (method == OutlierMethod::ZScore) {
            double mean = std::accumulate(colVals.begin(), colVals.end(), 0.0) / colVals.size();
            double sq_sum = std::inner_product(colVals.begin(), colVals.end(), colVals.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / colVals.size() - mean * mean);
            if (stddev == 0) continue;

            for (size_t i = 0; i < data.size(); ++i)
                if (!isMissing(data[i][col])) {
                    double z = (data[i][col] - mean) / stddev;
                    if (std::abs(z) > threshold) to_remove[i] = true;
                }
        } else if (method == OutlierMethod::IQR) {
            // In dropOutliers function:
            std::sort(colVals.begin(), colVals.end());
            size_t n = colVals.size();

            // Create permanent vectors (lvalues)
            std::vector<double> firstHalf(colVals.begin(), colVals.begin() + n / 2);
            std::vector<double> secondHalf(colVals.begin() + (n + 1) / 2, colVals.end());

            double q1 = median(firstHalf);
            double q3 = median(secondHalf);
            double iqr = q3 - q1;
            double lower = q1 - threshold * iqr;
            double upper = q3 + threshold * iqr;

            for (size_t i = 0; i < data.size(); ++i)
                if (!isMissing(data[i][col]) && (data[i][col] < lower || data[i][col] > upper)) to_remove[i] = true;
        }
    }

    std::vector<std::vector<double>> filtered;
    for (size_t i = 0; i < data.size(); ++i)
        if (!to_remove[i]) filtered.push_back(data[i]);

    data = std::move(filtered);
}

void dropColumns(Dataset& dataset, const std::vector<size_t>& columnsToRemove) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty() || columnsToRemove.empty()) return;
    std::set<size_t> columnsSet(columnsToRemove.begin(), columnsToRemove.end());

    for (auto& row : data) {
        std::vector<double> newRow;
        for (size_t i = 0; i < row.size(); ++i)
            if (columnsSet.find(i) == columnsSet.end()) newRow.push_back(row[i]);
        row = std::move(newRow);
    }
}

void oneHotEncode(Dataset& dataset, const std::vector<size_t>& categoricalColumns) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    if (data.empty() || categoricalColumns.empty()) return;
    size_t rows = data.size();
    size_t cols = data[0].size();

    // Find max value in each categorical column to determine number of categories
    std::vector<size_t> maxCategories(categoricalColumns.size(), 0);
    for (size_t i = 0; i < categoricalColumns.size(); ++i) {
        size_t col = categoricalColumns[i];
        size_t max_val = 0;
        for (size_t row = 0; row < rows; ++row) {
            if (data[row][col] > max_val) max_val = static_cast<size_t>(data[row][col]);
        }
        maxCategories[i] = max_val + 1; // categories count
    }

    // New data will have expanded columns
    size_t newCols = cols;
    for (auto c : maxCategories) newCols += c - 1; // remove original cat col, add one-hot cols

    std::vector<std::vector<double>> newData(rows, std::vector<double>(newCols, 0.0));

    for (size_t row = 0; row < rows; ++row) {
        size_t new_col_idx = 0;
        for (size_t col = 0; col < cols; ++col) {
            auto it = std::find(categoricalColumns.begin(), categoricalColumns.end(), col);
            if (it != categoricalColumns.end()) {
                size_t cat_idx = std::distance(categoricalColumns.begin(), it);
                size_t cat_val = static_cast<size_t>(data[row][col]);
                for (size_t k = 0; k < maxCategories[cat_idx]; ++k) {
                    newData[row][new_col_idx++] = (k == cat_val) ? 1.0 : 0.0;
                }
            } else {
                newData[row][new_col_idx++] = data[row][col];
            }
        }
    }

    data = std::move(newData);
}

void shuffleRows(Dataset& dataset) {
    auto& data = const_cast<std::vector<std::vector<double>>&>(dataset.getData());
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}

}
