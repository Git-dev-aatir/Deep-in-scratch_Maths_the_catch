#include "../../include/Preprocessing/dataset_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <random>
#include <cctype>
#include <locale>
#include <utility>
#include <type_traits>

template<typename T>
Dataset<T> loadDataset(const std::string &filename, char delimiter, 
                       bool has_header, bool multiple_spaces) {
    std::ifstream file(filename);
    Dataset<T> dataset;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    if (has_header && getline(file, line)) {
        // Skip header line
    }

    while (getline(file, line)) {
        trim(line);
        if (line.empty()) continue;
        std::vector<std::string> tokens = split(line, delimiter, multiple_spaces);
        DataRow<T> row;

        for (const auto &token : tokens) {
            row.push_back(parseToken<T>(token));
        }

        dataset.push_back(row);
    }

    file.close();
    return dataset;
}

template<typename T>
void saveDatasetToCSV(const Dataset<T> &dataset, const std::string &outputFilename) {
    std::ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << outputFilename << std::endl;
        return;
    }

    for (const auto &row : dataset) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i != row.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }

    outFile.close();
    std::cout << "Dataset saved to " << outputFilename << std::endl;
}

template<typename T>
void head(const Dataset<T> &dataset, int n_rows) {
    if (dataset.empty()) {
        std::cout << "Dataset is empty!" << std::endl;
        return;
    }

    size_t n_cols = dataset[0].size();

    std::cout << std::endl << std::string(n_cols * 12, '-') << std::endl;
    std::cout << std::left;
    for (size_t col = 0; col < n_cols; ++col) {
        std::cout << std::setw(12) << ("Col " + std::to_string(col));
    }
    std::cout << std::endl;

    std::cout << std::string(n_cols * 12, '-') << std::endl;

    bool flag = 0;
    for (const auto &row : dataset) {
        for (const auto &val : row) {
            std::cout << std::setw(12) << val;
        }
        std::cout << std::endl;
        --n_rows;
        if (n_rows <= 0) break;
        else std::cout << std::endl;
    }
    std::cout << std::string(n_cols * 12, '-') << std::endl << std::endl;
}

template<typename T>
void printDimensions(const Dataset<T>& data) {
    size_t rows = data.size();
    size_t cols = rows > 0 ? data[0].size() : 0;
    std::cout << "Dimensions: [" << rows << " x " << cols << "]" << std::endl;
}

template<typename T>
void describeDataset(const Dataset<T>& data) {
    static_assert(std::is_arithmetic<T>::value, "Describe only works with numeric types.");

    if (data.empty()) {
        std::cout << "Dataset is empty, cannot describe." << std::endl;
        return;
    }

    size_t n_cols = data[0].size();

    // Header
    std::cout << std::endl << std::string(102, '-') << std::endl;
    std::cout << std::left << std::setw(10) << "Column" 
         << std::setw(15) << "Mean"
         << std::setw(15) << "StdDev"  
         << std::setw(10) << "Min" 
         << std::setw(15) << "25%" 
         << std::setw(15) << "Median" 
         << std::setw(15) << "75%" 
         << std::setw(10) << "Max" 
         << std::endl;
    std::cout << std::string(102, '-') << std::endl;

    for (size_t col = 0; col < n_cols; ++col) {
        std::vector<T> col_vals;
        for (const auto& row : data)
            if (!isMissing(row[col]))
                col_vals.push_back(row[col]);

        if (col_vals.empty()) {
            std::cout << std::setw(10) << col << "No valid data" << std::endl;
            continue;
        }

        std::sort(col_vals.begin(), col_vals.end());
        double min_val = static_cast<double>(col_vals.front());
        double max_val = static_cast<double>(col_vals.back());
        double mean = std::accumulate(col_vals.begin(), col_vals.end(), 0.0) / col_vals.size();

        // Computing Standard Deviation
        size_t n = col_vals.size();
        double sq_sum = 0.0;
        for (const auto& val : col_vals) {
            double diff = val - mean;
            sq_sum += diff * diff;
        }
        double stddev = sqrt(sq_sum / n);

        double median = get_median(col_vals, 0, n - 1);
        size_t mid = n / 2;
        double q1 = get_median(col_vals, 0, mid - 1);
        double q3 = (n & 1) ? get_median(col_vals, mid+1, n - 1)
                                : get_median(col_vals, mid, n - 1);

        if (col != 0) std::cout << std::endl;
        std::cout << std::left << std::setw(10) << col
             << std::setw(15) << std::fixed << std::setprecision(2) << mean
             << std::setw(15) << std::fixed << std::setprecision(2) << stddev
             << std::setw(10) << std::fixed << std::setprecision(2) << min_val
             << std::setw(15) << std::fixed << std::setprecision(2) << q1
             << std::setw(15) << std::fixed << std::setprecision(2) << median
             << std::setw(15) << std::fixed << std::setprecision(2) << q3
             << std::setw(10) << std::fixed << std::setprecision(2) << max_val
             << std::endl;
    }
    std::cout << std::string(102, '-') << std::endl << std::endl;
}

template<typename T>
std::pair <Dataset<T>, Dataset<T>> splitFeaturesAndLabels(const Dataset<T> &dataset) {
    Dataset<T> features;
    Dataset<T> labels;
    for (const auto &row : dataset) {
        if (row.size() < 2) {
            std::cerr << "Warning: Row with insufficient data encountered. Skipping." << std::endl;
            continue;
        }

        DataRow<T> featureRow(row.begin(), row.end() - 1);
        features.push_back(featureRow);
        labels.push_back(DataRow<T>{row.back()});
    }
    return {features, labels};
}

template<typename T>
std::pair <Dataset<T>, Dataset<T>> trainTestSplit(Dataset<T> dataset,
                                            double testFraction, 
                                            bool shuffle) {

    if (testFraction < 0.0 || testFraction > 1.0) {
        std::cerr << "Warning: testFraction should be in [0,1]. Using default 0.2." << std::endl;
        testFraction = 0.2;
    }

    // Get the number of rows in the dataset
    size_t n = dataset.size();

    // Generate shuffled indices if needed
    std::vector<size_t> indices = getIndices(n, shuffle);

    // Calculate the index at which to split
    size_t testSize = static_cast<size_t>(n * testFraction);
    size_t trainSize = n - testSize;

    // Create the training and testing sets
    Dataset<T> trainSet, testSet;

    for (size_t i = 0; i < trainSize; ++i) {
        trainSet.push_back(dataset[indices[i]]);
    }

    for (size_t i = trainSize; i < n; ++i) {
        testSet.push_back(dataset[indices[i]]);
    }

    return {trainSet, testSet};
}

std::vector<size_t> getIndices(const size_t& n, bool shuffle) {
    std::vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, 2, ..., n-1

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    return indices;
}

template<typename T>
Dataset<T> selectRowsByIndices(const Dataset<T>& dataset, const std::vector<size_t>& indices) {
    Dataset<T> selectedRows;

    for (size_t idx : indices) {
        if (idx < dataset.size()) {
            selectedRows.push_back(dataset[idx]);
        } else {
            std::cerr << "Warning: Index out of bounds: " << idx << std::endl;
        }
    }

    return selectedRows;
}

template<typename T>
void saveDatasetToBinary(const Dataset<T>& dataset, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error opening binary file for writing: " << filename << std::endl;
        return;
    }

    size_t rowCount = dataset.size();
    size_t colCount = rowCount > 0 ? dataset[0].size() : 0;

    outFile.write(reinterpret_cast<char*>(&rowCount), sizeof(rowCount));
    outFile.write(reinterpret_cast<char*>(&colCount), sizeof(colCount));

    for (const auto &row : dataset) {
        outFile.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));
    }

    outFile.close();
}

template<typename T>
Dataset<T> loadDatasetFromBinary(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    Dataset<T> dataset;

    if (!inFile.is_open()) {
        std::cerr << "Error opening binary file for reading: " << filename << std::endl;
        return dataset;
    }

    size_t rowCount, colCount;
    inFile.read(reinterpret_cast<char*>(&rowCount), sizeof(rowCount));
    inFile.read(reinterpret_cast<char*>(&colCount), sizeof(colCount));

    for (size_t i = 0; i < rowCount; ++i) {
        DataRow<T> row(colCount);
        inFile.read(reinterpret_cast<char*>(row.data()), colCount * sizeof(T));
        dataset.push_back(row);
    }

    inFile.close();
    return dataset;
}

template<>
void saveDatasetToBinary<std::string>(const Dataset<std::string>& dataset, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error opening binary file for writing: " << filename << std::endl;
        return;
    }

    size_t rowCount = dataset.size();
    size_t colCount = rowCount > 0 ? dataset[0].size() : 0;

    outFile.write(reinterpret_cast<char*>(&rowCount), sizeof(rowCount));
    outFile.write(reinterpret_cast<char*>(&colCount), sizeof(colCount));

    for (const auto &row : dataset) {
        for (const auto &str : row) {
            size_t len = str.size();
            outFile.write(reinterpret_cast<char*>(&len), sizeof(len));
            outFile.write(str.c_str(), len);
        }
    }

    outFile.close();
}

template<>
Dataset<std::string> loadDatasetFromBinary<std::string>(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    Dataset<std::string> dataset;

    if (!inFile.is_open()) {
        std::cerr << "Error opening binary file for reading: " << filename << std::endl;
        return dataset;
    }

    size_t rowCount, colCount;
    inFile.read(reinterpret_cast<char*>(&rowCount), sizeof(rowCount));
    inFile.read(reinterpret_cast<char*>(&colCount), sizeof(colCount));

    for (size_t i = 0; i < rowCount; ++i) {
        DataRow<std::string> row(colCount);
        for (size_t j = 0; j < colCount; ++j) {
            size_t len;
            inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
            row[j].resize(len);
            inFile.read(&row[j][0], len);
        }
        dataset.push_back(row);
    }

    inFile.close();
    return dataset;
}

template <typename T>
std::vector<T> squeeze(const std::vector<std::vector<T>>& matrix) {
    std::vector<T> flattened;

    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    return flattened;
}

template <typename T>
std::vector<std::vector<T>> unsqueeze(const std::vector<T>& vector, int axis) {
    if (axis != 0 && axis != 1) {
        throw std::invalid_argument("unsqueeze: axis must be 0 or 1");
    }

    std::vector<std::vector<T>> result;

    if (axis == 0) {
        // Unsqueeze along rows (vertical stacking)
        result.push_back(vector);
    } else {
        // Unsqueeze along columns (horizontal stacking)
        for (const auto& val : vector) {
            result.push_back(std::vector<T>{val});
        }
    }

    return result;
}

template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix) {
    if (matrix.empty()) return {};

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[j][i] = matrix[i][j];

    return result;
}

template<typename T>
std::vector<std::vector<T>> reshape(const std::vector<std::vector<T>>& matrix, 
                                    size_t newRows, size_t newCols) {
    std::vector<T> flat = squeeze(matrix);
    if (flat.size() != newRows * newCols) {
        throw std::runtime_error("Reshape dimensions do not match total size");
    }

    std::vector<std::vector<T>> reshaped(newRows, std::vector<T>(newCols));
    for (size_t i = 0; i < newRows; ++i)
        for (size_t j = 0; j < newCols; ++j)
            reshaped[i][j] = flat[i * newCols + j];

    return reshaped;
}



// -------------------------------------------------------------------------------------

// Explicit instantiations for int, double, and std::string

// loadDataset
template Dataset<int> loadDataset<int>(const std::string &filename, char delimiter, bool multiple_spaces, bool has_header);
template Dataset<double> loadDataset<double>(const std::string &filename, char delimiter, bool multiple_spaces, bool has_header);
template Dataset<std::string> loadDataset<std::string>(const std::string &filename, char delimiter, bool multiple_spaces, bool has_header);

// saveDatasetToCSV
template void saveDatasetToCSV<int>(const Dataset<int> &dataset, const std::string &outputFilename);
template void saveDatasetToCSV<double>(const Dataset<double> &dataset, const std::string &outputFilename);
template void saveDatasetToCSV<std::string>(const Dataset<std::string> &dataset, const std::string &outputFilename);

// head
template void head<int>(const Dataset<int> &dataset, int n_rows);
template void head<double>(const Dataset<double> &dataset, int n_rows);
template void head<std::string>(const Dataset<std::string> &dataset, int n_rows);

// printDimensions
template void printDimensions<int>(const Dataset<int>& data);
template void printDimensions<double>(const Dataset<double>& data);
template void printDimensions<std::string>(const Dataset<std::string>& data);

// describeDataset
template void describeDataset<int>(const Dataset<int>& data);
template void describeDataset<double>(const Dataset<double>& data);
// describeDataset<std::string> not instantiated because static_assert restricts to numeric types

// splitFeaturesAndLabels
template std::pair<Dataset<int>, Dataset<int>> splitFeaturesAndLabels<int>(const Dataset<int> &dataset);
template std::pair<Dataset<double>, Dataset<double>> splitFeaturesAndLabels<double>(const Dataset<double> &dataset);
template std::pair<Dataset<std::string>, Dataset<std::string>> splitFeaturesAndLabels<std::string>(const Dataset<std::string> &dataset);

// trainTestSplit
template std::pair<Dataset<int>, Dataset<int>> trainTestSplit<int>(Dataset<int> dataset, double testFraction, bool shuffle);
template std::pair<Dataset<double>, Dataset<double>> trainTestSplit<double>(Dataset<double> dataset, double testFraction, bool shuffle);
template std::pair<Dataset<std::string>, Dataset<std::string>> trainTestSplit<std::string>(Dataset<std::string> dataset, double testFraction, bool shuffle);

// selectRowsByIndices
template Dataset<int> selectRowsByIndices<int>(const Dataset<int>& dataset, const std::vector<size_t>& indices);
template Dataset<double> selectRowsByIndices<double>(const Dataset<double>& dataset, const std::vector<size_t>& indices);
template Dataset<std::string> selectRowsByIndices<std::string>(const Dataset<std::string>& dataset, const std::vector<size_t>& indices);

// saveDatasetToBinary
template void saveDatasetToBinary<int>(const Dataset<int>& dataset, const std::string& filename);
template void saveDatasetToBinary<double>(const Dataset<double>& dataset, const std::string& filename);
template void saveDatasetToBinary<std::string>(const Dataset<std::string>& dataset, const std::string& filename);

// loadDatasetFromBinary
template Dataset<int> loadDatasetFromBinary<int>(const std::string& filename);
template Dataset<double> loadDatasetFromBinary<double>(const std::string& filename);
template Dataset<std::string> loadDatasetFromBinary<std::string>(const std::string& filename);

// squeeze
template std::vector<int> squeeze<int>(const std::vector<std::vector<int>>& matrix);
template std::vector<double> squeeze<double>(const std::vector<std::vector<double>>& matrix);
template std::vector<std::string> squeeze<std::string>(const std::vector<std::vector<std::string>>& matrix);

// unsqueeze
template std::vector<std::vector<int>> unsqueeze<int>(const std::vector<int>& vector, int axis);
template std::vector<std::vector<double>> unsqueeze<double>(const std::vector<double>& vector, int axis);
template std::vector<std::vector<std::string>> unsqueeze<std::string>(const std::vector<std::string>& vector, int axis);

// transpose
template std::vector<std::vector<int>> transpose<int>(const std::vector<std::vector<int>>& matrix);
template std::vector<std::vector<double>> transpose<double>(const std::vector<std::vector<double>>& matrix);
template std::vector<std::vector<std::string>> transpose<std::string>(const std::vector<std::vector<std::string>>& matrix);

// reshape
template std::vector<std::vector<int>> reshape<int>(const std::vector<std::vector<int>>& matrix, size_t newRows, size_t newCols);
template std::vector<std::vector<double>> reshape<double>(const std::vector<std::vector<double>>& matrix, size_t newRows, size_t newCols);
template std::vector<std::vector<std::string>> reshape<std::string>(const std::vector<std::vector<std::string>>& matrix, size_t newRows, size_t newCols);
