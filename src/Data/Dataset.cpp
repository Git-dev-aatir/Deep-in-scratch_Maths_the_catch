#include "Data/Dataset.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <random>
#include <numeric>
#include <cmath>
// #include <filesystem>

// Helper: Parse CSV line with optional multi-space handling
std::vector<double> Dataset::parseCSVLine(const std::string& line, char delimiter, bool multiple_spaces) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string token;
    
    if (multiple_spaces && delimiter == ' ') {
        // Handle multiple spaces as single delimiter
        std::istringstream iss(line);
        while (iss >> token) {
            row.push_back(std::stod(token));
        }
    } else {
        // Standard delimiter parsing
        while (std::getline(ss, token, delimiter)) {
            if (token.empty()) continue;
            row.push_back(std::stod(token));
        }
    }
    return row;
}

// Validate consistent dimensions
void Dataset::validateDimensions() {
    if (data.empty()) {
        num_rows = 0;
        num_cols = 0;
        return;
    }
    
    num_rows = data.size();
    num_cols = data[0].size();
    
    for (size_t i = 1; i < num_rows; ++i) {
        if (data[i].size() != num_cols) {
            throw std::runtime_error("Inconsistent row dimensions in dataset");
        }
    }
}

// Helper function to compute the Percentiles
double Dataset::computePercentile(const std::vector<double>& sorted_data, double percentile) const {
    if (sorted_data.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    const double index = percentile/100.0 * (sorted_data.size()-1);
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = static_cast<size_t>(std::ceil(index));
    
    if (lower == upper) {
        return sorted_data[lower];
    }
    
    const double fraction = index - lower;
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower]);
}

// Constructors
Dataset::Dataset(const std::vector<std::vector<double>>& data) : data(data) {
    validateDimensions();
}

Dataset::Dataset(std::vector<std::vector<double>>&& data) : data(std::move(data)) {
    validateDimensions();
}

// CSV Loading
void Dataset::loadCSV(const std::string& filename, char delimiter, bool has_header, bool multiple_spaces) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);
    
    data.clear();
    std::string line;
    
    if (has_header) std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        data.push_back(parseCSVLine(line, delimiter, multiple_spaces));
    }
    
    validateDimensions();
}

// Binary Loading
void Dataset::loadBinary(const std::string& filename, bool skip_header) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);
    
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    
    // Adjust row count if skipping header
    size_t data_rows = rows;
    if (skip_header && rows > 0) {
        // Skip the first row (header)
        file.seekg(cols * sizeof(double), std::ios::cur);
        data_rows = rows - 1;
    }
    
    // Resize and read data
    data.resize(data_rows, std::vector<double>(cols));
    for (size_t i = 0; i < data_rows; ++i) {
        file.read(reinterpret_cast<char*>(data[i].data()), cols * sizeof(double));
    }
    
    num_rows = data_rows;
    num_cols = cols;
}

// CSV Saving
void Dataset::saveCSV(const std::string& filename, char delimiter, bool write_header) const {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Cannot create file: " + filename);
    bool first = true;
    for (const auto& row : data) {
        if (first && write_header) {
            first = false;
            continue;
        }
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << delimiter;
        }
        file << '\n';
    }
}

// Binary Saving
void Dataset::saveBinary(const std::string& filename, bool write_header) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create file: " + filename);
    
    size_t rows = data.size();
    size_t cols = rows > 0 ? data[0].size() : 0;
    
    // Determine start row and adjust row count
    size_t start_row = 0;
    if (!write_header && rows > 0) {
        start_row = 1;  // Skip first row
        rows -= 1;
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    // Write data rows
    for (size_t r = start_row; r < data.size(); ++r) {
        if (data[r].size() != cols) {
            throw std::runtime_error("Inconsistent column count in row " + std::to_string(r));
        }
        file.write(reinterpret_cast<const char*>(data[r].data()), cols * sizeof(double));
    }
}


// Data inspection
void Dataset::head(size_t n_rows) const {
    size_t display = std::min(n_rows, data.size());
    for (size_t i = 0; i < display; ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            std::cout << data[i][j];
            if (j < data[i].size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

std::pair<size_t, size_t> Dataset::shape() const {
    return {num_rows, num_cols};
}

void Dataset::printShape() const {
    std::cout << "Shape : [" << num_rows << " x " << num_cols << " ]\n";
}

void Dataset::describe() const {
    // Print header
    std::cout << "\nColumn\t\tCountNull\tCountUnique\tMean\t\tStd\t\tMin\t\t25%\t\t50%\t\t75%\t\tMax\n";
    
    for (size_t col = 0; col < num_cols; ++col) {
        std::vector<double> column_data;
        column_data.reserve(num_rows);
        
        // Extract column data and count nulls
        size_t count_null = 0;
        for (size_t row = 0; row < num_rows; ++row) {
            const double value = data[row][col];
            if (std::isnan(value)) {
                count_null++;
            } else {
                column_data.push_back(value);
            }
        }
        
        // Skip calculation if no valid data
        if (column_data.empty()) {
            std::cout << col << "\t\t" << count_null << "\t\t0\t\tnan\t\tnan\t\tnan\t\tnan\t\tnan\t\tnan\t\tnan\n";
            continue;
        }
        
        // Sort for percentiles and unique count
        std::sort(column_data.begin(), column_data.end());
        
        // Count unique values
        std::set<double> unique_set(column_data.begin(), column_data.end());
        const size_t count_unique = unique_set.size();
        
        // Calculate mean
        const double sum = std::accumulate(column_data.begin(), column_data.end(), 0.0);
        const double mean = sum / column_data.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double value : column_data) {
            variance += (value - mean) * (value - mean);
        }
        variance /= column_data.size();
        const double std_dev = std::sqrt(variance);
        
        // Calculate percentiles
        const double min_val = column_data.front();
        const double max_val = column_data.back();
        const double q1 = computePercentile(column_data, 25.0);
        const double median = computePercentile(column_data, 50.0);
        const double q3 = computePercentile(column_data, 75.0);
        
        // Format and print
        std::cout << col << "\t\t"
                  << count_null << "\t\t"
                  << count_unique << "\t\t"
                  << std::fixed << std::setprecision(4)
                  << mean << "\t\t"
                  << std_dev << "\t\t"
                  << min_val << "\t\t"
                  << q1 << "\t\t"
                  << median << "\t\t"
                  << q3 << "\t\t"
                  << max_val << "\n";
    }
    std::cout << std::endl;
}

// Data manipulation
std::pair<Dataset, Dataset> Dataset::splitFeaturesLabels(int label_col) const {
    if (data.empty()) return {Dataset(), Dataset()};

    if (label_col == -1) 
        label_col = this->num_cols - 1;
    
    size_t num_cols = data[0].size();
    if (label_col >= num_cols || label_col < 0) {
        throw std::out_of_range("Label column index out of bounds");
    }
    
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
    
    for (const auto& row : data) {
        if (row.size() != num_cols) {
            std::cout << "Skipped an inconsistent row \n";
            continue;  // Skip inconsistent rows
        }
        
        // Extract features (all columns except label)
        std::vector<double> feat_row;
        for (size_t i = 0; i < row.size(); ++i) {
            if (i != label_col) {
                feat_row.push_back(row[i]);
            }
        }
        features.push_back(std::move(feat_row));
        
        // Extract label
        labels.push_back({row[label_col]});
    }
    
    return {Dataset(features), Dataset(labels)};
}


Dataset Dataset::selectRows(const std::vector<size_t>& indices) const {
    std::vector<std::vector<double>> selected;
    for (auto idx : indices) {
        if (idx < data.size()) {
            selected.push_back(data[idx]);
        }
    }
    return Dataset(selected);
}

std::pair<Dataset, Dataset> Dataset::trainTestSplit(double test_fraction,
                                                   int stratify, 
                                                   bool shuffle) const {
    std::vector<size_t> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(std::random_device{}());

    if (stratify != -1) {
        // Validate column index
        if (stratify < 0 || stratify >= static_cast<int>(num_cols)) {
            throw std::out_of_range("Stratify column index out of bounds");
        }

        // Prepare stratification labels
        std::vector<int> labels;
        for (size_t i = 0; i < num_rows; ++i) {
            labels.push_back(static_cast<int>(data[i][stratify]));
        }

        // Group indices by class
        std::map<int, std::vector<size_t>> class_indices;
        for (size_t i = 0; i < num_rows; ++i) {
            class_indices[labels[i]].push_back(i);
        }

        std::vector<size_t> train_indices, test_indices;
        
        for (auto& [class_label, indices_in_class] : class_indices) {
            if (shuffle) {
                std::shuffle(indices_in_class.begin(), indices_in_class.end(), rng);
            }
            
            size_t class_test_size = static_cast<size_t>(indices_in_class.size() * test_fraction);
            if (class_test_size == 0) class_test_size = 1;
            
            // Add to test set
            for (size_t i = 0; i < class_test_size; ++i) {
                test_indices.push_back(indices_in_class[i]);
            }
            
            // Add to train set
            for (size_t i = class_test_size; i < indices_in_class.size(); ++i) {
                train_indices.push_back(indices_in_class[i]);
            }
        }
        
        // Shuffle final sets
        if (shuffle) {
            std::shuffle(train_indices.begin(), train_indices.end(), rng);
            std::shuffle(test_indices.begin(), test_indices.end(), rng);
        }
        
        return {selectRows(train_indices), selectRows(test_indices)};
    } 
    else {
        // Non-stratified split
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        size_t test_size = static_cast<size_t>(num_rows * test_fraction);
        std::vector<size_t> test_indices(indices.begin(), indices.begin() + test_size);
        std::vector<size_t> train_indices(indices.begin() + test_size, indices.end());
        
        return {selectRows(train_indices), selectRows(test_indices)};
    }
}

// Transformation
Dataset Dataset::transpose() const {
    if (data.empty()) return Dataset();
    
    std::vector<std::vector<double>> transposed(num_cols, std::vector<double>(num_rows));
    
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            transposed[j][i] = data[i][j];
        }
    }
    
    return Dataset(transposed);
}

void Dataset::toOneHot() {
    // Validate dataset has exactly one column
    if (num_cols != 1) {
        throw std::runtime_error("toOneHot() requires single-column dataset");
    }

    // Find max label value
    double max_label = 0.0;
    for (const auto& row : data) {
        if (row[0] > max_label) {
            max_label = row[0];
        }
    }
    size_t num_classes = static_cast<size_t>(max_label) + 1;

    // Create new one-hot encoded data
    std::vector<std::vector<double>> new_data;
    new_data.reserve(num_rows);
    
    for (const auto& row : data) {
        double label_value = row[0];
        if (label_value < 0 || std::isnan(label_value)) {
            throw std::runtime_error("Invalid label value: " + std::to_string(label_value));
        }
        
        size_t label_index = static_cast<size_t>(label_value);
        if (label_index >= num_classes) {
            throw std::runtime_error("Label index exceeds class count");
        }
        
        // Create one-hot vector
        std::vector<double> one_hot(num_classes, 0.0);
        one_hot[label_index] = 1.0;
        new_data.push_back(std::move(one_hot));
    }

    // Replace data
    data = std::move(new_data);
    num_cols = num_classes;
}


// Accessors
const std::vector<std::vector<double>>& Dataset::getData() const { 
    return data; 
}

size_t Dataset::rows() const { 
    return num_rows; 
}

size_t Dataset::cols() const { 
    return num_cols; 
}

// Row access
const std::vector<double>& Dataset::operator[](size_t index) const {
    if (index >= num_rows) throw std::out_of_range("Index out of range");
    return data[index];
}

std::vector<double>& Dataset::operator[](size_t index) {
    if (index >= num_rows) throw std::out_of_range("Index out of range");
    return data[index];
}
