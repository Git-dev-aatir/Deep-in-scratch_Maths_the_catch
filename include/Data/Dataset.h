#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <utility>

class Dataset {
private:
    std::vector<std::vector<double>> data;
    size_t num_rows = 0;
    size_t num_cols = 0;

    // Helper functions
    std::vector<double> parseCSVLine(const std::string& line, char delimiter, bool multiple_spaces);
    void validateDimensions();
    double computePercentile(const std::vector<double>& sorted_data, double percentile) const;

public:
    // Constructors
    Dataset() = default;
    explicit Dataset(const std::vector<std::vector<double>>& data);
    explicit Dataset(std::vector<std::vector<double>>&& data);

    // Loading methods
    void loadCSV(const std::string& filename, 
                 char delimiter = ',', 
                 bool has_header = false,
                 bool multiple_spaces = false);
                
    void loadBinary(const std::string& filename);

    // Saving methods
    void saveCSV(const std::string& filename, char delimiter = ',') const;
    void saveBinary(const std::string& filename) const;

    // Data inspection
    void head(size_t n_rows = 5) const;
    std::pair<size_t, size_t> shape() const;
    void describe() const;

    // Data manipulation
    std::pair<Dataset, Dataset> splitFeaturesLabels() const;
    Dataset selectRows(const std::vector<size_t>& indices) const;
    std::pair<Dataset, Dataset> trainTestSplit(double test_fraction,
                                               int stratify = -1, 
                                               bool shuffle = false
    ) const;

    // Transformation
    Dataset transpose() const;
    Dataset reshape(size_t new_rows, size_t new_cols) const;
    std::vector<double> flatten() const;
    /**
     * @brief Converts integer label dataset to one-hot encoded format
     * @throws std::runtime_error if dataset is not single-column
     */ 
    void toOneHot();

    // Accessors
    const std::vector<std::vector<double>>& getData() const;
    size_t rows() const;
    size_t cols() const;
    
    // Row access
    const std::vector<double>& operator[](size_t index) const;
    std::vector<double>& operator[](size_t index);
};
