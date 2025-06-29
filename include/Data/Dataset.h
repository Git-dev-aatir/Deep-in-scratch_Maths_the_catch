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

/**
 * @class Dataset
 * @brief Core data container for neural network operations
 * 
 * Handles dataset loading, manipulation, inspection, and transformation.
 * Supports both CSV and binary formats with configurable parsing options.
 */
class Dataset {
private:
    std::vector<std::vector<double>> data; ///< Row-major data storage
    size_t num_rows = 0;                   ///< Number of rows in dataset
    size_t num_cols = 0;                   ///< Number of columns in dataset

    // Helper functions
    std::vector<double> parseCSVLine(const std::string& line, char delimiter, bool multiple_spaces);
    void validateDimensions();
    double computePercentile(const std::vector<double>& sorted_data, double percentile) const;

public:
    // =====================
    // Construction Interface
    // =====================
    
    /**
     * @brief Default constructor
     */
    Dataset() = default;
    
    /**
     * @brief Construct from existing data (copy semantics)
     * @param data 2D vector containing dataset values
     */
    explicit Dataset(const std::vector<std::vector<double>>& data);
    
    /**
     * @brief Construct from existing data (move semantics)
     * @param data 2D vector containing dataset values
     */
    explicit Dataset(std::vector<std::vector<double>>&& data);

    // =================
    // Loading Interface
    // =================
    
    /**
     * @brief Load dataset from CSV file
     * @param filename Path to CSV file
     * @param delimiter Character separating values (default ',')
     * @param has_header Whether first row contains column names (default false)
     * @param multiple_spaces Treat consecutive spaces as single delimiter (default false)
     * @throws std::runtime_error On file open failure or dimension mismatch
     */
    void loadCSV(const std::string& filename, 
                 char delimiter = ',', 
                 bool has_header = false,
                 bool multiple_spaces = false);
                
    /**
     * @brief Load dataset from binary file
     * @param filename Path to binary file
     * @param skip_header Whether to ignore first sizeof(size_t)*2 bytes (default false)
     * @throws std::runtime_error On file open failure or read error
     */
    void loadBinary(const std::string& filename, bool skip_header = false);

    // =================
    // Saving Interface
    // =================
    
    /**
     * @brief Save dataset to CSV file
     * @param filename Output file path
     * @param delimiter Value separator (default ',')
     * @param write_header Include row/column count in first line (default true)
     */
    void saveCSV(const std::string& filename, 
                 char delimiter = ',', 
                 bool write_header = true) const;

    /**
     * @brief Save dataset to binary file
     * @param filename Output file path
     * @param write_header Include row/column count prefix (default true)
     */
    void saveBinary(const std::string& filename, bool write_header = true) const;

    // ====================
    // Inspection Interface
    // ====================
    
    /**
     * @brief Print first N rows to console
     * @param n_rows Number of rows to display (default 5)
     */
    void head(size_t n_rows = 5) const;
    
    /**
     * @brief Get dataset dimensions
     * @return Pair of (rows, columns) counts
     */
    std::pair<size_t, size_t> shape() const;
    
    /**
     * @brief Print dataset dimensions to console
     */
    void printShape() const;
    
    /**
     * @brief Display statistical summary
     * 
     * Shows for each column:
     * - Min/Max values
     * - Mean/Median
     * - Standard deviation
     * - 25th/75th percentiles
     */
    void describe() const;

    // ========================
    // Manipulation Interface
    // ========================
    
    /**
     * @brief Separate features and labels
     * @param label_col Column index containing labels (-1 for last column)
     * @return Pair of (features, labels) datasets
     * @throws std::out_of_range For invalid column index
     */
    std::pair<Dataset, Dataset> splitFeaturesLabels(int label_col = -1) const;
    
    /**
     * @brief Create subset from specific rows
     * @param indices Vector of row indices to select
     * @return New dataset containing selected rows
     * @throws std::out_of_range For invalid indices
     */
    Dataset selectRows(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Split dataset into training and test sets
     * @param test_fraction Fraction of data for testing (0.0-1.0)
     * @param stratify Column index for stratified sampling (-1 to disable)
     * @param shuffle Whether to randomize row order (default false)
     * @return Pair of (training, test) datasets
     */
    std::pair<Dataset, Dataset> trainTestSplit(double test_fraction,
                                               int stratify = -1, 
                                               bool shuffle = false) const;

    // ======================
    // Transformation Interface
    // ======================
    
    /**
     * @brief Transpose dataset (rows ↔ columns)
     * @return New transposed dataset
     */
    Dataset transpose() const;
    
    /**
     * @brief Reshape dataset dimensions
     * @param new_rows Target row count
     * @param new_cols Target column count
     * @return New reshaped dataset
     * @throws std::invalid_argument If total elements don't match
     */
    Dataset reshape(size_t new_rows, size_t new_cols) const;
    
    /**
     * @brief Convert 2D dataset to 1D vector
     * @return Flattened data in row-major order
     */
    std::vector<double> flatten() const;
    
    /**
     * @brief Convert integer labels to one-hot encoding
     * 
     * Converts single-column integer labels to multi-column one-hot representation.
     * Original label column is replaced.
     * 
     * @throws std::runtime_error If dataset has multiple columns
     * @throws std::invalid_argument If labels aren't integers
     */
    void toOneHot();

    // =================
    // Accessor Interface
    // =================
    
    /**
     * @brief Get raw data reference
     * @return Const reference to underlying 2D data
     */
    const std::vector<std::vector<double>>& getData() const;
    
    /**
     * @brief Get row count
     * @return Number of rows
     */
    size_t rows() const;
    
    /**
     * @brief Get column count
     * @return Number of columns
     */
    size_t cols() const;
    
    // =================
    // Indexing Operators
    // =================
    
    /**
     * @brief Const row access
     * @param index Row index
     * @return Const reference to row data
     */
    const std::vector<double>& operator[](size_t index) const;
    
    /**
     * @brief Mutable row access
     * @param index Row index
     * @return Mutable reference to row data
     */
    std::vector<double>& operator[](size_t index);
};
