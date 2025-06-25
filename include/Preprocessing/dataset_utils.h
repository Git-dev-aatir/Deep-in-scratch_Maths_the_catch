#pragma once

#include "Helper_functions.h"

/**
 * @brief Type alias for a row in the dataset.
 * @tparam T The data type of the elements in the row.
 */
template<typename T>
using DataRow = std::vector<T>;

/**
 * @brief Type alias for a dataset, which is a vector of DataRow.
 * @tparam T The data type of the elements in the dataset.
 */
template<typename T>
using Dataset = std::vector<DataRow<T>>;

/**
 * @brief Loads a dataset from a file.
 * 
 * Reads the file line by line, optionally skipping the header line,
 * splits each line by the given delimiter (with optional multiple space handling),
 * and parses each token into type T.
 * 
 * @tparam T Data type of the dataset elements.
 * @param filename Path to the input file.
 * @param delimiter Character used to separate values in the file (default ',').
 * @param has_header If true, skips the first line as header (default false).
 * @param multiple_spaces If true and delimiter is space, treat multiple spaces as one delimiter.
 * @return Dataset<T> Loaded dataset.
 */
template<typename T>
Dataset<T> loadDataset(const std::string &filename, char delimiter = ',', 
                       bool has_header=false, bool multiple_spaces=false);

/**
 * @brief Saves a dataset to a CSV file.
 * 
 * Each row is written as a comma-separated line.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Dataset to save.
 * @param outputFilename Path of the CSV file to write.
 */
template<typename T>
void saveDatasetToCSV(const Dataset<T> &dataset, const std::string &outputFilename);

/**
 * @brief Prints the first n rows of the dataset in a formatted table.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Dataset to display.
 * @param n_rows Number of rows to print (default 5).
 */
template<typename T>
void head(const Dataset<T> &dataset, int n_rows = 5);

/**
 * @brief Prints the dimensions of the dataset (rows x columns).
 * 
 * @tparam T Data type of the dataset elements.
 * @param data Dataset whose dimensions are to be printed.
 */
template<typename T>
void printDimensions(const Dataset<T>& data);

/**
 * @brief Displays descriptive statistics of numeric dataset columns.
 * 
 * Calculates mean, standard deviation, min, 25th percentile (Q1), median,
 * 75th percentile (Q3), and max values for each column.
 * Requires that T is an arithmetic type.
 * 
 * @tparam T Numeric data type of the dataset elements.
 * @param data Dataset to describe.
 */
template<typename T>
void describeDataset(const Dataset<T>& data);

/**
 * @brief Splits dataset into features and labels.
 * 
 * Assumes last column is the label; all other columns are features.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Original dataset to split.
 * @return std::pair<Dataset<T>, Dataset<T>> Pair of feature dataset and label dataset.
 */
template<typename T>
std::pair<Dataset<T>, Dataset<T>> splitFeaturesAndLabels(const Dataset<T> &dataset);

/**
 * @brief Generates a vector of indices from 0 to n-1.
 * 
 * Optionally shuffles the indices.
 * 
 * @param n Number of indices to generate.
 * @param shuffle If true, shuffle the indices (default true).
 * @return std::vector<size_t> Generated indices.
 */
std::vector<size_t> getIndices(const size_t& n, bool shuffle = true);

/**
 * @brief Selects rows from the dataset by specified indices.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Dataset from which to select rows.
 * @param indices Vector of row indices to select.
 * @return Dataset<T> Subset dataset containing selected rows.
 */
template<typename T>
Dataset<T> selectRowsByIndices(const Dataset<T>& dataset, const std::vector<size_t>& indices);

/**
 * @brief Splits dataset into training and testing subsets.
 * 
 * Shuffles the dataset if requested, then splits according to testFraction.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Dataset to split.
 * @param testFraction Fraction of data to use as test set (default 0.2).
 * @param shuffle Whether to shuffle the dataset before splitting (default true).
 * @return std::pair<Dataset<T>, Dataset<T>> Pair of training and testing datasets.
 */
template<typename T>
std::pair<Dataset<T>, Dataset<T>> trainTestSplit(Dataset<T> dataset,
                                                double testFraction = 0.2,
                                                bool shuffle = true);

/**
 * @brief Saves a dataset to a binary file.
 * 
 * Works for numeric types like int, double.
 * 
 * @tparam T Data type of the dataset elements.
 * @param dataset Dataset to save.
 * @param filename Path to the binary output file.
 */
template<typename T>
void saveDatasetToBinary(const Dataset<T>& dataset, const std::string& filename);

/**
 * @brief Loads a dataset from a binary file.
 * 
 * Works for numeric types like int, double.
 * 
 * @tparam T Data type of the dataset elements.
 * @param filename Path to the binary input file.
 * @return Dataset<T> Loaded dataset.
 */
template<typename T>
Dataset<T> loadDatasetFromBinary(const std::string& filename);

/**
 * @brief Specialization: Saves a dataset of strings to a binary file.
 * 
 * Strings are saved by writing the length followed by characters.
 * 
 * @param dataset Dataset of strings to save.
 * @param filename Path to the binary output file.
 */
template<>
inline void saveDatasetToBinary<std::string>(const Dataset<std::string>& dataset, 
                                             const std::string& filename);

/**
 * @brief Specialization: Loads a dataset of strings from a binary file.
 * 
 * Reads length-prefixed strings row by row.
 * 
 * @param filename Path to the binary input file.
 * @return Dataset<std::string> Loaded string dataset.
 */
template<>
inline Dataset<std::string> loadDatasetFromBinary<std::string>(const std::string& filename);

/**
 * @brief Flattens a 2D matrix (vector of vectors) into a 1D vector.
 * 
 * @tparam T Data type of the elements.
 * @param matrix 2D vector (matrix) to flatten.
 * @return std::vector<T> Flattened 1D vector.
 */
template <typename T>
std::vector<T> squeeze(const std::vector<std::vector<T>>& matrix);

/**
 * @brief Adds a new dimension to a 1D vector, converting it to a 2D vector along a specified axis.
 * 
 * Axis 0: result is a single row (1 x N).
 * Axis 1: result is a single column (N x 1).
 * 
 * @tparam T Data type of the elements.
 * @param vector 1D vector to unsqueeze.
 * @param axis Axis along which to insert the new dimension (default 1).
 * @return std::vector<std::vector<T>> Resulting 2D vector.
 * @throws std::invalid_argument if axis is not 0 or 1.
 */
template <typename T>
std::vector<std::vector<T>> unsqueeze(const std::vector<T>& vector, int axis=1);

/**
 * @brief Transposes a 2D matrix (swaps rows and columns).
 * 
 * @tparam T Data type of the elements.
 * @param matrix 2D vector (matrix) to transpose.
 * @return std::vector<std::vector<T>> Transposed matrix.
 */
template<typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix);

/**
 * @brief Reshapes a 2D matrix to new dimensions.
 * 
 * The total number of elements must remain the same.
 * 
 * @tparam T Data type of the elements.
 * @param matrix 2D vector (matrix) to reshape.
 * @param newRows Number of rows in the reshaped matrix.
 * @param newCols Number of columns in the reshaped matrix.
 * @return std::vector<std::vector<T>> Reshaped matrix.
 * @throws std::runtime_error if total size does not match.
 */
template<typename T>
std::vector<std::vector<T>> reshape(const std::vector<std::vector<T>>& matrix, 
                                    size_t newRows, size_t newCols);
