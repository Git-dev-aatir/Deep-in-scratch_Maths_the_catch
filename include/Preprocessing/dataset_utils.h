#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <random>
#include <cctype>
#include <locale>
#include <utility>

using namespace std;

template<typename T>
using DataRow = vector<T>;

template<typename T>
using Dataset = vector<DataRow<T>>;

/**
 * @brief Trim whitespace from the start of a string (in place).
 * @param s The string to trim.
 */
inline void ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(),
        [](unsigned char ch) { return !isspace(ch); }));
}

/**
 * @brief Trim whitespace from the end of a string (in place).
 * @param s The string to trim.
 */
inline void rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) { return !isspace(ch); }).base(), s.end());
}

/**
 * @brief Trim whitespace from both ends of a string (in place).
 * @param s The string to trim.
 */
inline void trim(string &s) {
    ltrim(s);
    rtrim(s);
}


/**
 * @brief Split a string into tokens based on a delimiter.
 * @param line The string to split.
 * @param delimiter The character used as delimiter.
 * @param multiple_spaces If true, treats multiple spaces as one delimiter.
 * @return A vector of string tokens.
 */
inline vector<string> split(const string &line, char delimiter, bool multiple_spaces) {
    vector<string> tokens;

    if (multiple_spaces && delimiter == ' ') {
        regex re(R"(\s+)");
        sregex_token_iterator it(line.begin(), line.end(), re, -1);
        sregex_token_iterator reg_end;

        for (; it != reg_end; ++it) {
            if (!it->str().empty()) {
                tokens.push_back(it->str());
            }
        }
    } else {
        string token;
        istringstream tokenStream(line);
        while (getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

/**
 * @brief Parse a token string to the target type.
 * @tparam T Target type.
 * @param token The string token to parse.
 * @return Parsed value of type T.
 */
template<typename T>
T parseToken(const string &token);

template<>
inline int parseToken<int>(const string &token) {
    try {
        return stoi(token);
    } catch (...) {
        cerr << "Warning: Non-int value \"" << token << "\" encountered. Storing 0." << endl;
        return 0;
    }
}

template<>
inline double parseToken<double>(const string &token) {
    try {
        return stod(token);
    } catch (...) {
        cerr << "Warning: Non-double value \"" << token << "\" encountered. Storing 0.0." << endl;
        return 0.0;
    }
}

template<>
inline string parseToken<string>(const string &token) {
    return token;
}

/**
 * @brief Load a dataset from a file.
 * @tparam T Data type of the dataset.
 * @param filename File to load data from.
 * @param delimiter Delimiter used in the file.
 * @param multiple_spaces True if multiple spaces should be treated as a delimiter.
 * @return Loaded dataset.
 */
template<typename T>
Dataset<T> loadDataset(const string &filename, char delimiter = ',', bool multiple_spaces=false) {
    ifstream file(filename);
    Dataset<T> dataset;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return dataset;
    }

    string line;
    while (getline(file, line)) {
        trim(line);
        if (line.empty()) continue;
        vector<string> tokens = split(line, delimiter, multiple_spaces);
        DataRow<T> row;

        for (const auto &token : tokens) {
            row.push_back(parseToken<T>(token));
        }

        dataset.push_back(row);
    }

    file.close();
    return dataset;
}

/**
 * @brief Save a dataset to a CSV file.
 * @tparam T Data type of the dataset.
 * @param dataset The dataset to save.
 * @param outputFilename Name of the output CSV file.
 */
template<typename T>
void saveDatasetToCSV(const Dataset<T> &dataset, const string &outputFilename) {
    ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << outputFilename << endl;
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
    cout << "Dataset saved to " << outputFilename << endl;
}

/**
 * @brief Print the first n rows of the dataset.
 * @tparam T Data type of the dataset.
 * @param dataset The dataset to display.
 * @param n_rows Number of rows to display.
 */
template<typename T>
void head(const Dataset<T> &dataset, int n_rows = 5) {
    if (dataset.empty()) {
        cout << "Dataset is empty!" << endl;
        return;
    }

    size_t n_cols = dataset[0].size();

    // Print column headers
    cout << endl << string(n_cols * 12, '-') << endl;
    cout << left;
    for (size_t col = 0; col < n_cols; ++col) {
        cout << setw(12) << ("Col " + to_string(col));
    }
    cout << endl;

    // Print separator line
    cout << string(n_cols * 12, '-') << endl;

    // Print first n_rows rows
    bool flag = 0;
    for (const auto &row : dataset) {
        for (const auto &val : row) {
            cout << setw(12) << val;
        }
        cout << endl;
        --n_rows;
        if (n_rows <= 0) break;
        else cout << endl;
    }
    cout << string(n_cols * 12, '-') << endl << endl;
}

/**
 * @brief Print the dimensions of the dataset.
 * @tparam T Data type of the dataset.
 * @param vec The dataset.
 */ 
template<typename T>
void printDimensions(const Dataset<T>& vec) {
    size_t rows = vec.size();
    size_t cols = rows > 0 ? vec[0].size() : 0;
    cout << "Dimensions: [" << rows << " x " << cols << "]" << endl;
}


/**
 * @brief Split the dataset into features and labels.
 * @tparam T Data type of the dataset.
 * @param dataset The dataset to split.
 * @return A pair containing features and labels datasets.
 */
template<typename T>
pair <Dataset<T>, Dataset<T>> splitFeaturesAndLabels(const Dataset<T> &dataset) {
    Dataset<T> features;
    Dataset<T> labels;
    for (const auto &row : dataset) {
        if (row.size() < 2) {
            cerr << "Warning: Row with insufficient data encountered. Skipping." << endl;
            continue;
        }

        DataRow<T> featureRow(row.begin(), row.end() - 1);
        features.push_back(featureRow);
        labels.push_back(DataRow<T>{row.back()});
    }
    return {features, labels};
}

/**
 * @brief Generate shuffled or ordered indices.
 * @param n Total number of indices.
 * @param shuffle If true, shuffle the indices.
 * @return Vector of indices.
 */
vector<size_t> getIndices(const size_t& n, bool shuffle = true) {
    vector<size_t> indices(n);
    // Fill indices with 0, 1, 2, ..., n-1
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        random_device rd;
        mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    return indices;
}

/**
 * @brief Select rows from the dataset using specific indices.
 * @tparam T Data type of the dataset.
 * @param dataset The original dataset.
 * @param indices Indices of the rows to select.
 * @return Subset of the dataset.
 */
template<typename T>
Dataset<T> selectRowsByIndices(const Dataset<T>& dataset, const vector<size_t>& indices ) {
    Dataset<T> subset;
    for (size_t idx : indices) {
        subset.push_back(dataset[idx]);
    }
    return subset;
}

/**
 * @brief Split the dataset into training and test sets.
 * @tparam T Data type of the dataset.
 * @param dataset The dataset to split.
 * @param testFraction Fraction of data to be used for testing.
 * @param shuffle If true, shuffle the data before splitting.
 * @return A pair of training and test datasets.
 */
template<typename T>
pair <Dataset<T>, Dataset<T>> trainTestSplit(Dataset<T> dataset,
                                            double testFraction = 0.2,
                                            bool shuffle = true) {
    if (dataset.empty()) {
        cerr << "Dataset is empty! Returning empty splits." << endl;
        return { {}, {} };
    }
    if (testFraction < 0.0 || testFraction > 1.0) {
        cerr << "Test fraction must be between 0 and 1. Returning empty splits." << endl;
        return { {}, {} };
    }

    size_t totalSize = dataset.size();
    size_t testSize = static_cast<size_t>(totalSize * testFraction);

    vector<size_t> indices = getIndices(dataset.size(), shuffle=shuffle);
    vector<size_t> test_indices(indices.begin(), indices.begin() + testSize);
    vector<size_t> train_indices(indices.begin() + testSize, indices.end());

    Dataset<T> trainSet = selectRowsByIndices(dataset, train_indices);
    Dataset<T> testSet = selectRowsByIndices(dataset, test_indices);

    return { trainSet, testSet };
}


/**
 * @brief Save a dataset to a binary file (general version for int, double).
 * @tparam T Data type of the dataset.
 * @param dataset The dataset to save.
 * @param filename Output binary file name.
 */
template<typename T>
void saveDatasetToBinary(const Dataset<T>& dataset, const string& filename) {
    ofstream outFile(filename, ios::binary);
    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    size_t rows = dataset.size();
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));

    for (const auto& row : dataset) {
        size_t cols = row.size();
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        outFile.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(T));
    }

    outFile.close();
}

/**
 * @brief Load a dataset from a binary file (general version for int, double).
 * @tparam T Data type of the dataset.
 * @param filename Binary file name to read.
 * @return Loaded dataset.
 */
template<typename T>
Dataset<T> loadDatasetFromBinary(const string& filename) {
    ifstream inFile(filename, ios::binary);
    Dataset<T> dataset;

    if (!inFile.is_open()) {
        cerr << "Error opening file for reading: " << filename << endl;
        return dataset;
    }

    size_t rows;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));

    for (size_t i = 0; i < rows; ++i) {
        size_t cols;
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        DataRow<T> row(cols);
        inFile.read(reinterpret_cast<char*>(row.data()), cols * sizeof(T));
        dataset.push_back(row);
    }

    inFile.close();
    return dataset;
}

/**
 * @brief Specialization: Save a dataset of strings to a binary file.
 * @param dataset The string dataset to save.
 * @param filename Output binary file name.
 */
template<>
inline void saveDatasetToBinary<string>(const Dataset<string>& dataset, const string& filename) {
    ofstream outFile(filename, ios::binary);
    if (!outFile.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    size_t rows = dataset.size();
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));

    for (const auto& row : dataset) {
        size_t cols = row.size();
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        for (const auto& str : row) {
            size_t len = str.size();
            outFile.write(reinterpret_cast<const char*>(&len), sizeof(len));
            outFile.write(str.data(), len);
        }
    }

    outFile.close();
}

/**
 * @brief Specialization: Load a dataset of strings from a binary file.
 * @param filename Binary file name to read.
 * @return Loaded string dataset.
 */
template<>
inline Dataset<string> loadDatasetFromBinary<string>(const string& filename) {
    ifstream inFile(filename, ios::binary);
    Dataset<string> dataset;

    if (!inFile.is_open()) {
        cerr << "Error opening file for reading: " << filename << endl;
        return dataset;
    }

    size_t rows;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));

    for (size_t i = 0; i < rows; ++i) {
        size_t cols;
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        DataRow<string> row;
        for (size_t j = 0; j < cols; ++j) {
            size_t len;
            inFile.read(reinterpret_cast<char*>(&len), sizeof(len));
            string str(len, '\0');
            inFile.read(&str[0], len);
            row.push_back(str);
        }
        dataset.push_back(row);
    }

    inFile.close();
    return dataset;
}

/**
 * @brief Flattens a 2D matrix (vector of vectors) into a 1D vector.
 *
 * This function removes the 2D structure of the matrix and creates a single 
 * 1D vector containing all the elements of the matrix in row-major order.
 *
 * @param matrix A 2D vector containing the matrix elements.
 * @return A 1D vector containing the flattened elements of the matrix.
 */
template <typename T>
std::vector<T> squeeze(const std::vector<std::vector<T>>& matrix) {
    std::vector<T> result;

    // Iterate through the rows and columns of the matrix
    for (const auto& row : matrix) {
        for (T value : row) {
            result.push_back(value);
        }
    }

    return result;
}

/**
 * @brief Adds a new dimension to a 1D vector, converting it to a 2D vector along a given axis.
 *
 * This function "unsqueezes" a 1D vector by inserting a new dimension of size 1
 * at the specified axis. The function supports higher-dimensional tensors by choosing
 * the axis at which the dimension is inserted.
 *
 * @param vector A 1D vector to be unsqueezed.
 * @param axis The axis index at which to insert the size-1 dimension.
 * @return A 2D vector where the original vector is placed along the specified axis.
 */
template <typename T>
std::vector<std::vector<T>> unsqueeze(const std::vector<T>& vector, int axis=1) {
    std::vector<std::vector<T>> result;

    if (axis == 0) {
        // Insert the 1D vector as the only row of a 2D matrix
        result.push_back(vector);
    } else if (axis == 1) {
        // Insert the 1D vector as the only column of a 2D matrix
        for (T value : vector) {
            result.push_back({value});
        }
    } else {
        std::cerr << "Invalid axis value. Only axis 0 and 1 are supported in this case.\n";
    }

    return result;
}

#endif // DATASET_UTILS_H
