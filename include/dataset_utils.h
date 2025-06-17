#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include <vector>
#include <string>
#include <iostream>
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

// trim from start (in place)
inline void ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(),
        [](unsigned char ch) { return !isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(),
        [](unsigned char ch) { return !isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(string &s) {
    ltrim(s);
    rtrim(s);
}

// Function to split a string by a delimiter
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

// Generic parse function template
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

// Function to load dataset of type T
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

// Function to save dataset to CSV file
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

// Function to display first 'n_rows' of the dataset
template<typename T>
void head(const Dataset<T> &dataset, int n_rows = 5) {
    for (const auto &row : dataset) {
        for (const auto &val : row) {
            cout << val << "\t| ";
        }
        cout << endl;
        --n_rows;
        if (n_rows <= 0) break;
    }
}

// print dimensions of Dataset 
template<typename T>
void printDimensions(const vector<vector<T>>& vec) {
    size_t rows = vec.size();
    size_t cols = rows > 0 ? vec[0].size() : 0;
    cout << "Dimensions: [" << rows << " x " << cols << "]" << endl;
}


// Split dataset into features and labels
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

// Split dataset into training set and test set
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

template<typename T>
Dataset<T> selectRowsByIndices(const Dataset<T>& dataset, const vector<size_t>& indices ) {
    Dataset<T> subset;
    for (size_t idx : indices) {
        subset.push_back(dataset[idx]);
    }
    return subset;
}

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


template<typename T>
using Dataset = vector<DataRow<T>>;

// Save Dataset to binary (general version for int, double)
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

// Load Dataset from binary (general version for int, double)
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

// Specialization for string - Save
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

// Specialization for string - Load
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



#endif // DATASET_UTILS_H
