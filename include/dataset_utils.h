#ifndef DATASET_UTILS_H
#define DATASET_UTILS_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <locale>

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
inline vector<string> split(const string &line, char delimiter) {
    vector<string> tokens;

    if (delimiter == ' ') {
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
Dataset<T> loadDataset(const string &filename, char delimiter = ',') {
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
        vector<string> tokens = split(line, delimiter);
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
            cout << val << '\t';
        }
        cout << endl;
        --n_rows;
        if (n_rows <= 0) break;
    }
}

// Split dataset into features and labels
template<typename T>
void splitFeaturesAndLabels(const Dataset<T> &dataset, Dataset<T> &features, Dataset<T> &labels) {
    for (const auto &row : dataset) {
        if (row.size() < 2) {
            cerr << "Warning: Row with insufficient data encountered. Skipping." << endl;
            continue;
        }

        DataRow<T> featureRow(row.begin(), row.end() - 1);
        features.push_back(featureRow);
        labels.push_back(DataRow<T>(row.back()));
    }
}

#endif // DATASET_UTILS_H
