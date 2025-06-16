#include "../include/dataset_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>

// Function to split a string by a delimiter
vector<string> split(const string &line, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(line);

    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to safely parse int (returns 0 on failure)
int parseInt(const string &token) {
    try {
        return stoi(token);
    } catch (...) {
        cerr << "Warning: Non-integer value \"" << token << "\" encountered. Storing as 0." << endl;
        return 0;
    }
}

// Function to load dataset as ints
Dataset loadDataset(const string &filename, char delimiter=',') {
    ifstream file(filename);
    Dataset dataset;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return dataset;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> tokens = split(line, delimiter);
        DataRow row;

        for (const auto &token : tokens) {
            row.push_back(parseInt(token));
        }

        dataset.push_back(row);
    }

    file.close();
    return dataset;
}

// Function to save dataset to CSV file
void saveDatasetToCSV(const Dataset &dataset, const string &outputFilename) {
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
