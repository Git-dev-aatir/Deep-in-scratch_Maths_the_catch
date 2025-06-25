#include <iostream>
#include "../../include/Preprocessing/dataset_utils.h"
#include "../../include/Preprocessing/preprocessing.h"


#define DATA_PATH "../../Datasets/Iris/"

using namespace std;

int main() {

    Dataset<double> X_train = loadDataset<double>(string(DATA_PATH)+"X_train.csv", ',');    
    Dataset<double> y_train = loadDataset<double>(string(DATA_PATH)+"y_train.csv", ',');
    Dataset<double> X_test = loadDataset<double>(string(DATA_PATH)+"X_test.csv", ',');
    Dataset<double> y_test = loadDataset<double>(string(DATA_PATH)+"y_test.csv", ',');    

    describeDataset(X_train);

    // handling missing values
    findMissingValues(X_train);
    findMissingValues(y_train);
    findMissingValues(X_test);
    findMissingValues(y_test);

    // squeezing labels to 1D vectors
    vector<double> y_train_1d = squeeze(y_train);
    vector<double> y_test_1d = squeeze(y_test);
    printDimensions(X_train);
    printDimensions(y_train);
    printDimensions(X_test);
    printDimensions(y_test);

    // analyzing correlations
    // computeCorrelationMatrix(X_train);
    // computeCorrelationWithTarget(X_train, y_train_1d);

    // standardize data
    standardize(X_train);
    standardize(X_test);

    // save standardized data 
    saveDatasetToBinary(X_train, string(DATA_PATH)+"X_train.bin");
    saveDatasetToBinary(X_test, string(DATA_PATH)+"X_test.bin");
    saveDatasetToBinary(y_train, string(DATA_PATH)+"y_train.bin");
    saveDatasetToBinary(y_test, string(DATA_PATH)+"y_test.bin");

    return 0;
}