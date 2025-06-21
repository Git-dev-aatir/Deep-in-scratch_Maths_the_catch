#include <iostream>
#include "../include/Preprocessing/dataset_utils.h"
#include "../include/Models/Sequential.h"

#define DATA_PATH "../Datasets/test_binary/"

int main() {

    Dataset<double> X_train = loadDataset<double>(string(DATA_PATH)+"X_train.csv", ',');
    Dataset<double> y_train = loadDataset<double>(string(DATA_PATH)+"y_train.csv", ',');
    Dataset<double> X_test = loadDataset<double>(string(DATA_PATH)+"X_test.csv", ',');
    Dataset<double> y_test = loadDataset<double>(string(DATA_PATH)+"y_test.csv", ',');

    // printDimensions(X_train);
    // head(X_train);

    // printDimensions(y_train);
    // head(y_train);

    // printDimensions(X_test);
    // head(X_test);

    // printDimensions(y_test);
    // head(y_test);

    Sequential model;
    model.addLayer(new Dense(3, 6));
    model.addLayer(new ActivationLayer(ActivationType::RELU));
    model.addLayer(new Dense(6, 7));
    model.addLayer(new ActivationLayer(ActivationType::RELU));
    model.addLayer(new Dense(7, 1));
    model.addLayer(new ActivationLayer(ActivationType::SIGMOID));

    model.initializeParameters(21, 0, 1);

    model.summary();

    size_t epochs = 100;
    for (size_t epoch=0; epoch<epochs; ++epoch) {

        

    }

    return 0;
}