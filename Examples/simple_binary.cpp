#include <iostream>
#include <chrono>
#include "../include/Preprocessing/Dataset_utils.h"
#include "../include/Models/Sequential.h"
#include "../include/Layers/Layers.h"
#include "../include/Metrics/Losses.h"   // Assuming you have loss functions here
#include "../include/Optimizers/Optim.h"

#define DATA_PATH "./Datasets/test_binary/"

using namespace std;

// Simple Mean Squared Error loss and its derivative
double Bce_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::bce_loss(y_true, y_pred);
}

std::vector<double> Bce_loss_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::bce_derivative(y_true, y_pred);
}

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

    Sequential model = Sequential(
        new DenseLayer(X_train[0].size(), 6),
        new ActivationLayer(ActivationType::RELU),
        new DenseLayer(6, 7),
        new ActivationLayer(ActivationType::RELU),
        new DenseLayer(7, 1),
        new ActivationLayer(ActivationType::SIGMOID)
    );

    model.initializeParameters(21, 0, 1);

    model.summary();

    size_t epochs = 100;
    double lr = 0.01;

// ----------------------------------SGD---------------------------------------------

    SGD optimizer1(lr); // You can replace with BatchGradientDescent if you want batch training

    // Flatten y_train and y_test to 1D vector<double>
    vector<double> y_train_flat, y_test_flat;
    for (const auto& row : y_train) y_train_flat.push_back(row[0]);
    for (const auto& row : y_test)  y_test_flat.push_back(row[0]);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        double loss = model.train(X_train, y_train, 
                                  Bce_loss,
                                  Bce_loss_derivative,
                                  &optimizer1
        );

        // Evaluate on test set
        size_t correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> output = model.forward(X_test[i]);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y_test_flat[i]);
            if (predicted == actual) ++correct;
        }
        double accuracy = 100.0 * correct / X_test.size();

        std::cout << "Epoch " << epoch + 1 << " | Loss: " << loss << " | Accuracy: " << accuracy << "%\n";
    }

// ----------------------------------MiniBatch---------------------------------------------

    MiniBatchGD optimizer2(lr); // You can replace with BatchGradientDescent if you want batch training

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = model.train(X_train, y_train, 
                                  Bce_loss,
                                  Bce_loss_derivative,
                                  &optimizer1
        );

        // Evaluate on test set
        size_t correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> output = model.forward(X_test[i]);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y_test_flat[i]);
            if (predicted == actual) ++correct;
        }
        double accuracy = 100.0 * correct / X_test.size();

        std::cout << "Epoch " << epoch + 1 << " | Loss: " << loss << " | Accuracy: " << accuracy << "%\n";
    }
 
// ----------------------------------BatchGD---------------------------------------------

    BatchGD optimizer3(lr); // You can replace with BatchGradientDescent if you want batch training

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = model.train(X_train, y_train, 
                                  Bce_loss,
                                  Bce_loss_derivative,
                                  &optimizer1
        );

        // Evaluate on test set
        size_t correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> output = model.forward(X_test[i]);
            int predicted = output[0] >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(y_test_flat[i]);
            if (predicted == actual) ++correct;
        }
        double accuracy = 100.0 * correct / X_test.size();

        std::cout << "Epoch " << epoch + 1 << " | Loss: " << loss << " | Accuracy: " << accuracy << "%\n";
    }

    return 0;
}