#include <iostream>
#include "../include/Models/Sequential.h"
#include "../include/Layers/Layers.h"
#include "../include/Optimizers/Optim.h"
#include "../include/Metrics/Losses.h"
#include "../include/Preprocessing/Dataset_utils.h"

#define DATA_PATH "./Datasets/test_linear/"

using namespace std; 

// Simple Mean Squared Error loss and its derivative
double Mse_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::mse_loss(y_true, y_pred);
}

std::vector<double> Mse_loss_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::mse_derivative(y_true, y_pred);
}

// testing purely linear data with 3 features and no noise
int main () {

    Dataset<double> X_train = loadDataset<double>(string(DATA_PATH) + "X_train.csv", ',');
    Dataset<double> y_train = loadDataset<double>(string(DATA_PATH) + "y_train.csv", ',');
    Dataset<double> X_test = loadDataset<double>(string(DATA_PATH) + "X_test.csv", ',');
    Dataset<double> y_test = loadDataset<double>(string(DATA_PATH) + "y_test.csv", ',');
    
    // printDimensions(X_train);
    // head(X_train, X_train.size());

    // printDimensions(y_train);
    // head(y_train, y_train.size());

    // printDimensions(X_test);
    // head(X_test);

    // printDimensions(y_test);
    // head(y_test);
    // return 0;

    // Create a simple model: Input -> Dense(1 unit) -> Sigmoid
    Sequential model(
        new DenseLayer(X_train[0].size(), 1)               // 1 input -> 1 output neuron
    );

    model.initializeParameters(); // Initialize weights and biases

    // Optimizer: Stochastic Gradient Descent (learning rate 0.1)
    double lr = 0.01; 
    SGD optimizer(lr);

    // Train for 100 epochs
    size_t epochs = 100;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = model.train(X_train, y_train, Mse_loss, Mse_loss_derivative, &optimizer);
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss << std::endl;
    }

    // 5. Test the model
    cout << "\nTesting the trained model:" << endl;
    for (size_t i=0; i<X_test.size(); ++i) {
        const vector<double> x = X_test[i];
        vector<double> pred = model.forward(x);
        cout << "Real value: " << y_test[i][0] << " -> Prediction: " << pred[0] << endl;
    }

    // 6. Print summary 
    cout << "\nSummary of model \n";
    model.summary();

    // // 6. Print learned weights and biases
    // cout << "\nLearned Weight: \n";
    // head(model.getWeights());
    // cout << "Learned Bias: " << model.getBiases()[0] << endl;
    


    return 0;
}