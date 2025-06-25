#include <iostream>
#include <vector>
#include "../include/Preprocessing/Dataset_utils.h"
#include "../include/Models/Sequential.h"
#include "../include/Optimizers/Optim.h"
#include "../include/Metrics/Losses.h"
#include "../include/Utils/Activations.h"

#define DATA_PATH "./Datasets/test_crossentropy/"

using namespace std;

double ce_loss(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred) {
    return Losses::cross_entropy_loss_batch(y_true, y_pred, true);
}

std::vector<std::vector<double>> ce_loss_derivative(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred) {
    return Losses::cross_entropy_derivative_batch(y_true, y_pred, true);
}

int main() {
    // Load dataset (features and one-hot encoded labels)
    Dataset<double> X_train = loadDataset<double>(string(DATA_PATH) + "X_train.csv");    
    Dataset<double> y_train = loadDataset<double>(string(DATA_PATH) + "y_train.csv");
    Dataset<double> X_test  = loadDataset<double>(string(DATA_PATH) + "X_test.csv");
    Dataset<double> y_test  = loadDataset<double>(string(DATA_PATH) + "y_test.csv");

    cout << "Dataset loaded. Training samples: " << X_train.size() << ", Test samples: " << X_test.size() << endl;

    // Create model structure
    Sequential model = Sequential(
        new DenseLayer(X_train[0].size(), 8), // Input size to hidden layer 1
        new ActivationLayer(ActivationType::RELU),
        new DenseLayer(8, 8),                 // Hidden layer 1 to hidden layer 2
        new ActivationLayer(ActivationType::RELU),
        new DenseLayer(8, y_train[0].size()) // Hidden layer 2 to output layer
        // new ActivationLayer(ActivationType::SIGMOID) // Softmax for multi-class classification
    );

    // Initialize weights and biases
    model.initializeParameters(21, 0, 1);

    // Print model summary
    // model.summary();

    size_t epochs = 200;
    double learning_rate = 0.01;

    // SGD Optimizer
    SGD optimizer(learning_rate);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Train using batch-wise cross entropy
        double loss = model.train(X_train, y_train, 
                                  ce_loss,
                                  ce_loss_derivative,
                                  &optimizer);

        // Evaluate on test set
        size_t correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> output = Activations::softmax(model.forward(X_test[i]));

            // Get predicted class (argmax)
            size_t predicted_class = distance(output.begin(), max_element(output.begin(), output.end()));
            size_t actual_class = distance(y_test[i].begin(), max_element(y_test[i].begin(), y_test[i].end()));
            // here max_element() gives iterator the highest probability class in prediction
            // We then get index of that iterator by using distance(begin_iter, class_iter)

            if (predicted_class == actual_class) ++correct;
        }
        if ((epoch+1) % 20 == 0) {
            double accuracy = 100.0 * correct / X_test.size();
            cout << "Epoch " << epoch + 1 
                << " | Loss: " << loss 
                << " | Accuracy: " << accuracy << "%\n";
        }
    }

    return 0;
}
