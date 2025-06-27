#include <iostream>
#include "../../include/Preprocessing/Dataset_utils.h"
#include "../../include/Utils/Sequential.h"
#include "../../include/Utils/SGD.h"
#include "../../include/Metrics/Losses.h"
#include "../../include/Utils/Activations.h"
#include "../../include/Data/CsvDataset.h"
#include "../../include/Data/DataLoader.h"

#define DATA_PATH "./Datasets/Iris/"

using namespace std;

double ce_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::cross_entropy_loss(y_true, y_pred, true);
}

std::vector<double> ce_loss_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    return Losses::cross_entropy_derivative(y_true, y_pred, true);
}

double ce_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred) {
    return Losses::cross_entropy_loss_batch(y_true, y_pred, true);
}

std::vector<std::vector<double>> ce_loss_derivative_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred) {
    return Losses::cross_entropy_derivative_batch(y_true, y_pred, true);
}

int main() {
    // // Load dataset (features and one-hot encoded labels)
    // Dataset<double> X_train = loadDataset<double>(string(DATA_PATH) + "X_train.csv", ',');    
    // Dataset<double> y_train = loadDataset<double>(string(DATA_PATH) + "y_train.csv", ',');
    // Dataset<double> X_test  = loadDataset<double>(string(DATA_PATH) + "X_test.csv", ',');
    // Dataset<double> y_test  = loadDataset<double>(string(DATA_PATH) + "y_test.csv", ',');


    cout << "Dataset loaded. Training samples: " << X_train.size() << ", Test samples: " << X_test.size() << endl;

    // Create model structure
    int hidden = 4;
    // Sequential model = Sequential(
    //     new DenseLayer(X_train[0].size(), hidden), // Input size to hidden layer 1
    //     new ActivationLayer(ActivationType::LEAKY_RELU),
    //     new DenseLayer(hidden, hidden),
    //     new ActivationLayer(ActivationType::LEAKY_RELU),
    //     new DenseLayer(hidden, y_train[0].size()) 
    // );

    Sequential model = Sequential(
        std::make_unique<DenseLayer>(X_train[0].size(), hidden),
        std::make_unique<ActivationLayer>(ActivationType::RELU),
        std::make_unique<DenseLayer>(hidden, hidden),
        std::make_unique<ActivationLayer>(ActivationType::RELU),
        std::make_unique<DenseLayer>(hidden, y_train[0].size())
    );

    // Initialize weights and biases
    model.initializeParameters(21, 0, 1);

    // Print model summary
    // model.summary();

    size_t epochs = 2000;
    double learning_rate = 0.06;

    SGD optimizer(learning_rate, 0.0);
    size_t num_samples = X_train.size();
    size_t batch_size = 32;

    // Training loop for Iris dataset
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        int total_loss = 0;
        for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, num_samples);
            
            // Forward pass (whole batch)
            auto output = model.forward(X_train[batch_start]);
            
            // Compute loss (whole batch)
            auto loss = ce_loss(y_train[batch_start], output);
            total_loss += loss;
            auto grad = ce_loss_derivative(y_train[batch_start], output);
            
            // Backward pass (whole batch)
            model.backward(grad, 0.01);
            
            // Optimizer step (updates all layers)
            optimizer.step(model.getLayers(), batch_end - batch_start);
        }

        // Evaluate on test set
        size_t correct = 0;
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> output = model.forward(X_test[i]);
            output = Activations::softmax(output);

            // Get predicted class (argmax)
            size_t predicted_class = distance(output.begin(), max_element(output.begin(), output.end()));
            size_t actual_class = distance(y_test[i].begin(), max_element(y_test[i].begin(), y_test[i].end()));
            // here max_element() gives iterator the highest probability class in prediction
            // We then get index of that iterator by using distance(begin_iter, class_iter)

            if (predicted_class == actual_class) ++correct;
        }
        if ((epoch+1) % 200 == 0) {
            double accuracy = 100.0 * correct / X_test.size();
            cout << "\nEpoch " << epoch + 1 
                << " | Loss: " << total_loss 
                << " | Accuracy: " << accuracy << "%\n";
        }
    }

    // // SGD Optimizer
    // SGD optimizer(learning_rate);

    // for (size_t epoch = 0; epoch < epochs; ++epoch) {
    //     // Train using batch-wise cross entropy
    //     double loss = model.train(X_train, y_train, 
    //                               ce_loss,
    //                               ce_loss_derivative,
    //                               &optimizer);

    //     // Evaluate on test set
    //     size_t correct = 0;
    //     for (size_t i = 0; i < X_test.size(); ++i) {
    //         vector<double> output = model.forward(X_test[i]);
    //         output = Activations::softmax(output);

    //         // Get predicted class (argmax)
    //         size_t predicted_class = distance(output.begin(), max_element(output.begin(), output.end()));
    //         size_t actual_class = distance(y_test[i].begin(), max_element(y_test[i].begin(), y_test[i].end()));
    //         // here max_element() gives iterator the highest probability class in prediction
    //         // We then get index of that iterator by using distance(begin_iter, class_iter)

    //         if (predicted_class == actual_class) ++correct;
    //     }
    //     if ((epoch+1) % 200 == 0) {
    //         double accuracy = 100.0 * correct / X_test.size();
    //         cout << "\nEpoch " << epoch + 1 
    //             << " | Loss: " << loss 
    //             << " | Accuracy: " << accuracy << "%\n";
    //     }
    // }

    return 0;
}
