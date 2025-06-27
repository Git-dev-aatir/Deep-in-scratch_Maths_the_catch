#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "Data/Dataset.h"
#include "Data/DataLoader.h"
#include "Models/Sequential.h"
#include "Optimizers/SGD.h"
#include "Layers/Layers.h"
#include "Metrics/Losses.h"
#include "Data/Preprocessing.h"
#include "Utils/Activations.h"
#include "Utils/Scheduler.h"
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

int main() {
    // Load dataset
    Dataset iris;
    iris.loadCSV("Datasets/iris/iris.data", ',', true);
    
    // Inspect dataset
    auto shape = iris.shape();
    std::cout << "Dataset dimensions: " << shape.first << " rows x " << shape.second << " columns\n";
    
    // Train-test split with stratification
    auto [train_set, test_set] = iris.trainTestSplit(0.2, iris.cols()-1, true);
    
    // Split features and labels
    auto [X_train, y_train] = train_set.splitFeaturesLabels();
    auto [X_test, y_test] = test_set.splitFeaturesLabels();
    
    // Normalize features
    Preprocessing::standardize(X_train);
    Preprocessing::standardize(X_test);
    
    // Convert labels to one-hot
    y_train.toOneHot();
    y_test.toOneHot();
    
    // Build model
    size_t hidden_unit = 4;  // Increased capacity
    Sequential model(
        std::make_unique<DenseLayer>(X_train.cols(), hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, y_train.cols())
    );
    model.initializeParameters(21);
    model.summary();

    size_t epochs = 100;
    
    // Create optimizer
    const double base_lr = 0.03;
    const size_t base_batch_size = 32;
    const size_t batches_per_epoch = ceil(static_cast<double>(X_train.rows()) / base_batch_size);
    const size_t total_steps = epochs * batches_per_epoch;
    auto scheduler = Schedulers::cosine(total_steps);
    SGD optimizer(
        base_lr,      // learning_rate
        0.9,          // momentum
        scheduler   // LR scheduler
    );

    
    DataLoader loader(X_train, base_batch_size, true);

    auto start = chrono::high_resolution_clock::now();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        model.clearGradients();
        
        double epoch_loss = model.train(
            X_train, 
            y_train,
            optimizer,
            32,
            // [](const vector<double>& y_true, const vector<double>& y_pred) {
            //     return Losses::cross_entropy_loss(y_true, y_pred, true);
            // },
            // [](const vector<double>& y_true, const vector<double>& y_pred) {
            //     return Losses::cross_entropy_derivative(y_true, y_pred, true);
            [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) {
                return Losses::cross_entropy_loss_batch(y_true, y_pred, true);
            },
            [](const vector<vector<double>>& y_true, const vector<vector<double>>& y_pred) {
                return Losses::cross_entropy_derivative_batch(y_true, y_pred, true);
        }
    );
        
        // Test evaluation
        size_t correct = 0;
        for (size_t i = 0; i < X_test.rows(); ++i) {
            vector<double> output = model.forward(X_test[i]);
            output = Activations::softmax(output);
            size_t pred_class = distance(output.begin(), max_element(output.begin(), output.end()));
            size_t true_class = distance(y_test[i].begin(), max_element(y_test[i].begin(), y_test[i].end()));
            if (pred_class == true_class) correct++;
        }
        double accuracy = static_cast<double>(correct) / X_test.rows() * 100;
        
        // Print every 10 epochs
        if (epoch % 10 == 0 || epoch == epochs-1 || epoch < 10) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " | LR: " << optimizer.getLearningRate()
                      << " | Loss: " << epoch_loss / X_train.rows()
                      << " | Acc: " << accuracy << "%\n";
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    
    std::cout << "Training completed at " << duration.count() << " \n";
    return 0;
}
