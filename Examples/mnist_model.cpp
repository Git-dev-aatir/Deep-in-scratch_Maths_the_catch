#include <iostream>
#include "Models/Sequential.h"
#include "Metrics/Losses.h"
#include "Data/Dataset.h"
#include "Data/Preprocessing.h"
#include "Utils/Activations.h"
#include "Utils/Scheduler.h"

using namespace std;

int main() {

    // Dataset mnist_train;
    // mnist_train.loadCSV("Datasets/MNIST/mnist_train.csv", ',', true); 
    // Dataset mnist_test;
    // mnist_test.loadCSV("Datasets/MNIST/mnist_train.csv", ',', true); 

    // // mnist_train.describe();
    // mnist_train.head();

    // mnist_train.saveBinary("Datasets/MNIST/mnist_train.bin");
    // mnist_test.saveBinary("Datasets/MNIST/mnist_test.bin");

// ----------------------Converted-csv-to-binary-------------------------------

    Dataset mnist_train;
    mnist_train.loadBinary("Datasets/MNIST/mnist_train.bin", true); // not uploaded in repo
    Dataset mnist_test;
    mnist_test.loadBinary("Datasets/MNIST/mnist_train.bin", true); // you can test your own dataset

    cout << "Training set : ";
    mnist_train.printShape();
    // mnist_train.head(1);

    cout << "Testing set : "; 
    mnist_test.printShape();
    // mnist_test.head(1);

    auto [X_train, y_train] = mnist_train.splitFeaturesLabels(0);
    auto [X_test, y_test] = mnist_test.splitFeaturesLabels(0);
    
    y_test.describe();
    y_train.describe();

    // Normalize features
    Preprocessing::standardize(X_train);
    Preprocessing::standardize(X_test);

    // Convert labels to one-hot
    y_train.toOneHot();
    y_test.toOneHot();
    
    // Build model
    size_t hidden_unit = 10;  // Increased capacity
    Sequential model(
        std::make_unique<DenseLayer>(X_train.cols(), hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, y_train.cols())
    );
    model.initializeParameters(21);
    model.summary();

    size_t epochs = 100;
    
    // Create optimizer
    const double base_lr = 0.0001;
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

    // auto start = chrono::high_resolution_clock::now();
    
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

    // auto end = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    
    // std::cout << "Training completed at " << duration.count() << " \n";    

    return 0; 
}