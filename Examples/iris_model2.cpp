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

    // Load dataset
    Dataset iris;
    iris.loadCSV("Datasets/iris/iris.data", ',', true);
    
    // Inspect dataset
    auto shape = iris.shape();
    std::cout << "Dataset dimensions: " << shape.first << " rows x " << shape.second << " columns\n";
    
    // Train-test split with stratification
    auto [train_set, test_set] = iris.trainTestSplit(0.2, iris.cols()-1, true);
    
    // Split features and labels
    auto [X_train, y_train] = train_set.splitFeaturesLabels(4);
    auto [X_test, y_test] = test_set.splitFeaturesLabels(4);
    
    // Standardize features
    Preprocessing::standardize(X_train);
    Preprocessing::standardize(X_test);
    
    y_test.toOneHot();
    y_train.toOneHot();
    
    // Build model
    size_t hidden_unit1 = 4, hidden_unit2 = 4, hidden_unit3 = 4;  // Increased capacity
    Sequential model(
        std::make_unique<DenseLayer>(X_train.cols(), hidden_unit1),
        std::make_unique<ActivationLayer>(ActivationType::LEAKY_RELU),
        std::make_unique<DenseLayer>(hidden_unit1, hidden_unit2),
        std::make_unique<ActivationLayer>(ActivationType::LEAKY_RELU),
        // std::make_unique<DenseLayer>(hidden_unit2, hidden_unit3),
        // std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit3, y_train.cols())
    );
    model.initializeParameters(21);
    model.summary();

    size_t epochs = 35;
    
    // Create optimizer
    const double base_lr = 0.005;
    const size_t base_batch_size = 1;
    const size_t batches_per_epoch = ceil(static_cast<double>(X_train.rows()) / base_batch_size);
    const size_t total_steps = epochs * batches_per_epoch;
    auto scheduler = Schedulers::cosine_warmup(1e-4, total_steps, total_steps/4);
    // auto scheduler = Schedulers::step(8*batches_per_epoch, 0.9);
    SGD optimizer(
        base_lr, // / X_train.rows() * 32,      // learning_rate
        0.9,          // momentum
        base_batch_size, 
        scheduler   // LR scheduler
    );
    optimizer.setGradientClip(0.1);
    // double lr = optimizer.getLearningRate();
    // optimizer.setLearningRate(lr / X_train.rows() * 32);
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        
        double epoch_loss = model.train(
            X_train, 
            y_train,
            optimizer,
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
            },
            21
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
                      << " | Loss: " << epoch_loss
                      << " | Acc: " << accuracy << "%\n";
        }
    }

    return 0; 
}