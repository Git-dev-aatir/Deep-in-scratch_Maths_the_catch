#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "Data/Dataset.h"
#include "Data/DataLoader.h"
#include "Utils/Sequential.h"
#include "Utils/SGD.h"
#include "Layers/Layers.h"
#include "Metrics/Losses.h"
#include "Data/Preprocessing.h"
#include "utils/Activations.h"

using namespace std;

double loss_fun (vector<double> correct, vector<double> predics) {
    Losses::cross_entropy_loss(correct, predics, true);
}

vector<double> loss_derivative (vector<double> correct, vector<double> predics) {
    Losses::cross_entropy_derivative(correct, predics, true);
}

int main() {
    // Load dataset
    Dataset iris;
    iris.loadCSV("Examples/3_iris/iris.data", ',', true);  // Assumes CSV with header
    
    // Inspect dataset
    std::cout << "Dataset dimensions: ";
    auto shape = iris.shape();
    std::cout << shape.first << " rows x " << shape.second << " columns\n";
    std::cout << "First 5 rows:\n";
    // iris.head(5);

    // train-test-split 
    auto [train_set, test_set] = iris.trainTestSplit(0.2, true, iris.cols()-1);
    
    auto [train_rows, train_cols] = train_set.shape();
    auto [test_rows, test_cols] = test_set.shape();
    cout << "Train set : " << train_rows << " x " << train_cols << endl;
    cout << "test set : " << test_rows << " x " << test_cols << endl;


    // Split features and labels (last column is label)
    auto [X, y] = train_set.splitFeaturesLabels();
    // auto split = iris.splitFeaturesLabels();
    // Dataset& X = split.first;
    // Dataset& y = split.second;

    auto [X_test, y_test] = test_set.splitFeaturesLabels();

    Preprocessing::standardize(X);
    Preprocessing::standardize(X_test);


    // Convert integer labels to one-hot encoding
    // Preprocessing::oneHotEncode(y, {0});  // Column 0 is the label column
    // y_test.toOneHot();
    y.toOneHot();
    std::cout << "After one-hot encoding, label dimensions: " 
              << y.shape().first << "x" << y.shape().second << "\n";

    // Test with known values
    std::vector<double> y_true = {1,0,0};
    std::vector<double> y_pred = {0.01, 0.99, 0.0}; // Bad prediction
    double loss = Losses::cross_entropy_loss(y_true, y_pred, false);
    std::cout << "Test loss: " << loss; // Should be ~4.6





    // Create DataLoader for training
    DataLoader loader(X, 32, true);  // batch_size=32, shuffle=true

    // Build model
    Sequential model(
        std::make_unique<DenseLayer>(X.cols(), 4),  // 4 features -> 16 neurons
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(4, 4),  // 4 features -> 16 neurons
        std::make_unique<ActivationLayer>(ActivationType::LEAKY_RELU),
        std::make_unique<DenseLayer>(4, y.cols())   // 16 neurons -> 3 outputs
        // std::make_unique<ActivationLayer>(ActivationType::SOFTMAX)
    );

    // Initialize parameters with Xavier initialization
    model.initializeParameters(21);  // seed=21 for reproducibility
    model.summary();

    // Check untrained model predictions
    std::vector<double> sample = X[0];
    std::cout << "Sample features: ";
    for (auto v : sample) std::cout << v << " ";
    std::cout << "\n";

    auto pred = model.forward(sample);
    std::cout << "Untrained prediction: ";
    for (auto p : pred) std::cout << p << " ";
    std::cout << "\n";

    // Create optimizer with momentum
    // In training loop
    double base_lr = 0.008;
    size_t base_batch_size = 16;
    SGD optimizer(base_lr, 0.9);  // lr=0.01, momentum=0.9

    // Training loop
    size_t epochs = 1000;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        
        double current_lr = base_lr * 0.5 * (1 + cos(3.1417 * epoch / epochs));
        // current_lr = base_lr;
        optimizer.setLearningRate(current_lr);

        for (auto batch : loader) {
            const auto& batch_data = batch.getData();
            size_t current_batch_size = batch_data.size();
            

            // Process each sample in batch
            for (size_t i = 0; i < current_batch_size; ++i) {
                const auto& x = batch_data[i];
                const auto& y_true = y[i];  // Corresponding label

                // Forward pass
                auto y_pred = model.forward(x);
                
                // Compute loss
                epoch_loss += Losses::cross_entropy_loss(y_true, y_pred, true);
                
                // Compute gradient
                auto grad = Losses::cross_entropy_derivative(y_true, y_pred, true);
                
                // Backward pass
                model.backward(grad, current_lr);  // lr not used in backward
            }
            
            // Update parameters
            if (current_batch_size!=0) {
                optimizer.step(model.getLayers(), current_batch_size);
                // std::cout << "Epoch : " << epoch+1 << "batch-size is 0\n";
            } 
            else std::cout << "Epoch : " << epoch+1 << "batch-size is 0\n";
        }
        // epoch_loss = model.fit(X, y, optimizer, 32, loss_fun, loss_derivative);


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
        
        // Print every epoch
        if ((epoch+1) % (epochs/10) == 0) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << "\t | LR: " << current_lr
                  << "\t | Loss: " << epoch_loss / X.rows()
                  << "\t | Acc: " << accuracy << "%\n";
        }
        
        // Learning rate decay
        // if ((epoch + 1) % 30 == 0) {
        //     optimizer.decayLearningRate(0.5);  // Halve learning rate
        //     std::cout << "Reduced learning rate to: " << optimizer.getLearningRate() << "\n";
        // }
    }

    std::cout << "Training completed!\n";
    return 0;
}
