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
#include "utils/Activations.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

int main() {
    // Load dataset
    Dataset iris;
    iris.loadCSV("Examples/3_iris/iris.data", ',', true);
    
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
    
    // Build model with proper architecture
    size_t hidden_unit = 4;
    Sequential model(
        std::make_unique<DenseLayer>(X_train.cols(), hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, hidden_unit),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(hidden_unit, 3),
        std::make_unique<ActivationLayer>(ActivationType::SELU),
        std::make_unique<DenseLayer>(3, y_train.cols())
    );
    model.initializeParameters(21);
    model.summary();
    
    // Create optimizer
    double base_lr = 0.001;
    size_t base_batch_size = 32;
    SGD optimizer(base_lr, 0.9);
    
    DataLoader loader(X_train, base_batch_size, true);
    size_t epochs = 101;
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        
        // Cosine learning rate decay
        double lr = base_lr * 0.5 * (1 + cos(M_PI * epoch / epochs));
        optimizer.setLearningRate(lr);
        
        // Training
        for (auto it = loader.begin(); it != loader.end(); ++it) {
            Dataset batch = *it;  // Get batch data
            const auto& batch_data = batch.getData();
            
            // Get indices from the ITERATOR (not DataLoader)
            auto batch_indices = it.getCurrentIndices();
            size_t current_batch_size = batch_data.size();
            
            for (size_t i = 0; i < current_batch_size; ++i) {
                const auto& x = batch_data[i];
                const auto& y_true = y_train[batch_indices[i]];
                
                auto y_pred = model.forward(x);
                epoch_loss += Losses::cross_entropy_loss(y_true, y_pred, true);
                auto grad = Losses::cross_entropy_derivative(y_true, y_pred, true);
                model.backward(grad, lr);
            }
            optimizer.step(model.getLayers(), current_batch_size);
        }
        
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
        if ((epoch) % (epochs/10) == 0) {
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << "\t | LR: " << lr
                  << "\t | Loss: " << epoch_loss / X_train.rows()
                  << "\t | Acc: " << accuracy << "%\n";
        }
    }
    
    std::cout << "Training completed!\n";
    return 0;
}
