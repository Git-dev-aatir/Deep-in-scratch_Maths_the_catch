#include "Models/Sequential.h"
#include <iostream>
#include <random>
#include <functional> 
#include <numeric>
#include <algorithm>
// #include <chrono>

void Sequential::initializeParameters(unsigned int seed, double a, double b, double sparsity, double bias_value) {
    for (size_t i = 0; i < this->layers.size(); ++i) {
        auto* dense_layer = dynamic_cast<DenseLayer*>(this->layers[i].get());
        if (dense_layer) {
            InitMethod method = InitMethod::XAVIER_UNIFORM; // default
            if (i + 1 < this->layers.size()) {
                auto* act_layer = dynamic_cast<ActivationLayer*>(this->layers[i + 1].get());
                if (act_layer) {
                    switch (act_layer->getActivationType()) {
                        case ActivationType::RELU:
                        case ActivationType::LEAKY_RELU:
                            method = InitMethod::HE_UNIFORM;
                            break;
                        case ActivationType::SIGMOID:
                        case ActivationType::TANH:
                            method = InitMethod::XAVIER_UNIFORM;
                            break;
                        case ActivationType::SELU:
                            method = InitMethod::LECUN_UNIFORM;
                            break;
                        default:
                            method = InitMethod::XAVIER_UNIFORM;
                            break;
                    }
                }
            }
            dense_layer->initializeWeights(method, seed, a, b, sparsity, bias_value);
            dense_layer->initializeBiases(InitMethod::BIAS, seed, a, b, sparsity, bias_value);
        }
    }
}

std::vector<double> Sequential::forward(const std::vector<double>& input) const {
    std::vector<double> output = input;
    for (auto& layer : this->layers) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<double> Sequential::backward(const std::vector<double>& grad_output, double lr) {
    // Add debug print
    // std::cout << "Backprop grad: ";
    // for (auto g : grad_output) std::cout << g << " ";
    // std::cout << "\n";
    
    std::vector<double> grad = grad_output;
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void Sequential::summary() const {
    std::cout << "Sequential Model Summary:\n";
    std::cout << "========================\n";
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << ": ";
        this->layers[i]->summary();
    }
    std::cout << "Total Layers: " << this->layers.size() << "\n";
    std::cout << "========================\n";
}

int Sequential::fit(const Dataset& X_train,
                    const Dataset& y_train,
                    SGD& optimizer,
                    size_t batch_size,
                    std::function<double(const std::vector<double>&, 
                                         const std::vector<double>&)> loss_fn,
                    std::function<std::vector<double>(const std::vector<double>&, 
                                                      const std::vector<double>&)> grad_fn
) {
    DataLoader loader(X_train, batch_size, true);
    double epoch_loss = 0.0;
    
    // Use explicit iterator syntax to access getIndices()
    for (auto it = loader.begin(); it != loader.end(); ++it) {
        Dataset batch = *it;
        const auto& batch_data = batch.getData();
        
        // Get indices from the ITERATOR
        auto batch_indices = it.getCurrentIndices();
        size_t current_batch_size = batch_data.size();
        
        for (size_t i = 0; i < current_batch_size; ++i) {
            const auto& x = batch_data[i];
            const auto& y = y_train[batch_indices[i]];  // Correct global index
            
            auto y_pred = forward(x);
            epoch_loss += loss_fn(y, y_pred);
            auto grad = grad_fn(y, y_pred);
            backward(grad, 0.01);
        }
        
        optimizer.step(getLayers(), current_batch_size);
    }
    return epoch_loss;

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // if (verbose) {
    //     std::cout << "Epoch " << (epoch + 1) << "/" << epochs
    //                 << " | Avg Loss: " << (epoch_loss / X_train.rows())
    //                 << " | Time: " << duration.count() << "ms\n";
    // }
}