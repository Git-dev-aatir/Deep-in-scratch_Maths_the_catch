#include "Models/Sequential.h"
#include <iostream>
#include <stdexcept>

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

std::vector<double> Sequential::backward(const std::vector<double>& grad_output) {
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

double Sequential::train(const Dataset& X_train,
                         const Dataset& y_train,
                         BaseOptim& optimizer,
                         std::function<double(const std::vector<double>&, 
                                              const std::vector<double>&)> loss_fn,
                         std::function<std::vector<double>(const std::vector<double>&, 
                                                           const std::vector<double>&)> grad_fn,
                         unsigned int seed
) {
    size_t batch_size = optimizer.getBatchSize();
    if (batch_size == 0) {
        batch_size = X_train.rows();
        optimizer.setBatchSize(batch_size);
    }
    DataLoader loader(X_train, batch_size, true, seed);
    double total_loss = 0.0;
    
    for (auto it = loader.begin(); it != loader.end(); ++it) {
        Dataset batch = *it;
        const auto& batch_data = batch.getData();
        auto batch_indices = it.getIndices();
        size_t current_batch_size = batch_data.size();

        // clear gradient cache 
        this->clearGradients();
        
        // Process batch
        for (size_t i = 0; i < current_batch_size; ++i) {
            const auto& x = batch_data[i];
            const auto& y_true = y_train[batch_indices[i]];
            
            // Forward pass
            auto y_pred = forward(x);
            
            // Compute loss and gradient
            total_loss += loss_fn(y_true, y_pred);
            auto grad = grad_fn(y_true, y_pred);
            
            backward(grad);
        }
        
        // Update parameters
        optimizer.step(getLayers(), current_batch_size);

        // Notify optimizer after step (for schedulers)
        optimizer.afterStep();
    }
    return total_loss;
}

double Sequential::train(
    const Dataset& X_train,
    const Dataset& y_train,
    BaseOptim& optimizer,
    std::function<double(const std::vector<std::vector<double>>&, 
                         const std::vector<std::vector<double>>&)> batch_loss_fn,
    std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&, 
                                                   const std::vector<std::vector<double>>&)> batch_grad_fn,
    unsigned int seed
) {
    size_t batch_size = optimizer.getBatchSize();
    if (batch_size == 0) {
        batch_size = X_train.rows();
        optimizer.setBatchSize(batch_size);
    }
    DataLoader loader(X_train, batch_size, true, seed);
    double total_loss = 0.0;
    
    for (auto it = loader.begin(); it != loader.end(); ++it) {
        Dataset batch = *it;
        const auto& batch_data = batch.getData();
        auto batch_indices = it.getIndices();
        size_t current_batch_size = batch_data.size();
        
        // Prepare batch inputs and labels
        std::vector<std::vector<double>> batch_y;
        batch_y.reserve(current_batch_size);
        for (auto idx : batch_indices) {
            batch_y.push_back(y_train[idx]);
        }

        // clearing gradient cache
        this->clearGradients();
        
        // Forward pass for entire batch
        std::vector<std::vector<double>> batch_preds;
        batch_preds.reserve(current_batch_size);
        for (const auto& x : batch_data) {
            batch_preds.push_back(forward(x));
        }
        
        // Compute batch loss
        double batch_loss = batch_loss_fn(batch_y, batch_preds); 
        total_loss += batch_loss * current_batch_size;
        
        // Compute batch gradients
        auto batch_grads = batch_grad_fn(batch_y, batch_preds);
        
        // Backward pass for each sample in batch
        for (const auto& grad : batch_grads) {
            backward(grad);
        }

        // // After backward pass in train():
        // double max_grad = 0;
        // for (auto layer : getLayers()) {
        //     if (auto dense = dynamic_cast<DenseLayer*>(layer)) {
        //         for (const auto& row : dense->getGradWeights()) {
        //             for (double g : row) {
        //                 if (fabs(g) > max_grad) max_grad = fabs(g);
        //             }
        //         }
        //     }
        // }
        // std::cout << "Max Gradient: " << max_grad << std::endl;
        
        // Update parameters
        optimizer.step(getLayers(), current_batch_size);
        optimizer.afterStep();
    } 
    return total_loss / X_train.rows();
}


void Sequential::clearGradients() {
    std::vector<BaseLayer*> all_layers = this->getLayers();
    for (auto& layer : all_layers) {
        DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer);
        if (!dense_layer) return;
        dense_layer->clearGradients();
    }
}