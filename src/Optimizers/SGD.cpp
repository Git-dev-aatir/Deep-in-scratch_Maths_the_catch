#include "Optimizers/SGD.h"
#include "Layers/Layers.h"
#include <stdexcept>
#include <iostream>

SGD::SGD(double lr, double momentum, std::function<double(double, size_t)> scheduler) 
    : learning_rate(lr), initial_lr(lr), momentum(momentum), lr_scheduler(scheduler) {}

void SGD::afterStep() {
    step_count++;
    if (lr_scheduler) {
        learning_rate = lr_scheduler(initial_lr, step_count);
    }
}


void SGD::setLRScheduler(std::function<double(double, size_t)> scheduler) {
    lr_scheduler = scheduler;
}

void SGD::setLearningRate(double lr) {
    learning_rate = lr;
    // Reset initial LR if needed
    if (!lr_scheduler) initial_lr = lr;
}

void SGD::decayLearningRate(double decay_factor) {
    learning_rate *= decay_factor;
    if (!lr_scheduler) initial_lr = learning_rate;
}

void SGD::step(std::vector<BaseLayer*> layers, size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    for (BaseLayer* layer : layers) {
        updateLayer(layer, batch_size);
    }
}

void SGD::updateLayer(BaseLayer* layer, size_t batch_size) {
    DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(layer);
    if (!dense_layer) return;

    // Get references to parameters and gradients
    const auto& weights = dense_layer->getWeights();
    const auto& biases = dense_layer->getBiases();
    auto& grad_weights = dense_layer->getGradWeights();
    auto& grad_biases = dense_layer->getGradBiases();
    
    // Get matrix dimensions
    const size_t output_size = weights.size();
    const size_t input_size = weights.empty() ? 0 : weights[0].size();

    // Initialize velocity buffers if using momentum
    if (momentum > 0) {
        if (velocity_weights.find(layer) == velocity_weights.end()) {
            velocity_weights[layer] = std::vector<std::vector<double>>(
                output_size, 
                std::vector<double>(input_size, 0.0)  // Corrected: input_size columns
            );
            velocity_biases[layer] = std::vector<double>(biases.size(), 0.0);
        }
    }

    const double lr = this->learning_rate / batch_size;
    
    // Create updated parameters
    auto new_weights = weights;
    auto new_biases = biases;

    // Update weights
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            if (momentum > 0) {
                velocity_weights[layer][i][j] = 
                    momentum * velocity_weights[layer][i][j] + 
                    lr * grad_weights[i][j];
                new_weights[i][j] -= velocity_weights[layer][i][j];
            } else {
                new_weights[i][j] -= lr * grad_weights[i][j];
            }
        }
    }

    // Update biases
    for (size_t i = 0; i < new_biases.size(); ++i) {
        if (momentum > 0) {
            velocity_biases[layer][i] = 
                momentum * velocity_biases[layer][i] + 
                lr * grad_biases[i];
            new_biases[i] -= velocity_biases[layer][i];
        } else {
            new_biases[i] -= lr * grad_biases[i];
        }
    }

    // Update parameters via setters
    dense_layer->setWeights(new_weights);
    dense_layer->setBiases(new_biases);
    
    // Clear gradients after update
    dense_layer->clearGradients();
}
