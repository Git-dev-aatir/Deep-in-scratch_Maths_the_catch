#include "../../include/Models/Sequential.h"
#include "../../include/Preprocessing/Dataset_utils.h"
#include <iostream>

void Sequential::initializeParameters(unsigned int seed, 
                        double a, double b, 
                        double sparsity, double bias_value) 
{
    for (size_t i = 0; i < this->layers.size(); ++i) {
        DenseLayer* dense_layer = dynamic_cast<DenseLayer*>(this->layers[i]);
        InitMethod method;
        if (dense_layer) {
            method = InitMethod::XAVIER_NORMAL;

            if (i + 1 < this->layers.size()) {
                ActivationLayer* act_layer = dynamic_cast<ActivationLayer*>(layers[i + 1]);
                if (act_layer) {
                    ActivationType act_type = act_layer->getActivationType();
                    switch (act_type) {
                        case ActivationType::RELU:
                        case ActivationType::LEAKY_RELU:
                            method = InitMethod::HE_NORMAL;
                            break;
                        case ActivationType::SIGMOID:
                        case ActivationType::TANH:
                            method = InitMethod::XAVIER_NORMAL;
                            break;
                        case ActivationType::SELU:
                            method = InitMethod::LECUN_NORMAL;
                            break;
                        default:
                            method = InitMethod::XAVIER_NORMAL;
                    }
                }
            }
            else {
                method = InitMethod::XAVIER_UNIFORM;
            }

            dense_layer->initializeWeights(method, seed, a, b, sparsity, bias_value);
            dense_layer->initializeBiases(InitMethod::BIAS, seed, a, b, sparsity, bias_value);
        }
    }
    this->is_initialized = true; // mark as initialized
}

std::vector<double> Sequential::forward(const std::vector<double>& input) const {
    if (!is_initialized) {
        throw std::runtime_error("Error: Model parameters not initialized. Call initializeParameters() before forward().");
    }

    std::vector<double> output = input;
    for (BaseLayer* layer : this->layers) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<double> Sequential::backward(const std::vector<double>& grad_output, 
                                         double learning_rate) {
    std::vector<double> grad = grad_output;
    for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }
    return grad;
}

double Sequential::train(const std::vector<std::vector<double>>& X,
             const std::vector<std::vector<double>>& Y_true,
             double (*loss_func)(const std::vector<double>&, const std::vector<double>&),
             std::vector<double> (*loss_derivative)(const std::vector<double>&, const std::vector<double>&),
             Optimizer* optimizer)
{
    if (!is_initialized) {
        throw std::runtime_error("Error: Model parameters not initialized. Call initializeParameters() before training.");
    }

    size_t batch_size = X.size();
    std::vector<DenseLayer*> dense_layers;

    for (BaseLayer* layer : this->layers) {
        if (auto dense = dynamic_cast<DenseLayer*>(layer)) {
            dense_layers.push_back(dense);
        }
    }

    optimizer->before_epoch(dense_layers);
    optimizer->clear_gradients(dense_layers);

    double total_loss = 0.0;
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<double> output = this->forward(X[i]);
        const std::vector<double>& y_true_vec = Y_true[i];
        total_loss += loss_func(y_true_vec, output);

        std::vector<double> grad_output = loss_derivative(y_true_vec, output);
        this->backward(grad_output);
        optimizer->step_per_sample(dense_layers);

    }

    optimizer->step_after_batch(dense_layers);
    // optimizer->clear_gradients(dense_layers);

    return total_loss / batch_size;
}

double Sequential::train(const std::vector<std::vector<double>>& X,
             const std::vector<std::vector<double>>& Y_true,
             double (*loss_func)(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&),
             std::vector<std::vector<double>> (*loss_derivative)(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&),
             Optimizer* optimizer)
{
    if (!is_initialized) {
        throw std::runtime_error("Error: Model parameters not initialized. Call initializeParameters() before training.");
    }

    size_t batch_size = X.size();
    std::vector<DenseLayer*> dense_layers;

    for (BaseLayer* layer : this->layers) {
        if (auto dense = dynamic_cast<DenseLayer*>(layer)) {
            dense_layers.push_back(dense);
        }
    }

    optimizer->before_epoch(dense_layers);
    optimizer->clear_gradients(dense_layers);

    // // Forward pass for entire batch
    // std::vector<std::vector<double>> outputs(batch_size);
    // for (size_t i = 0; i < batch_size; ++i) {
    //     outputs[i] = this->forward(X[i]);
    // }

    double total_loss = 0.0;
    // std::vector<std::vector<double>> grad_outputs = loss_derivative(Y_true, outputs);

    for (size_t i = 0; i < batch_size; ++i) {

        // Forward pass for single sample
        std::vector<double> output = this->forward(X[i]);

        // Calculate loss for this sample
        std::vector<std::vector<double>> single_output = {output};
        std::vector<std::vector<double>> single_target = {Y_true[i]};
        
        double sample_loss = loss_func(single_target, single_output);
        total_loss += sample_loss;
        
        // Get gradient for this sample
        std::vector<std::vector<double>> grad_outputs = loss_derivative(single_target, single_output);
        
        // Backward pass for single sample

        this->backward(grad_outputs[0]);
        optimizer->step_per_sample(dense_layers);
        // std::cout << "\nWeights after iteration '" << i << "' :\n";
        // const auto& weights = dense_layers[dense_layers.size()-1]->getWeights();
        // head(weights, weights.size());
        // std::cout << std::endl;
    }

    // Average the accumulated gradients
    // for (auto* layer : dense_layers) {
    //     auto& grad_weights = const_cast<std::vector<std::vector<double>>&>(layer->getGradWeights());
    //     auto& grad_biases = const_cast<std::vector<double>&>(layer->getGradBiases());
        
    //     for (auto& row : grad_weights) {
    //         for (auto& val : row) {
    //             val /= batch_size;
    //         }
    //     }
    //     for (auto& val : grad_biases) {
    //         val /= batch_size;
    //     }
    // }

    // Update weights once per batch
    // optimizer->step(dense_layers);
    
    optimizer->step_after_batch(dense_layers);
    // optimizer->clear_gradients(dense_layers);

    return total_loss / batch_size;
}

void Sequential::summary() const {
    std::cout << "Sequential Model Summary:\n";
    for (const BaseLayer* layer : this->layers) {
        layer->summary();
    }
}

Sequential::~Sequential() {
    for (BaseLayer* layer : this->layers) {
        delete layer;
    }
}
