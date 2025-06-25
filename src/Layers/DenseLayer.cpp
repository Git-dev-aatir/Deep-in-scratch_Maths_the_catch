#include "../../include/Layers/DenseLayer.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>

DenseLayer::DenseLayer(size_t in_features, size_t out_features, bool init_params)
    : input_size(in_features), output_size(out_features) 
{
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Layer dimensions must be positive");
    }
    
    if (init_params) {
        this->weights.resize(this->output_size, std::vector<double>(this->input_size, 0.0));
        this->biases.resize(this->output_size, 0.0);
    }
    this->grad_weights.resize(this->output_size, std::vector<double>(this->input_size, 0.0));
    this->grad_biases.resize(this->output_size, 0.0);
}

void DenseLayer::initializeWeights(InitMethod method, unsigned int seed,
                                   double a, double b, double sparsity, double bias_value) {
    if (this->weights.empty()) {
        // Initialize weights if not already done
        this->weights.resize(this->output_size, std::vector<double>(this->input_size, 0.0));
    }
    
    this->weights = initializeParameters(this->input_size, this->output_size,
                                         method, seed, a, b, sparsity, bias_value);
}

void DenseLayer::initializeBiases(InitMethod method, unsigned int seed,
                                  double a, double b, double sparsity, double bias_value) {
    if (this->biases.empty()) {
        // Initialize biases if not already done
        this->biases.resize(this->output_size, 0.0);
    }
    
    std::vector<std::vector<double>> temp = initializeParameters(this->output_size, 1,
                                                                 method, seed, a, b, sparsity, bias_value);
    this->biases = temp[0];
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    if (input.size() != this->input_size) {
        throw std::invalid_argument("Input size mismatch. Expected " + 
                                   std::to_string(this->input_size) + 
                                   ", got " + std::to_string(input.size()));
    }
    
    if (this->weights.empty() || this->biases.empty()) {
        throw std::runtime_error("Layer weights/biases not initialized");
    }
    
    // Cache input for backward pass
    this->input_cache = input;
    
    // Compute output: y = Wx + b
    std::vector<double> output(this->output_size, 0.0);
    for (size_t i = 0; i < this->output_size; ++i) {
        // Compute dot product of weights[i] and input
        double sum = 0.0;
        for (size_t j = 0; j < this->input_size; ++j) {
            sum += this->weights[i][j] * input[j];
        }
        output[i] = sum + this->biases[i];
    }
    
    return output;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& grad_output, double lr) {
    if (grad_output.size() != this->output_size) {
        throw std::invalid_argument("Gradient output size mismatch. Expected " + 
                                   std::to_string(this->output_size) + 
                                   ", got " + std::to_string(grad_output.size()));
    }
    
    if (this->input_cache.empty()) {
        throw std::runtime_error("No cached input found. Must call forward() before backward()");
    }
    
    // Compute gradient w.r.t. input: grad_input = W^T * grad_output
    std::vector<double> grad_input(this->input_size, 0.0);
    for (size_t i = 0; i < this->input_size; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < this->output_size; ++j) {
            sum += this->weights[j][i] * grad_output[j];
        }
        grad_input[i] = sum;
    }
    
    // Accumulate gradients w.r.t. weights: grad_W = grad_output * input^T
    for (size_t i = 0; i < this->output_size; ++i) {
        for (size_t j = 0; j < this->input_size; ++j) {
            this->grad_weights[i][j] += grad_output[i] * this->input_cache[j];
        }
        // Accumulate gradients w.r.t. biases: grad_b = grad_output
        this->grad_biases[i] += grad_output[i];
    }
    
    return grad_input;
}

void DenseLayer::applyGradients(double learning_rate, size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    
    // Apply gradients with learning rate and batch averaging
    double lr_normalized = learning_rate / static_cast<double>(batch_size);
    
    for (size_t i = 0; i < this->output_size; ++i) {
        for (size_t j = 0; j < this->input_size; ++j) {
            this->weights[i][j] -= lr_normalized * this->grad_weights[i][j];
        }
        this->biases[i] -= lr_normalized * this->grad_biases[i];
    }
}

void DenseLayer::clearGradients() {
    for (auto& row : this->grad_weights) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(this->grad_biases.begin(), this->grad_biases.end(), 0.0);
}

void DenseLayer::summary() const {
    size_t total_params = (this->input_size * this->output_size) + this->output_size;
    std::cout << "Dense Layer Summary:" << std::endl;
    std::cout << "  Input size:  " << this->input_size << std::endl;
    std::cout << "  Output size: " << this->output_size << std::endl;
    std::cout << "  Parameters:  " << total_params << " (" 
              << (this->input_size * this->output_size) << " weights + " 
              << this->output_size << " biases)" << std::endl;
}

void DenseLayer::printWeights() const {
    if (this->weights.empty()) {
        std::cout << "Weights not initialized" << std::endl;
        return;
    }
    
    std::cout << "Weights [" << this->output_size << "x" << this->input_size << "]:" << std::endl;
    for (size_t i = 0; i < this->output_size; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < this->input_size; ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << this->weights[i][j];
            if (j < this->input_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void DenseLayer::printBiases() const {
    if (this->biases.empty()) {
        std::cout << "Biases not initialized" << std::endl;
        return;
    }
    
    std::cout << "Biases [" << this->output_size << "]:" << std::endl;
    std::cout << "  [";
    for (size_t i = 0; i < this->output_size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << this->biases[i];
        if (i < this->output_size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

size_t DenseLayer::getParameterCount() const {
    return (this->input_size * this->output_size) + this->output_size;
}

// Getters
const std::vector<std::vector<double>>& DenseLayer::getGradWeights() const { 
    return this->grad_weights; 
}

const std::vector<double>& DenseLayer::getGradBiases() const { 
    return this->grad_biases; 
}

const std::vector<std::vector<double>>& DenseLayer::getWeights() const { 
    return this->weights; 
}

const std::vector<double>& DenseLayer::getBiases() const { 
    return this->biases; 
}

// Setters with validation
void DenseLayer::setWeights(const std::vector<std::vector<double>>& new_weights) {
    if (new_weights.size() != this->output_size) {
        throw std::invalid_argument("Weight matrix row count mismatch");
    }
    for (const auto& row : new_weights) {
        if (row.size() != this->input_size) {
            throw std::invalid_argument("Weight matrix column count mismatch");
        }
    }
    this->weights = new_weights;
}

void DenseLayer::setBiases(const std::vector<double>& new_biases) {
    if (new_biases.size() != this->output_size) {
        throw std::invalid_argument("Bias vector size mismatch");
    }
    this->biases = new_biases;
}
