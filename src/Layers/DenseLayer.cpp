#include "../../include/Layers/DenseLayer.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath> // For fabs

// Constructor with enhanced validation
DenseLayer::DenseLayer(size_t in_features, size_t out_features, bool init_params)
    : input_size(in_features), output_size(out_features)
{
    if (in_features == 0 || out_features == 0)
    {
        throw std::invalid_argument("DenseLayer: Input and output features must be > 0");
    }

    // Initialize gradient storage
    grad_weights.resize(output_size, std::vector<double>(input_size, 0.0));
    grad_biases.resize(output_size, 0.0);

    // Initialize parameters if requested
    if (init_params)
    {
        weights.resize(output_size, std::vector<double>(input_size, 0.0));
        biases.resize(output_size, 0.0);
    }
}

// Weight initialization - removed redundant bias_value parameter
void DenseLayer::initializeWeights(InitMethod method, unsigned int seed,
                                   double a, double b, double sparsity, double constant_value)
{
    weights = initializeParameters(input_size, output_size, method, seed, a, b, sparsity, constant_value);
}

// Bias initialization with constant_value parameter
void DenseLayer::initializeBiases(InitMethod method, unsigned int seed,
                                  double a, double b, double sparsity, double constant_value)
{
    // Initialize as matrix then convert to vector
    auto temp = initializeParameters(output_size, 1, method, seed, a, b, sparsity, constant_value);

    // Validate and convert
    if (temp.size() != 1 || temp[0].size() != output_size) {
        throw std::runtime_error("Bias initialization returned incorrect dimensions");
    }
    
    biases = temp[0];
}

// Forward pass with bounds checking
std::vector<double> DenseLayer::forward(const std::vector<double> &input)
{
    if (input.size() != input_size) {
        throw std::invalid_argument("DenseLayer::forward: Input size mismatch. Expected " + 
                                    std::to_string(input_size) + ", got " + 
                                    std::to_string(input.size()));
    }

    if (weights.empty() || biases.empty()) {
        throw std::runtime_error("DenseLayer::forward: Parameters not initialized");
    }

    // Cache input for backward pass
    input_cache = input;

    // Pre-allocate output
    std::vector<double> output(output_size, 0.0);

    // Optimized computation: y = Wx + b
    for (size_t i = 0; i < output_size; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < input_size; ++j) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + biases[i];
    }

    return output;
}

// Backward pass with gradient computation
std::vector<double> DenseLayer::backward(const std::vector<double> &grad_output)
{
    if (grad_output.size() != output_size) {
        throw std::invalid_argument("DenseLayer::backward: Gradient size mismatch. Expected " + 
                                    std::to_string(output_size) + ", got " + 
                                    std::to_string(grad_output.size()));
    }

    if (input_cache.empty()) {
        throw std::logic_error("DenseLayer::backward: Forward pass not cached");
    }

    // Compute input gradient: dL/dx = W^T * dL/dy
    std::vector<double> grad_input(input_size, 0.0);
    for (size_t j = 0; j < input_size; ++j) {
        for (size_t i = 0; i < output_size; ++i) {
            grad_input[j] += weights[i][j] * grad_output[i];
        }
    }

    // Accumulate parameter gradients
    for (size_t i = 0; i < output_size; ++i) {
        // Weight gradients: dL/dW = dL/dy * x^T
        for (size_t j = 0; j < input_size; ++j) {
            grad_weights[i][j] += grad_output[i] * input_cache[j];
        }
        // Bias gradients: dL/db = dL/dy
        grad_biases[i] += grad_output[i];
    }

    return grad_input;
}

// Reset accumulated gradients
void DenseLayer::clearGradients()
{
    for (auto &row : grad_weights) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0);
}

// Display layer summary
void DenseLayer::summary() const
{
    const size_t total_params = (input_size * output_size) + output_size;
    std::cout << "Dense Layer: " << input_size << " -> " << output_size
              << " | Parameters: " << total_params << " ("
              << input_size * output_size << " weights, "
              << output_size << " biases)\n";
}

// Print weights with formatting
void DenseLayer::printWeights() const
{
    if (weights.empty()) {
        std::cout << "Weights not initialized" << std::endl;
        return;
    }

    std::cout << "Weights [" << output_size << "×" << input_size << "]:\n";
    for (size_t i = 0; i < output_size; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < input_size; ++j) {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << weights[i][j];
            if (j < input_size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

// Print biases with formatting
void DenseLayer::printBiases() const
{
    if (biases.empty()) {
        std::cout << "Biases not initialized" << std::endl;
        return;
    }

    std::cout << "Biases [" << output_size << "]:\n  [";
    for (size_t i = 0; i < output_size; ++i) {
        std::cout << std::fixed << std::setprecision(5) << std::setw(8) << biases[i];
        if (i < output_size - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Get total learnable parameters
size_t DenseLayer::getParameterCount() const {
    return (input_size * output_size) + output_size;
}

// Getters with const correctness
const std::vector<std::vector<double>>& DenseLayer::getGradWeights() const {
    return grad_weights;
}

const std::vector<double>& DenseLayer::getGradBiases() const {
    return grad_biases;
}

const std::vector<std::vector<double>>& DenseLayer::getWeights() const {
    return weights;
}

const std::vector<double>& DenseLayer::getBiases() const {
    return biases;
}

// Setters with enhanced validation
void DenseLayer::setWeights(const std::vector<std::vector<double>>& new_weights)
{
    if (new_weights.size() != output_size) {
        throw std::invalid_argument("DenseLayer::setWeights: Row count mismatch");
    }
    for (const auto& row : new_weights) {
        if (row.size() != input_size) {
            throw std::invalid_argument("DenseLayer::setWeights: Column count mismatch");
        }
    }
    weights = new_weights;
}

void DenseLayer::setWeights(const std::vector<std::vector<double>>&& new_weights)
{
    if (new_weights.size() != output_size) {
        throw std::invalid_argument("DenseLayer::setWeights: Row count mismatch");
    }
    for (const auto& row : new_weights) {
        if (row.size() != input_size) {
            throw std::invalid_argument("DenseLayer::setWeights: Column count mismatch");
        }
    }
    weights = std::move(new_weights);
}

void DenseLayer::setBiases(const std::vector<double>& new_biases)
{
    if (new_biases.size() != output_size) {
        throw std::invalid_argument("DenseLayer::setBiases: Size mismatch");
    }
    biases = new_biases;
}

void DenseLayer::setBiases(const std::vector<double>&& new_biases)
{
    if (new_biases.size() != output_size) {
        throw std::invalid_argument("DenseLayer::setBiases: Size mismatch");
    }
    biases = std::move(new_biases);
}
