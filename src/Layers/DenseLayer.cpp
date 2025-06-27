#include "../../include/Layers/DenseLayer.h"
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath> // For fabs

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

void DenseLayer::initializeWeights(InitMethod method, unsigned int seed,
                                   double a, double b, double sparsity, double bias_value)
{
    if (weights.empty())
    {
        weights.resize(output_size, std::vector<double>(input_size, 0.0));
    }

    weights = initializeParameters(input_size, output_size, method, seed, a, b, sparsity, bias_value);
}

void DenseLayer::initializeBiases(InitMethod method, unsigned int seed,
                                  double a, double b, double sparsity, double bias_value)
{
    if (biases.empty())
    {
        biases.resize(output_size, 0.0);
    }

    // Initialize as matrix then convert to vector
    auto temp = initializeParameters(output_size, 1, method, seed, a, b, sparsity, bias_value);

    // Validate and convert
    if (temp.size() != 1 || temp[0].size() != output_size)
    {
        throw std::runtime_error("Bias initialization returned incorrect dimensions");
    }
    biases = temp[0];
}

std::vector<double> DenseLayer::forward(const std::vector<double> &input)
{
    if (input.size() != input_size)
    {
        throw std::invalid_argument("DenseLayer::forward: Expected input size " +
                                    std::to_string(input_size) + ", got " +
                                    std::to_string(input.size()));
    }

    if (weights.empty() || biases.empty())
    {
        throw std::runtime_error("DenseLayer::forward: Weights/biases not initialized");
    }

    // Cache input for backward pass
    input_cache = input;

    // Pre-allocate output
    std::vector<double> output(output_size, 0.0);

    // Compute output: y = Wx + b
    for (size_t i = 0; i < output_size; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < input_size; ++j)
        {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + biases[i];
    }

    return output;
}

std::vector<double> DenseLayer::backward(const std::vector<double> &grad_output)
{
    if (grad_output.size() != output_size)
    {
        throw std::invalid_argument("DenseLayer::backward: Expected grad_output size " +
                                    std::to_string(output_size) + ", got " +
                                    std::to_string(grad_output.size()));
    }

    if (input_cache.empty())
    {
        throw std::logic_error("DenseLayer::backward: Forward pass not called or cache cleared");
    }

    // Compute gradient w.r.t. input: dL/dx = W^T * dL/dy
    std::vector<double> grad_input(input_size, 0.0);
    for (size_t j = 0; j < input_size; ++j)
    {
        for (size_t i = 0; i < output_size; ++i)
        {
            grad_input[j] += weights[i][j] * grad_output[i];
        }
    }

    // Accumulate gradients w.r.t. weights and biases
    for (size_t i = 0; i < output_size; ++i)
    {
        // dL/dW_i = dL/dy_i * x^T
        for (size_t j = 0; j < input_size; ++j)
        {
            grad_weights[i][j] += grad_output[i] * input_cache[j];
        }
        // dL/db_i = dL/dy_i
        grad_biases[i] += grad_output[i];
    }

    return grad_input;
}

void DenseLayer::clearGradients()
{
    for (auto &row : grad_weights)
    {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(grad_biases.begin(), grad_biases.end(), 0.0);
}

void DenseLayer::summary() const
{
    const size_t total_params = (input_size * output_size) + output_size;
    std::cout << "Dense Layer: " << input_size << " -> " << output_size
              << " | Parameters: " << total_params << " ("
              << input_size * output_size << " weights + "
              << output_size << " biases)\n";
}

void DenseLayer::printWeights() const
{
    if (this->weights.empty())
    {
        std::cout << "Weights not initialized" << std::endl;
        return;
    }

    std::cout << "Weights [" << this->output_size << "x" << this->input_size << "]:" << std::endl;
    for (size_t i = 0; i < this->output_size; ++i)
    {
        std::cout << " [";
        for (size_t j = 0; j < this->input_size; ++j)
        {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << this->weights[i][j];
            if (j < this->input_size - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void DenseLayer::printBiases() const
{
    if (this->biases.empty())
    {
        std::cout << "Biases not initialized" << std::endl;
        return;
    }

    std::cout << "Biases [" << this->output_size << "]:" << std::endl;
    std::cout << " [";
    for (size_t i = 0; i < this->output_size; ++i)
    {
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << this->biases[i];
        if (i < this->output_size - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

size_t DenseLayer::getParameterCount() const
{
    return (this->input_size * this->output_size) + this->output_size;
}

// Getters
const std::vector<std::vector<double>> &DenseLayer::getGradWeights() const
{
    return this->grad_weights;
}

const std::vector<double> &DenseLayer::getGradBiases() const
{
    return this->grad_biases;
}

const std::vector<std::vector<double>> &DenseLayer::getWeights() const
{
    return this->weights;
}

const std::vector<double> &DenseLayer::getBiases() const
{
    return this->biases;
}

// Setters with enhanced validation
void DenseLayer::setWeights(const std::vector<std::vector<double>> &new_weights)
{
    if (new_weights.size() != output_size)
    {
        throw std::invalid_argument("DenseLayer::setWeights: Row count mismatch. Expected " +
                                    std::to_string(output_size) + ", got " +
                                    std::to_string(new_weights.size()));
    }
    for (const auto &row : new_weights)
    {
        if (row.size() != input_size)
        {
            throw std::invalid_argument("DenseLayer::setWeights: Column count mismatch. Expected " +
                                        std::to_string(input_size) + ", got " +
                                        std::to_string(row.size()));
        }
    }
    weights = std::move(new_weights);
}

void DenseLayer::setBiases(const std::vector<double> &new_biases)
{
    if (new_biases.size() != output_size)
    {
        throw std::invalid_argument("DenseLayer::setBiases: Size mismatch. Expected " +
                                    std::to_string(output_size) + ", got " +
                                    std::to_string(new_biases.size()));
    }
    biases = std::move(new_biases);
}
