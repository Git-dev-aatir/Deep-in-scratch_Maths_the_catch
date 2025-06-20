#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "Layers.h"

using std::vector;
using std::string;

/**
 * @brief Enumeration of supported activation functions.
 */
enum class ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX,
    SELU
};

/**
 * @brief Applies the specified activation function to each element of the input vector.
 * 
 * @param x Input vector.
 * @param act_type Activation type to apply.
 * @param alpha LeakyReLU negative slope (default 0.01), also used in SELU.
 * @param lambda SELU scaling factor (default 1.0507).
 * @return Activated vector.
 */
vector<double> applyActivation(const vector<double>& x, ActivationType act_type, 
                               double alpha=0.01, double lambda = 1.0507) {
    
    vector<double> result(x.size());

    switch (act_type) {
        case ActivationType::RELU:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = std::max(0.0, x[i]);
            break;

        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = (x[i] > 0.0) ? x[i] : alpha * x[i];
            break;

        case ActivationType::SIGMOID:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = 1.0 / (1.0 + std::exp(-x[i]));
            break;

        case ActivationType::TANH:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = std::tanh(x[i]);
            break;

        case ActivationType::LINEAR:
            result = x;
            break;

        case ActivationType::SOFTMAX: {
            double max_elem = *std::max_element(x.begin(), x.end());
            double sum = 0.0;
            for (double val : x)
                sum += std::exp(val - max_elem);
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = std::exp(x[i] - max_elem) / sum;
            break;
        }

        case ActivationType::SELU:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = lambda * ((x[i] > 0) ? x[i] : alpha * (std::exp(x[i]) - 1));
            break;

        default:
            throw std::invalid_argument("Unsupported ActivationType");
    }

    return result;
}

/**
 * @brief Computes the derivative of the activation function element-wise.
 * 
 * @param x Input vector (pre-activation).
 * @param act_type Activation type.
 * @param alpha LeakyReLU negative slope (default 0.01), also used in SELU.
 * @param lambda SELU scaling factor (default 1.0507).
 * @return Derivative vector.
 */
vector<double> activationDerivative(const vector<double>& x, ActivationType act_type,
                                    double alpha = 0.01, double lambda = 1.0507) {
    
    vector<double> deriv(x.size());

    switch (act_type) {
        case ActivationType::RELU:
            for (size_t i = 0; i < x.size(); ++i)
                deriv[i] = (x[i] > 0.0) ? 1.0 : 0.0;
            break;

        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < x.size(); ++i)
                deriv[i] = (x[i] > 0.0) ? 1.0 : alpha;
            break;

        case ActivationType::SIGMOID:
            for (size_t i = 0; i < x.size(); ++i) {
                double sig = 1.0 / (1.0 + std::exp(-x[i]));
                deriv[i] = sig * (1.0 - sig);
            }
            break;

        case ActivationType::TANH:
            for (size_t i = 0; i < x.size(); ++i) {
                double t = std::tanh(x[i]);
                deriv[i] = 1.0 - t * t;
            }
            break;

        case ActivationType::LINEAR:
            for (size_t i = 0; i < x.size(); ++i)
                deriv[i] = 1.0;
            break;

        case ActivationType::SOFTMAX:
            throw std::logic_error("Softmax derivative is computed with cross-entropy.");
            break;

        case ActivationType::SELU:
            for (size_t i = 0; i < x.size(); ++i)
                deriv[i] = (x[i] > 0) ? lambda : lambda * alpha * std::exp(x[i]);
            break;

        default:
            throw std::invalid_argument("Unsupported ActivationType");
    }

    return deriv;
}

/**
 * @brief Converts ActivationType enum to readable string.
 * 
 * @param act_type Activation type.
 * @return String representation.
 */
string activationTypeToString(ActivationType act_type) {
    switch (act_type) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::LINEAR: return "Linear";
        case ActivationType::SOFTMAX: return "Softmax";
        case ActivationType::SELU: return "SELU";
        default: return "Unknown";
    }
}

/**
 * @brief Activation Layer (supports multiple activations).
 */
class ActivationLayer : public Layer {
private:
    ActivationType activation_type;
    vector<double> input_cache;

public:
    ActivationLayer(ActivationType act_type) : activation_type(act_type) {}

    vector<double> forward(const vector<double>& input) override {
        input_cache = input;
        return applyActivation(input, activation_type);
    }

    vector<double> backward(const vector<double>& grad_output, double learning_rate=0.1) override {
        vector<double> grad_input = activationDerivative(input_cache, activation_type);
        for (size_t i = 0; i < grad_input.size(); ++i)
            grad_input[i] *= grad_output[i];
        return grad_input;
    }

    void summary() const override {
        std::cout << "Activation Layer: " << activationTypeToString(activation_type) << std::endl;
    }
};

#endif // ACTIVATION_H
