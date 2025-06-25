#include "../../include/Layers/Activation_utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

std::vector<double> applyActivation(const std::vector<double>& x, ActivationType act_type,
                                    double alpha, double lambda) {
    std::vector<double> result(x.size());
    switch (act_type) {
        case ActivationType::RELU:
            for (size_t i = 0; i < x.size(); ++i) result[i] = std::max(0.0, x[i]);
            break;
        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < x.size(); ++i) result[i] = (x[i] > 0) ? x[i] : alpha * x[i];
            break;
        case ActivationType::SIGMOID:
            for (size_t i = 0; i < x.size(); ++i) result[i] = 1.0 / (1.0 + std::exp(-x[i]));
            break;
        case ActivationType::TANH:
            for (size_t i = 0; i < x.size(); ++i) result[i] = std::tanh(x[i]);
            break;
        case ActivationType::LINEAR:
            result = x;
            break;
        case ActivationType::SOFTMAX: {
            double max_elem = *std::max_element(x.begin(), x.end());
            double sum = 0.0;
            for (double val : x) sum += std::exp(val - max_elem);
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = std::exp(x[i] - max_elem) / sum;
            break;
        }
        case ActivationType::SELU:
            for (size_t i = 0; i < x.size(); ++i)
                result[i] = lambda * ((x[i] > 0) ? x[i] : alpha * (std::exp(x[i]) - 1));
            break;
    }
    return result;
}

std::vector<double> activationDerivative(const std::vector<double>& x, ActivationType act_type,
                                         double alpha, double lambda) {
    std::vector<double> deriv(x.size());
    switch (act_type) {
        case ActivationType::RELU:
            for (size_t i = 0; i < x.size(); ++i) deriv[i] = (x[i] > 0) ? 1.0 : 0.0;
            break;
        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < x.size(); ++i) deriv[i] = (x[i] > 0) ? 1.0 : alpha;
            break;
        case ActivationType::SIGMOID:
            for (size_t i = 0; i < x.size(); ++i) {
                double sig = 1.0 / (1.0 + std::exp(-x[i]));
                deriv[i] = sig * (1 - sig);
            }
            break;
        case ActivationType::TANH:
            for (size_t i = 0; i < x.size(); ++i) {
                double t = std::tanh(x[i]);
                deriv[i] = 1 - t * t;
            }
            break;
        case ActivationType::LINEAR:
            for (size_t i = 0; i < x.size(); ++i) deriv[i] = 1.0;
            break;
        case ActivationType::SOFTMAX:
            throw std::logic_error("Softmax derivative should be handled with cross-entropy.");
            break;
        case ActivationType::SELU:
            for (size_t i = 0; i < x.size(); ++i)
                deriv[i] = (x[i] > 0) ? lambda : lambda * alpha * std::exp(x[i]);
            break;
    }
    return deriv;
}

std::string activationTypeToString(ActivationType act_type) {
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
