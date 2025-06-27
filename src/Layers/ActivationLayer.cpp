#include "../../include/Layers/ActivationLayer.h"
#include <iostream>
#include <stdexcept>

ActivationLayer::ActivationLayer(ActivationType act_type, double alpha, double lambda) 
    : activation_type(act_type), alpha(alpha), lambda(lambda) {
    
    // Apply standard SELU parameters if using defaults
    if (act_type == ActivationType::SELU && alpha == 0.01) {
        this->alpha = 1.67326;  // Standard SELU alpha
    }
}

std::vector<double> ActivationLayer::forward(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("ActivationLayer: Input cannot be empty");
    }
    
    // Cache input for backward pass
    input_cache = input;
    
    // Apply activation function
    return applyActivation(input, activation_type, alpha, lambda);
}

std::vector<double> ActivationLayer::backward(const std::vector<double>& grad_output) {
    if (grad_output.empty()) {
        throw std::invalid_argument("ActivationLayer: Gradient output cannot be empty");
    }
    if (input_cache.size() != grad_output.size()) {
        throw std::logic_error("ActivationLayer: Input cache and gradient size mismatch");
    }
    
    // Handle softmax special case (combined with CE loss)
    if (activation_type == ActivationType::SOFTMAX) {
        return grad_output;  // Loss function handles derivative
    }
    
    // Compute activation derivative
    auto deriv = activationDerivative(input_cache, activation_type, alpha, lambda);
    
    // Element-wise gradient multiplication (chain rule)
    std::vector<double> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * deriv[i];
    }
    
    return grad_input;
}

void ActivationLayer::summary() const {
    std::cout << "Activation Layer: " << activationTypeToString(activation_type);
    
    // Display parameters for relevant activation types
    switch(activation_type) {
        case ActivationType::LEAKY_RELU:
            std::cout << " (alpha=" << alpha << ")";
            break;
        case ActivationType::SELU:
            std::cout << " (alpha=" << alpha << ", lambda=" << lambda << ")";
            break;
        default: 
            break;
    }
    std::cout << " | Input size: " << input_cache.size() << "\n";
}

ActivationType ActivationLayer::getActivationType() const {
    return activation_type;
}
