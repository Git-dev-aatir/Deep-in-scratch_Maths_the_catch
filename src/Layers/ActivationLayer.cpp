#include "../../include/Layers/ActivationLayer.h"

ActivationLayer::ActivationLayer(ActivationType act_type, double alpha, double lambda) 
    : activation_type(act_type), alpha(alpha), lambda(lambda) {
    
    // Set proper SELU parameters if SELU is selected and defaults are being used
    if (act_type == ActivationType::SELU && alpha == 0.01) {
        this->alpha = 1.67326;  // Standard SELU alpha
    }
}

std::vector<double> ActivationLayer::forward(const std::vector<double>& input) {
    this->input_cache = input;
    return applyActivation(input, this->activation_type, this->alpha, this->lambda);
}

std::vector<double> ActivationLayer::backward(const std::vector<double>& grad_output, 
                                              double lr) {
    // Special handling for softmax (usually used with cross-entropy)
    if (this->activation_type == ActivationType::SOFTMAX) {
        // For softmax with cross-entropy, the loss function typically computes
        // the gradient directly. Return the incoming gradient as-is.
        return grad_output;
    }
    
    std::vector<double> grad_input = activationDerivative( this->input_cache, 
                                                           this->activation_type, 
                                                           this->alpha, 
                                                           this->lambda );
    
    // Element-wise multiplication: chain rule application
    for (size_t i = 0; i < grad_input.size(); ++i) {
        grad_input[i] *= grad_output[i];
    }
    
    return grad_input;
}

void ActivationLayer::summary() const {
    std::cout << "Activation Layer: " << activationTypeToString(this->activation_type);
    
    // Display parameters for parameterized activations
    if (this->activation_type == ActivationType::LEAKY_RELU) {
        std::cout << " (alpha=" << this->alpha << ")";
    } else if (this->activation_type == ActivationType::SELU) {
        std::cout << " (alpha=" << this->alpha << ", lambda=" << this->lambda << ")";
    }
    
    std::cout << std::endl;
}

ActivationType ActivationLayer::getActivationType() const {
    return this->activation_type;
}
