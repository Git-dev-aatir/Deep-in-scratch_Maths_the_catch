#pragma once

#include "BaseLayer.h"
#include "Activation_utils.h"

/**
 * @brief Activation layer that applies non-linear activation functions.
 */
class ActivationLayer : public BaseLayer {
private:
    ActivationType activation_type; ///< Type of activation function.
    std::vector<double> input_cache; ///< Cached input for derivative computation.
    double alpha; ///< Parameter for Leaky ReLU and SELU
    double lambda; ///< Parameter for SELU

public:
    /**
     * @brief Constructor to specify the activation function.
     * @param act_type Activation function type.
     * @param alpha Parameter for Leaky ReLU (default 0.01) and SELU (default 1.67326)
     * @param lambda Parameter for SELU (default 1.0507)
     */
    ActivationLayer(ActivationType act_type, double alpha = 0.01, double lambda = 1.0507);

    /**
     * @brief Performs the forward pass of the activation layer.
     * 
     * Applies the activation function element-wise to the input vector.
     * The input is stored in a cache for use in the backward pass.
     * 
     * @param input A vector containing the input data to the activation layer.
     * @return A vector containing the output of the activation function applied to the input.
     */
    std::vector<double> forward(const std::vector<double>& input) override;
    
    /**
     * @brief Performs the backward pass of the activation layer.
     * 
     * Computes the gradient of the loss with respect to the inputs of the layer.
     * The derivative of the activation function is used to propagate the error gradient.
     * 
     * @param grad_output A vector containing the gradients of the loss with respect to the output of this layer.
     * @param lr The learning rate used for gradient descent.
     * @return A vector containing the gradients of the loss with respect to the inputs of this layer.
     */
    std::vector<double> backward(const std::vector<double>& grad_output) override;
    
    /**
     * @brief Prints the details of the activation layer.
     * 
     * Outputs information about the type of activation function used, 
     * along with the values of any relevant parameters such as alpha and lambda.
     */
    void summary() const override;

    /**
     * @brief Retrieves the type of activation function used in the layer.
     * 
     * @return The activation function type as an enum value of type ActivationType.
     */
    ActivationType getActivationType() const;
};
