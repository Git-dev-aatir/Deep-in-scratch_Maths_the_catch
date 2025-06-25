#pragma once

#include <vector>
#include <string>

/**
 * @brief Enumeration for different activation function types.
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
 * @brief Applies an activation function to the input vector.
 * 
 * Given an activation function type, applies the corresponding element-wise activation 
 * function to the input vector. The function can handle different activation types 
 * such as ReLU, Leaky ReLU, Sigmoid, Tanh, Linear, Softmax, and SELU.
 * 
 * @param x A vector containing the input values to which the activation function will be applied.
 * @param act_type The type of activation function to apply. Can be one of the values from the ActivationType enum.
 * @param alpha A parameter used by Leaky ReLU and SELU. Default is 0.01.
 * @param lambda A parameter used by SELU. Default is 1.0507.
 * @return A vector containing the result of applying the activation function element-wise to the input.
 */
std::vector<double> applyActivation(const std::vector<double>& x, ActivationType act_type,
                                    double alpha = 0.01, double lambda = 1.0507);

/**
 * @brief Computes the derivative of the activation function.
 * 
 * Given an activation function type, computes the derivative of the corresponding 
 * activation function with respect to its input values. The derivative is used 
 * in backpropagation for training neural networks.
 * 
 * @param x A vector containing the input values of the activation function.
 * @param act_type The type of activation function to compute the derivative for. 
 *                 Can be one of the values from the ActivationType enum.
 * @param alpha A parameter used by Leaky ReLU and SELU. Default is 0.01.
 * @param lambda A parameter used by SELU. Default is 1.0507.
 * @return A vector containing the derivatives of the activation function applied element-wise to the input.
 */
std::vector<double> activationDerivative(const std::vector<double>& x, ActivationType act_type,
                                         double alpha = 0.01, double lambda = 1.0507);

/**
 * @brief Converts activation type to its string representation.
 * 
 * Converts an activation function type (enum) into a human-readable string representation.
 * This is useful for logging, debugging, or summarizing model architecture.
 * 
 * @param act_type The activation function type to convert.
 * @return A string corresponding to the activation type, e.g., "ReLU", "Leaky ReLU", "Sigmoid", etc.
 */
std::string activationTypeToString(ActivationType act_type);
