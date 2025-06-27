#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "BaseLayer.h"
#include "../Utils/Initialization.h" // For parameter initialization methods
#include <cassert>

#define MANUAL_SEED 21

/**
 * @brief Dense (fully connected) layer class with gradient storage and weight initialization.
 *
 * This class represents a dense (fully connected) layer in a neural network, with 
 * support for forward and backward passes, gradient storage, and weight initialization.
 * It is used to compute the weighted sum of inputs, add biases, and apply an activation function.
 * The class also supports gradient-based optimization methods (e.g., gradient descent).
 */
class DenseLayer : public BaseLayer {
private:
    size_t input_size;                          ///< Number of input features.
    size_t output_size;                         ///< Number of output neurons.
    std::vector<std::vector<double>> weights;   ///< Weights matrix (size: input_size x output_size).
    std::vector<double> biases;                 ///< Bias vector (size: output_size).
    std::vector<std::vector<double>> grad_weights; ///< Gradient of weights (size: input_size x output_size).
    std::vector<double> grad_biases;               ///< Gradient of biases (size: output_size).
    std::vector<double> input_cache;               ///< Cached input values for backpropagation (size: input_size).

public:
    /**
     * @brief Constructor to initialize Dense layer dimensions and optionally initialize weights and biases.
     *
     * @param in_features Number of input features (neurons in the previous layer).
     * @param out_features Number of output neurons in this layer.
     * @param init_params Whether to initialize weights and biases (default: true).
     */
    DenseLayer(size_t in_features, size_t out_features, bool init_params = true);

    /**
     * @brief Initializes weights using the specified initialization method.
     *
     * This function will initialize the weights using one of the predefined methods, 
     * e.g., Xavier, He, or random initialization. The initialization method is passed 
     * as a parameter, and the seed can be set for reproducibility.
     *
     * @param method Initialization method to use (e.g., Xavier, He).
     * @param seed Seed for random number generation (default: 21).
     * @param a Lower bound of the range for weight initialization (default: 0).
     * @param b Upper bound of the range for weight initialization (default: 1.0).
     * @param sparsity The sparsity of the weight matrix (default: 0.0).
     * @param bias_value The value to initialize biases (default: 0.0).
     */
    void initializeWeights(InitMethod method,
                           unsigned int seed = MANUAL_SEED,
                           double a = 0,
                           double b = 1.0,
                           double sparsity = 0.0,
                           double bias_value = 0.0);

    /**
     * @brief Initializes biases using the specified initialization method.
     *
     * This function initializes the bias values using one of the predefined methods.
     *
     * @param method Initialization method for biases (default: zero initialization).
     * @param seed Seed for random number generation (default: 21).
     * @param a Lower bound of the range for bias initialization (default: 0).
     * @param b Upper bound of the range for bias initialization (default: 1.0).
     * @param sparsity The sparsity for bias initialization (default: 0.0).
     * @param bias_value The value to initialize biases (default: 0.0).
     */
    void initializeBiases(InitMethod method,
                          unsigned int seed = MANUAL_SEED,
                          double a = 0,
                          double b = 1.0,
                          double sparsity = 0.0,
                          double bias_value = 0.0);

    /**
     * @brief Forward pass through the dense layer.
     *
     * Computes the output of the layer by applying the weights and biases to the input.
     * The output is the result of multiplying the input by the weight matrix and adding the bias vector.
     *
     * @param input A vector representing the input to the layer (size: input_size).
     * @return A vector representing the output of the layer (size: output_size).
     */
    std::vector<double> forward(const std::vector<double>& input) override;

    /**
     * @brief Backward pass to compute gradients of the loss with respect to the weights and biases.
     *
     * This function computes the gradients of the weights, biases, and the input (used for further backpropagation),
     * given the gradient of the loss with respect to the output. The gradients are then stored for later use.
     *
     * @param grad_output The gradient of the loss with respect to the output of the layer (size: output_size).
     * @param lr The learning rate used for gradient descent (default: 0.01).
     * @return The gradient of the loss with respect to the input (size: input_size).
     */
    std::vector<double> backward(const std::vector<double>& grad_output) override;

    /**
     * @brief Prints a summary of the layer's parameters.
     *
     * This function displays the number of input features, output neurons, and the shape of the weight matrix
     * and bias vector, as well as any other relevant information about the layer.
     */
    void summary() const override;

    /**
     * @brief Clears the gradients stored in the layer.
     *
     * This function resets the gradient vectors (grad_weights and grad_biases) to zero after the gradients have been
     * applied, so that new gradients can be computed in the next backward pass.
     */
    void clearGradients();

    /**
     * @brief Prints the weights of the layer.
     *
     * This function prints the weight matrix of the layer.
     */
    void printWeights() const;

    /**
     * @brief Prints the biases of the layer.
     *
     * This function prints the bias vector of the layer.
     */
    void printBiases() const;

    /**
     * @brief Returns the total number of parameters (weights + biases) in the layer.
     *
     * This function computes the total number of learnable parameters in the dense layer.
     * It includes both the weights and biases.
     *
     * @return The total number of parameters in the layer.
     */
    size_t getParameterCount() const;

    // Getters
    /**
     * @brief Gets the gradient of the weights.
     * 
     * @return A reference to the gradient of the weights (size: input_size x output_size).
     */
    const std::vector<std::vector<double>>& getGradWeights() const;

    /**
     * @brief Gets the gradient of the biases.
     * 
     * @return A reference to the gradient of the biases (size: output_size).
     */
    const std::vector<double>& getGradBiases() const;

    /**
     * @brief Gets the current weight matrix.
     * 
     * @return A reference to the weight matrix (size: input_size x output_size).
     */
    const std::vector<std::vector<double>>& getWeights() const;

    /**
     * @brief Gets the current bias vector.
     * 
     * @return A reference to the bias vector (size: output_size).
     */
    const std::vector<double>& getBiases() const;

    // Setters
    /**
     * @brief Sets the weights of the layer.
     *
     * This function allows manually setting the weight matrix to a new set of values.
     *
     * @param new_weights The new weight matrix to set (size: input_size x output_size).
     */
    void setWeights(const std::vector<std::vector<double>>& new_weights);

    /**
     * @brief Sets the biases of the layer.
     *
     * This function allows manually setting the bias vector to a new set of values.
     *
     * @param new_biases The new bias vector to set (size: output_size).
     */
    void setBiases(const std::vector<double>& new_biases);
};

#endif // DENSELAYER_H
