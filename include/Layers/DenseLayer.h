#pragma once 

#include "BaseLayer.h"
#include "../Utils/Initialization.h"
#include <cstddef>
#include <vector>

/**
 * @class DenseLayer
 * @brief Fully connected neural network layer with configurable initialization
 * 
 * Implements a dense (fully connected) layer with forward/backward propagation,
 * gradient storage, and multiple weight initialization strategies. Supports:
 * - Various weight initialization methods (Xavier, He, etc.)
 * - Batch and stochastic gradient descent
 * - Activation-agnostic design (pair with activation layers)
 */
class DenseLayer : public BaseLayer {
private:
    size_t input_size;                          ///< Number of input features
    size_t output_size;                         ///< Number of output neurons
    std::vector<std::vector<double>> weights;   ///< Weight matrix [input_size x output_size]
    std::vector<double> biases;                 ///< Bias vector [output_size]
    std::vector<std::vector<double>> grad_weights; ///< Weight gradients
    std::vector<double> grad_biases;            ///< Bias gradients
    std::vector<double> input_cache;            ///< Cached inputs for backpropagation

public:
    /**
     * @brief Constructs a dense layer
     * @param in_features Input dimension
     * @param out_features Output dimension
     * @param init_params Whether to initialize parameters (default=true)
     */
    DenseLayer(size_t in_features, size_t out_features, bool init_params = false);

    virtual ~DenseLayer() = default;  ///< Virtual destructor for polymorphism

    /**
     * @brief Initializes weights using specified method
     * @param method Initialization strategy
     * @param seed RNG seed (default=DEFAULT_SEED)
     * @param a Lower bound for uniform distribution
     * @param b Upper bound for uniform distribution
     * @param sparsity Fraction of weights to set to zero
     */
    void initializeWeights(
        InitMethod method,
        unsigned int seed,
        double a = 0,
        double b = 1.0,
        double sparsity = 0.0,
        double constant_value = 0.0
    );

    /**
     * @brief Initializes biases using specified method
     * @param method Initialization strategy
     * @param seed RNG seed (default=DEFAULT_SEED)
     * @param constant_value Value for constant initialization
     * @param a Lower bound for uniform distribution
     * @param b Upper bound for uniform distribution
     * @param sparsity Fraction of biases to set to zero
     */
    void initializeBiases(
        InitMethod method,
        unsigned int seed,
        double a = 0,
        double b = 1.0,
        double sparsity = 0.0,
        double constant_value = 0.0
    );

////////////////////
// Core operations//
////////////////////

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
    
//////////////////////
// Utility functions//
//////////////////////

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
     * @brief Returns the total number of parameters (weights + biases) in the layer.
     *
     * This function computes the total number of learnable parameters in the dense layer.
     * It includes both the weights and biases.
     *
     * @return The total number of parameters in the layer.
     */
    size_t getParameterCount() const;

//////////////
// Debugging//
//////////////

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

//////////////
// Accessors//
//////////////

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

/////////////
// Mutators//
/////////////
    
    /**
     * @brief Sets the weights of the layer.
     *
     * This function allows manually setting the weight matrix to a new set of values.
     *
     * @param new_weights The new weight matrix to set (size: input_size x output_size).
     */
    void setWeights(const std::vector<std::vector<double>>& new_weights); // copy

    void setWeights(const std::vector<std::vector<double>>&& new_weights); // move

    /**
     * @brief Sets the biases of the layer.
     *
     * This function allows manually setting the bias vector to a new set of values.
     *
     * @param new_biases The new bias vector to set (size: output_size).
     */
    void setBiases(const std::vector<double>& new_biases); // copy 

    void setBiases(const std::vector<double>&& new_biases); // move
};
