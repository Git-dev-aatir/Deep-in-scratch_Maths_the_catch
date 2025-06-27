#ifndef BASELAYER_H
#define BASELAYER_H

#include <vector>
#include <string>
#include <iostream>

/**
 * @brief Abstract base class representing a generic neural network layer.
 * 
 * Provides a common interface for all derived layer types like DenseLayer and ActivationLayer.
 */
class BaseLayer {
public:
    /**
     * @brief Performs the forward pass computation.
     * @param input Input vector for the layer.
     * @return Output vector after applying the layer transformation.
     */
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;

    /**
     * @brief Performs the backward pass computation (backpropagation).
     * @param grad_output Gradient vector from the next layer.
     * @param learning_rate Learning rate used for updating parameters (if applicable).
     * @return Gradient vector with respect to the input of this layer.
     */
    virtual std::vector<double> backward(const std::vector<double>& grad_output) = 0;

    /**
     * @brief Prints a summary of the layer.
     */
    virtual void summary() const = 0;

    /**
     * @brief Virtual destructor for proper cleanup.
     */
    virtual ~BaseLayer() {}
};

#endif // BASELAYER_H
