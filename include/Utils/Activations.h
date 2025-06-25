#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

/**
 * @file Activations.h
 * @brief Declaration of commonly used activation functions (Sigmoid, Softmax) 
 *        for neural networks.
 *
 * This header provides implementations of activation functions used 
 * across different layers of the network. Functions include both scalar 
 * and vectorized versions where applicable.
 */

namespace Activations {

    /**
     * @brief Computes the sigmoid activation function for a single scalar input.
     * 
     * The sigmoid function is defined as:
     * \f[
     * \sigma(x) = \frac{1}{1 + e^{-x}}
     * \f]
     *
     * @param x Input value (double).
     * @return The sigmoid of the input value.
     */
    double sigmoid(double x);

    /**
     * @brief Computes the element-wise sigmoid activation for a vector.
     * 
     * Applies the sigmoid function to each element in the input vector independently.
     *
     * @param x Input vector of real numbers.
     * @return A vector where each element is the sigmoid of the corresponding input element.
     */
    std::vector<double> sigmoid(const std::vector<double>& x);

    /**
     * @brief Computes the softmax activation function for a vector.
     * 
     * The softmax function transforms a vector of real numbers into a probability distribution.
     * Numerically stable computation using the max-shift technique is employed:
     * 
     * \f[
     * \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
     * \f]
     *
     * @param x Input vector of real numbers.
     * @return A vector representing the probability distribution obtained from the input.
     *         All elements are in the range (0,1) and sum to 1.
     */
    std::vector<double> softmax(const std::vector<double>& x);

} // namespace Activations

#endif // ACTIVATIONS_H
