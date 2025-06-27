#pragma once

#include <vector>

/**
 * @file Activations.h
 * @brief Declaration of commonly used activation functions and their derivatives for neural networks.
 *
 * Provides scalar, vector, and batch versions for each activation and its derivative.
 */

namespace Activations {

    /**
     * @name Scalar Activations
     * @{
     */

    /**
     * @brief Computes the sigmoid activation for a scalar input.
     * @param x Input value.
     * @return Sigmoid of x.
     */
    double sigmoid(double x);

    /**
     * @brief Computes the ReLU activation for a scalar input.
     * @param x Input value.
     * @return ReLU of x.
     */
    double relu(double x);

    /**
     * @brief Computes the tanh activation for a scalar input.
     * @param x Input value.
     * @return Tanh of x.
     */
    double tanh(double x);

    /**
     * @brief Computes the softplus activation for a scalar input.
     * @param x Input value.
     * @return Softplus of x.
     */
    double softplus(double x);

    /** @} */

    /**
     * @name Vector Activations
     * @{
     */

    /**
     * @brief Computes the element-wise sigmoid activation for a vector.
     * @param x Input vector.
     * @return Vector where each element is the sigmoid of the corresponding input.
     */
    std::vector<double> sigmoid(const std::vector<double>& x);

    /**
     * @brief Computes the element-wise ReLU activation for a vector.
     * @param x Input vector.
     * @return Vector where each element is the ReLU of the corresponding input.
     */
    std::vector<double> relu(const std::vector<double>& x);

    /**
     * @brief Computes the element-wise tanh activation for a vector.
     * @param x Input vector.
     * @return Vector where each element is the tanh of the corresponding input.
     */
    std::vector<double> tanh(const std::vector<double>& x);

    /**
     * @brief Computes the softmax activation for a vector (probability distribution).
     * Uses the numerically stable max-shift technique.
     * @param x Input vector.
     * @return Vector representing the softmax probabilities (sum to 1).
     */
    std::vector<double> softmax(const std::vector<double>& x);

    /** @} */

    /**
     * @name Batch Activations
     * @{
     */

    /**
     * @brief Computes the element-wise sigmoid activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch where each vector is element-wise sigmoid of the input.
     */
    std::vector<std::vector<double>> sigmoid_batch(const std::vector<std::vector<double>>& x);

    /**
     * @brief Computes the element-wise ReLU activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch where each vector is element-wise ReLU of the input.
     */
    std::vector<std::vector<double>> relu_batch(const std::vector<std::vector<double>>& x);

    /**
     * @brief Computes the element-wise tanh activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch where each vector is element-wise tanh of the input.
     */
    std::vector<std::vector<double>> tanh_batch(const std::vector<std::vector<double>>& x);

    /**
     * @brief Computes the softmax activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch where each vector is the softmax probabilities of the input.
     */
    std::vector<std::vector<double>> softmax_batch(const std::vector<std::vector<double>>& x);

    /** @} */

    /**
     * @name Scalar Derivatives
     * @{
     */

    /**
     * @brief Computes the derivative of the sigmoid activation for a scalar input.
     * @param x Input value.
     * @return Derivative of sigmoid at x.
     */
    double sigmoid_derivative(double x);

    /**
     * @brief Computes the derivative of the ReLU activation for a scalar input.
     * @param x Input value.
     * @return Derivative of ReLU at x.
     */
    double relu_derivative(double x);

    /**
     * @brief Computes the derivative of the tanh activation for a scalar input.
     * @param x Input value.
     * @return Derivative of tanh at x.
     */
    double tanh_derivative(double x);

    /** @} */

    /**
     * @name Vector Derivatives
     * @{
     */

    /**
     * @brief Computes the element-wise derivative of the sigmoid activation for a vector.
     * @param x Input vector.
     * @return Vector of sigmoid derivatives.
     */
    std::vector<double> sigmoid_derivative(const std::vector<double>& x);

    /**
     * @brief Computes the element-wise derivative of the ReLU activation for a vector.
     * @param x Input vector.
     * @return Vector of ReLU derivatives.
     */
    std::vector<double> relu_derivative(const std::vector<double>& x);

    /**
     * @brief Computes the element-wise derivative of the tanh activation for a vector.
     * @param x Input vector.
     * @return Vector of tanh derivatives.
     */
    std::vector<double> tanh_derivative(const std::vector<double>& x);

    /** @} */

    /**
     * @name Batch Derivatives
     * @{
     */

    /**
     * @brief Computes the element-wise derivative of the sigmoid activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch of sigmoid derivatives.
     */
    std::vector<std::vector<double>> sigmoid_derivative_batch(const std::vector<std::vector<double>>& x);

    /**
     * @brief Computes the element-wise derivative of the ReLU activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch of ReLU derivatives.
     */
    std::vector<std::vector<double>> relu_derivative_batch(const std::vector<std::vector<double>>& x);

    /**
     * @brief Computes the element-wise derivative of the tanh activation for a batch of vectors.
     * @param x Batch of input vectors.
     * @return Batch of tanh derivatives.
     */
    std::vector<std::vector<double>> tanh_derivative_batch(const std::vector<std::vector<double>>& x);

    /** @} */

} // namespace Activations
