#pragma once

#include <vector>
#include <cstddef>

/**
 * @brief Enumeration for various weight initialization methods.
 */
enum class InitMethod {
    RANDOM_UNIFORM,
    RANDOM_NORMAL,
    XAVIER_UNIFORM,
    XAVIER_NORMAL,
    HE_UNIFORM,
    HE_NORMAL,
    LECUN_UNIFORM,
    LECUN_NORMAL,
    ORTHOGONAL,   ///< For square weight matrices only.
    BIAS          ///< Initialize bias to a constant value.
};

/**
 * @brief Clamp a value between a lower and upper bound.
 * 
 * @param val The value to clamp.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return Clamped value.
 */
double clamp(double val, double lo, double hi);

/**
 * @brief Initializes a 2D parameters matrix using the specified initialization method.
 * 
 * @param in_features Number of input features (columns).
 * @param out_features Number of output features (rows).
 * @param method Initialization method to apply.
 * @param seed Random seed (default = 21).
 * @param a Lower bound (for uniform) or mean (for normal).
 * @param b Upper bound (for uniform) or std deviation (for normal).
 * @param sparsity Fraction [0, 1] of weights to set to zero.
 * @param bias_value Constant value for bias initialization.
 * @return Initialized parameters matrix.
 */
std::vector<std::vector<double>> initializeParameters(
    size_t in_features,
    size_t out_features,
    InitMethod method,
    unsigned int seed = 21,
    double a = 0.0,
    double b = 1.0,
    double sparsity = 0.0,
    double bias_value = 0.0
);
