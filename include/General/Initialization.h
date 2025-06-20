#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

#define MANUAL_SEED 21

using std::vector;

double clamp (double val, double lo, double hi) {
    if (val < lo) return lo;
    else if (val > hi) return hi;
    else return val;
}

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
    ORTHOGONAL,   // For square weight matrices only
    BIAS          // Initialize bias to a constant value
};

/**
 * @brief Initializes a 2D parameters matrix according to the specified method.
 * 
 * @param in_features Number of input units.
 * @param out_features Number of output units.
 * @param method The initialization method to use (Radon, Xavier, He, LeCun, Orthogonal, Bias).
 * @param seed Random seed for reproducibility.
 * @param sparsity Fraction [0,1] of parameters to be set to zero (applies to all methods).
 * @param bias_value For BIAS initialization only: constant value to set.
 * @param a lower bound for UNIFORM, or mean for NORMAL distribution.
 * @param b upper bound for UNIFORM, or variance for NORMAL distribution.
 * @return Initialized parameters matrix.
 */
vector<vector<double>> initializeParameters(size_t in_features,
                       size_t out_features,
                       InitMethod method,
                       unsigned int seed = MANUAL_SEED,
                       double a = 0,
                       double b = 1.0,
                       double sparsity = 0.0,
                       double bias_value = 0.0) {

    assert(in_features > 0 && out_features > 0);
    assert(sparsity >= 0.0 && sparsity <= 1.0);

    vector<vector<double>> weights(out_features, vector<double>(in_features, 0.0));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni_dist(a, b);
    std::normal_distribution<double> norm_dist(a, b);
    std::uniform_real_distribution<double> sparsity_dist(0.0, 1.0);

    double scale = 1.0;

    switch (method) {
        case InitMethod::RANDOM_UNIFORM:
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = uni_dist(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::RANDOM_NORMAL:
            for (auto& row : weights) {
                for (auto& val : row) {
                    double z = norm_dist(rng);
                    val =clamp(z, a, b); // clamp to range
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::XAVIER_UNIFORM:
            scale = sqrt(6.0 / (in_features + out_features));
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::XAVIER_NORMAL:
            scale = sqrt(2.0 / (in_features + out_features));
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::HE_UNIFORM:
            scale = sqrt(6.0 / (in_features));
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::HE_NORMAL:
            scale = sqrt(2.0 / in_features);
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;
            
        case InitMethod::LECUN_UNIFORM:
            scale = sqrt(3.0 / in_features);
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::LECUN_NORMAL:
            scale = sqrt(1.0 / in_features);
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;

        case InitMethod::ORTHOGONAL:
            if (in_features != out_features) {
                std::cerr << "Orthogonal init requires square matrix.\n";
                return weights;
            } else {
                // Gram-Schmidt orthogonalization
                for (size_t i = 0; i < out_features; ++i) {
                    for (size_t j = 0; j < in_features; ++j)
                        weights[i][j] = norm_dist(rng);

                    // Orthogonalize against previous rows
                    for (size_t k = 0; k < i; ++k) {
                        double dot = 0.0;
                        for (size_t j = 0; j < in_features; ++j)
                            dot += weights[i][j] * weights[k][j];
                        for (size_t j = 0; j < in_features; ++j)
                            weights[i][j] -= dot * weights[k][j];
                    }

                    // Normalize
                    double norm = 0.0;
                    for (double v : weights[i]) norm += v * v;
                    norm = sqrt(norm);
                    for (auto& v : weights[i]) v /= norm;
                }

                // Apply sparsity after orthogonalization
                for (auto& row : weights) {
                    for (auto& val : row) {
                        if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                    }
                }
            }
            break;

        case InitMethod::BIAS:
            for (auto& row : weights) {
                for (auto& val : row) {
                    val = bias_value;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            }
            break;
    }

    return weights;
}

#endif // INITIALIZATION_H
