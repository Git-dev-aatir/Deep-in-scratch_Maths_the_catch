#include "../../include/Utils/Initialization.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <vector>

double clamp(double val, double lo, double hi) {
    return (val < lo) ? lo : (hi < val) ? hi : val;
}

std::vector<std::vector<double>> initializeParameters(
    size_t in_features,
    size_t out_features,
    InitMethod method,
    unsigned int seed,
    double a,
    double b,
    double sparsity,
    double bias_value)
{
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Input/output features must be > 0");
    }
    sparsity = clamp(sparsity, 0.0, 1.0);

    std::vector<std::vector<double>> parameters(out_features, std::vector<double>(in_features));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni_dist(0.0, 1.0); // For sparsity
    std::uniform_real_distribution<double> sparsity_dist(0.0, 1.0);

    switch (method) {
        case InitMethod::RANDOM_UNIFORM: {
            std::uniform_real_distribution<double> dist(a, b);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::RANDOM_NORMAL: {
            std::normal_distribution<double> dist(a, b);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::XAVIER_UNIFORM: {
            double limit = std::sqrt(6.0 / (in_features + out_features));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::XAVIER_NORMAL: {
            double stddev = std::sqrt(2.0 / (in_features + out_features));
            std::normal_distribution<double> dist(0.0, stddev);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::HE_UNIFORM: {
            double limit = std::sqrt(6.0 / in_features);
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::HE_NORMAL: {
            double stddev = std::sqrt(2.0 / in_features);
            std::normal_distribution<double> dist(0.0, stddev);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::LECUN_UNIFORM: {
            double limit = std::sqrt(3.0 / in_features);
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::LECUN_NORMAL: {
            double stddev = std::sqrt(1.0 / in_features);
            std::normal_distribution<double> dist(0.0, stddev);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            break;
        }
        case InitMethod::ORTHOGONAL: {
            if (in_features != out_features) {
                throw std::invalid_argument("Orthogonal init requires square matrix");
            }
            // Generate random matrix
            std::normal_distribution<double> dist(0.0, 1.0);
            for (auto& row : parameters)
                for (auto& val : row)
                    val = dist(rng);
            
            // Modified Gram-Schmidt
            for (size_t j = 0; j < in_features; ++j) {
                // Normalize j-th column
                double norm = 0.0;
                for (size_t i = 0; i < out_features; ++i)
                    norm += parameters[i][j] * parameters[i][j];
                norm = std::sqrt(norm);
                
                if (norm < 1e-10) continue;
                
                for (size_t i = 0; i < out_features; ++i)
                    parameters[i][j] /= norm;
                
                // Orthogonalize subsequent columns
                for (size_t k = j + 1; k < in_features; ++k) {
                    double dot = 0.0;
                    for (size_t i = 0; i < out_features; ++i)
                        dot += parameters[i][j] * parameters[i][k];
                    
                    for (size_t i = 0; i < out_features; ++i)
                        parameters[i][k] -= dot * parameters[i][j];
                }
            }
            break;
        }
        case InitMethod::BIAS: {
            for (auto& row : parameters)
                std::fill(row.begin(), row.end(), bias_value);
            break;
        }
        default:
            throw std::invalid_argument("Unsupported initialization method");
    }

    // Apply sparsity
    if (sparsity > 0.0) {
        for (auto& row : parameters) {
            for (auto& val : row) {
                if (sparsity_dist(rng) < sparsity) {
                    val = 0.0;
                }
            }
        }
    }

    return parameters;
}
