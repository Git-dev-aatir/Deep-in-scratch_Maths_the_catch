#include "../../include/Utils/Initialization.h"
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>

double clamp(double val, double lo, double hi) {
    if (val < lo) return lo;
    else if (val > hi) return hi;
    else return val;
}

std::vector<std::vector<double>> initializeParameters(
    size_t in_features,
    size_t out_features,
    InitMethod method,
    unsigned int seed,
    double a,
    double b,
    double sparsity,
    double bias_value
) {
    assert(in_features > 0 && out_features > 0);
    assert(sparsity >= 0.0 && sparsity <= 1.0);

    std::vector<std::vector<double>> weights(out_features, std::vector<double>(in_features, 0.0));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni_dist(a, b);
    std::normal_distribution<double> norm_dist(a, b);
    std::uniform_real_distribution<double> sparsity_dist(0.0, 1.0);

    double scale = 1.0;

    switch (method) {
        case InitMethod::RANDOM_UNIFORM:
            for (auto& row : weights)
                for (auto& val : row) {
                    val = uni_dist(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::RANDOM_NORMAL:
            for (auto& row : weights)
                for (auto& val : row) {
                    double z = norm_dist(rng);
                    val = clamp(z, a, b);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::XAVIER_UNIFORM:
            scale = sqrt(6.0 / (in_features + out_features));
            for (auto& row : weights)
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::XAVIER_NORMAL:
            scale = sqrt(2.0 / (in_features + out_features));
            for (auto& row : weights)
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::HE_UNIFORM:
            scale = sqrt(6.0 / in_features);
            for (auto& row : weights)
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::HE_NORMAL:
            scale = sqrt(2.0 / in_features);
            for (auto& row : weights)
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::LECUN_UNIFORM:
            scale = sqrt(3.0 / in_features);
            for (auto& row : weights)
                for (auto& val : row) {
                    val = std::uniform_real_distribution<double>(-scale, scale)(rng);
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::LECUN_NORMAL:
            scale = sqrt(1.0 / in_features);
            for (auto& row : weights)
                for (auto& val : row) {
                    val = norm_dist(rng) * scale;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;

        case InitMethod::ORTHOGONAL:
            if (in_features != out_features) {
                std::cerr << "Orthogonal init requires square matrix.\n";
                return weights;
            } else {
                // Gram-Schmidt Orthogonalization
                for (size_t i = 0; i < out_features; ++i) {
                    for (size_t j = 0; j < in_features; ++j)
                        weights[i][j] = norm_dist(rng);

                    for (size_t k = 0; k < i; ++k) {
                        double dot = 0.0;
                        for (size_t j = 0; j < in_features; ++j)
                            dot += weights[i][j] * weights[k][j];
                        for (size_t j = 0; j < in_features; ++j)
                            weights[i][j] -= dot * weights[k][j];
                    }

                    double norm = 0.0;
                    for (double v : weights[i]) norm += v * v;
                    norm = sqrt(norm);
                    for (auto& v : weights[i]) v /= norm;
                }

                for (auto& row : weights)
                    for (auto& val : row)
                        if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
            }
            break;

        case InitMethod::BIAS:
            for (auto& row : weights)
                for (auto& val : row) {
                    val = bias_value;
                    if (sparsity && sparsity_dist(rng) < sparsity) val = 0.0;
                }
            break;
    }

    return weights;
}
