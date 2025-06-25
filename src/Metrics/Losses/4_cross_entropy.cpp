/**
 * @file cross_entropy.cpp
 * @brief Implementation of Cross Entropy loss and its derivative.
 */

#include "../../../include/Metrics/Losses.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace Losses {

static inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

static inline std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exps(logits.size());
    double max_logit = *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }

    for (auto& val : exps)
        val /= sum;
    
    return exps;
}

double cross_entropy_loss(const std::vector<double>& y_true, 
                          const std::vector<double>& y_pred, 
                          bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Cross Entropy: Size mismatch or empty vector.");
    
    // Validate that y_true is one-hot encoded
    double sum_true = std::accumulate(y_true.begin(), y_true.end(), 0.0);
    if (std::abs(sum_true - 1.0) > 1e-6) {
        throw std::invalid_argument("Cross Entropy: y_true must be one-hot encoded (sum to 1).");
    }
    
    const double eps = 1e-7;
    std::vector<double> probs = from_logits ? softmax(y_pred) : y_pred;

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] > 0.0) {  // More flexible than checking exact equality
            double p = clamp(probs[i], eps, 1.0 - eps);
            loss -= std::log(p); // * y_true[i] -> Support soft targets
        }
    }

    return loss;
}

std::vector<double> cross_entropy_derivative(const std::vector<double>& y_true, 
                                             const std::vector<double>& y_pred, 
                                             bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Cross Entropy Derivative: Size mismatch or empty vector.");
    
    std::vector<double> grad(y_true.size());
    if (from_logits) {
        // For softmax + cross-entropy, derivative is simply: softmax_output - y_true
        std::vector<double> probs = softmax(y_pred);
        for (size_t i = 0; i < y_true.size(); ++i) {
            grad[i] = probs[i] - y_true[i];
        }
    } else {
        // For probability inputs, derivative is: -y_true / y_pred
        const double eps = 1e-7;
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = clamp(y_pred[i], eps, 1.0 - eps);
            grad[i] = -y_true[i] / p;
        }
    }
    return grad;
}

double cross_entropy_loss_batch(const std::vector<std::vector<double>>& y_true, 
                                const std::vector<std::vector<double>>& y_pred, 
                                bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Cross Entropy Batch: Size mismatch or empty batch.");
    double total = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        total += cross_entropy_loss(y_true[i], y_pred[i], from_logits);
    return total / y_true.size();
}

std::vector<std::vector<double>> cross_entropy_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                                const std::vector<std::vector<double>>& y_pred,
                                                                bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Cross Entropy Derivative Batch: Size mismatch or empty batch.");
    
    std::vector<std::vector<double>> grads(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grads[i] = cross_entropy_derivative(y_true[i], y_pred[i], from_logits);
        // Note: Don't divide by batch size here - let the optimizer handle averaging
        // If you want to average gradients, do it at the optimizer level
    }
    
    return grads;
}

}
