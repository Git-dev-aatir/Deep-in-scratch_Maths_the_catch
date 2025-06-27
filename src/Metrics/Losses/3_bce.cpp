/**
 * @file bce.cpp
 * @brief Implementation of Binary Cross Entropy (BCE) loss and its derivative.
 */

#include "../../../include/Metrics/Losses.h"
#include <stdexcept>
#include <cmath>

namespace Losses {

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

double bce_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("BCE: Size mismatch or empty vector.");
    
    const double eps = 1e-7;  // Standard epsilon value
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double p = from_logits ? sigmoid(y_pred[i]) : y_pred[i];
        p = clamp(p, eps, 1.0 - eps);
        loss -= (y_true[i] * std::log(p) + (1.0 - y_true[i]) * std::log(1.0 - p));
    }
    return loss / y_true.size();
}

std::vector<double> bce_derivative(const std::vector<double>& y_true, 
                                   const std::vector<double>& y_pred, 
                                   bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("BCE Derivative: Size mismatch or empty vector.");

    const double eps = 1e-7;
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        double p = from_logits ? sigmoid(y_pred[i]) : y_pred[i];
        p = clamp(p, eps, 1.0 - eps);

        if (from_logits) {
            // Derivative for logits: (p - y_true)
            grad[i] = (p - y_true[i]) / y_true.size();
        } else {
            // Derivative for probabilities: (p - y_true) / (p * (1 - p))
            grad[i] = (p - y_true[i]) / (p * (1 - p) * y_true.size());
        }
    }
    return grad;
}

double bce_loss_batch(const std::vector<std::vector<double>>& y_true, 
                      const std::vector<std::vector<double>>& y_pred, 
                      bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("BCE Batch: Size mismatch or empty batch.");
        
    double total_loss = 0.0;
    size_t total_elements = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size() || y_true[i].empty())
            throw std::invalid_argument("BCE Batch: Size mismatch at index " + std::to_string(i));
        
        total_loss += bce_loss(y_true[i], y_pred[i], from_logits) * y_true[i].size();
        total_elements += y_true[i].size();
    }
    
    return total_loss / total_elements;
}

std::vector<std::vector<double>> bce_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                      const std::vector<std::vector<double>>& y_pred,
                                                      bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("BCE Derivative Batch: Size mismatch or empty batch.");
    
    std::vector<std::vector<double>> grads(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size() || y_true[i].empty())
            throw std::invalid_argument("BCE Derivative Batch: Size mismatch at index " + std::to_string(i));
        
        // Use per-sample derivative without additional scaling
        grads[i] = bce_derivative(y_true[i], y_pred[i], from_logits);
    }
    return grads;
}

}
