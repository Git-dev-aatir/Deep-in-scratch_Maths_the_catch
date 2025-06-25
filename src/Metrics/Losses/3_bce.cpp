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
    const double eps = 1e-12;
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double p = from_logits ? sigmoid(y_pred[i]) : y_pred[i];
        p = clamp(p, eps, 1.0 - eps);
        loss += y_true[i] * std::log(p) + (1.0 - y_true[i]) * std::log(1.0 - p);
    }
    return loss / y_true.size();
}

std::vector<double> bce_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred, bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("BCE Derivative: Size mismatch or empty vector.");
    const double eps = 1e-12;
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        double p = from_logits ? sigmoid(y_pred[i]) : y_pred[i];
        p = clamp(p, eps, 1.0 - eps);
        grad[i] = (p - y_true[i]) / y_true.size();
        if (from_logits == false) grad[i] /= p * (1 - p);
    }
    return grad;
}

double bce_loss_batch(const std::vector<std::vector<double>>& y_true, 
                      const std::vector<std::vector<double>>& y_pred, 
                      bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("BCE Batch: Size mismatch or empty batch.");
    double total = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        total += bce_loss(y_true[i], y_pred[i], from_logits);
    return total / y_true.size();
}

std::vector<std::vector<double>> bce_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                      const std::vector<std::vector<double>>& y_pred,
                                                      bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("BCE Derivative Batch: Size mismatch or empty batch.");
    std::vector<std::vector<double>> grads(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grads[i] = bce_derivative(y_true[i], y_pred[i], from_logits);
        for (size_t j=0; j < y_true[i].size(); ++j) 
            grads[i][j] /= y_true.size();
    }
    return grads;
}

}
