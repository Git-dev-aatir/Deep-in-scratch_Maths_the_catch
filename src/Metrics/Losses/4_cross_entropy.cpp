#include "Metrics/Losses.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace Losses {

static inline double clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

static inline std::vector<double> softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    
    std::vector<double> exps(logits.size());
    double max_logit = *std::max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }

    // Avoid division by zero
    if (sum < 1e-15) sum = 1e-15;
    
    for (auto& val : exps)
        val /= sum;
    
    return exps;
}

double cross_entropy_loss(const std::vector<double>& y_true, 
                          const std::vector<double>& y_pred, 
                          bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Cross Entropy: Size mismatch or empty vector.");
    
    const double eps = 1e-7;
    std::vector<double> probs = from_logits ? softmax(y_pred) : y_pred;

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double p = clamp(probs[i], eps, 1.0 - eps);
        loss -= y_true[i] * std::log(p);
    }

    return loss;  // Removed averaging by class count
}

std::vector<double> cross_entropy_derivative(const std::vector<double>& y_true, 
                                             const std::vector<double>& y_pred, 
                                             bool from_logits) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Cross Entropy Derivative: Size mismatch or empty vector.");
    
    const double eps = 1e-7;
    std::vector<double> grad(y_true.size());
    
    if (from_logits) {
        std::vector<double> probs = softmax(y_pred);
        for (size_t i = 0; i < y_true.size(); ++i) {
            grad[i] = probs[i] - y_true[i];     // No averaging on number of classes
        }
    } else {
        for (size_t i = 0; i < y_true.size(); ++i) {
            double p = clamp(y_pred[i], eps, 1.0 - eps);
            grad[i] = p - y_true[i];           // No averaging on number of classes
            // grad[i] = -y_true[i] / p; // if last layer is sigmoid
        }
    }
    return grad;
}

double cross_entropy_loss_batch(const std::vector<std::vector<double>>& y_true, 
                                const std::vector<std::vector<double>>& y_pred, 
                                bool from_logits) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Cross Entropy Batch: Size mismatch or empty batch.");
    
    double total_loss = 0.0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("Cross Entropy Batch: Size mismatch at index " + std::to_string(i));
        
        total_loss += cross_entropy_loss(y_true[i], y_pred[i], from_logits);
    }
    
    return total_loss / y_true.size();  // Average over batch size
}

std::vector<std::vector<double>> cross_entropy_derivative_batch(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred,
    bool from_logits) 
{
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Cross Entropy Derivative Batch: Size mismatch or empty batch.");
    
    std::vector<std::vector<double>> grads(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("Cross Entropy Derivative Batch: Size mismatch at index " + std::to_string(i));
        
        grads[i] = cross_entropy_derivative(y_true[i], y_pred[i], from_logits);
        for (auto& ele : grads[i]) ele /= y_true.size();
    }
    return grads;
}

} // namespace Losses
