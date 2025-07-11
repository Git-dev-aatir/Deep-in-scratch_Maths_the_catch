/**
 * @file hinge.cpp
 * @brief Implementation of Hinge (SVM) loss and its derivative.
 */

#include "../../../include/Metrics/Losses.h"
#include <stdexcept>
#include <cmath>

namespace Losses {

double hinge_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Hinge Loss: Size mismatch or empty vector.");
    
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double margin = 1.0 - y_true[i] * y_pred[i];
        if (margin > 0.0) loss += margin;
    }
    return loss / y_true.size();
}

std::vector<double> hinge_loss_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("Hinge Derivative: Size mismatch or empty vector.");
    
    std::vector<double> grad(y_true.size(), 0.0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        double margin = 1.0 - y_true[i] * y_pred[i];
        if (margin > 0.0) grad[i] = -y_true[i] / y_true.size();
    }
    return grad;
}

double hinge_loss_batch(const std::vector<std::vector<double>>& y_true, 
                        const std::vector<std::vector<double>>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Hinge Batch: Size mismatch or empty batch.");
    
    double total_loss = 0.0;
    size_t total_elements = 0;
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("Hinge Batch: Size mismatch at index " + std::to_string(i));
        
        total_loss += hinge_loss(y_true[i], y_pred[i]) * y_true[i].size();
        total_elements += y_true[i].size();
    }
    
    return total_loss / total_elements;
}

std::vector<std::vector<double>> hinge_loss_derivative_batch(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred) 
{
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("Hinge Derivative Batch: Size mismatch or empty batch.");
    
    std::vector<std::vector<double>> grads(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("Hinge Derivative Batch: Size mismatch at index " + std::to_string(i));
        
        grads[i] = hinge_loss_derivative(y_true[i], y_pred[i]);
    }
    return grads;
}

} // namespace Losses
