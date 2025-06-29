/**
 * @file mse.cpp
 * @brief Implementation of Mean Squared Error (MSE) loss and its derivative.
 */

#include "../../../include/Metrics/Losses.h"
#include <stdexcept>
#include <cmath>

namespace Losses {

double mse_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MSE: Size mismatch or empty vector.");
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        sum += std::pow(y_true[i] - y_pred[i], 2);
    return sum / (2 * y_true.size());
}

std::vector<double> mse_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MSE Derivative: Size mismatch or empty vector.");
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i)
        grad[i] = (y_pred[i] - y_true[i]) / y_true.size() ;
    return grad;
}

double mse_loss_batch(const std::vector<std::vector<double>>& y_true, 
                      const std::vector<std::vector<double>>& y_pred) {
    if(y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MSE Batch: Size mismatch or empty batch.");
    
    size_t total_elements = 0;
    double total = 0.0;
    
    for(size_t i = 0; i < y_true.size(); ++i) {
        if(y_true[i].empty() || y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("MSE Batch: Size mismatch at index " + std::to_string(i));
        
        total_elements += y_true[i].size();
        
        for(size_t j = 0; j < y_true[i].size(); ++j)
            total += std::pow(y_true[i][j] - y_pred[i][j], 2);
    }
    
    return total / (2 * total_elements);  
}

std::vector<std::vector<double>> mse_derivative_batch(
    const std::vector<std::vector<double>>& y_true, 
    const std::vector<std::vector<double>>& y_pred) 
{
    if(y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MSE Derivative Batch: Size mismatch or empty batch.");
    
    // Compute total elements FIRST
    size_t total_elements = 0;
    for (const auto& vec : y_true) {
        total_elements += vec.size();
    }
    
    std::vector<std::vector<double>> grads(y_true.size());
    
    for(size_t i = 0; i < y_true.size(); ++i) {
        if(y_true[i].empty() || y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("MSE Derivative Batch: Size mismatch at index " + std::to_string(i));
        
        std::vector<double> grad_i(y_true[i].size());
        
        for(size_t j = 0; j < y_true[i].size(); ++j) {
            grad_i[j] = (y_pred[i][j] - y_true[i][j]) / total_elements;
        }
        
        grads[i] = grad_i;
    }
    return grads;
}

} // namespace Losses
