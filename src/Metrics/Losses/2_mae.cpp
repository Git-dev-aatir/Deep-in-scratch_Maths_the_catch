/**
 * @file mae.cpp
 * @brief Implementation of Mean Absolute Error (MAE) loss and its derivative.
 */

#include "../../../include/Metrics/Losses.h"
#include <stdexcept>
#include <cmath>

namespace Losses {

double mae_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MAE: Size mismatch or empty vector.");
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i)
        sum += std::abs(y_true[i] - y_pred[i]);
    return sum / (y_true.size());
}

std::vector<double> mae_derivative(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty())
        throw std::invalid_argument("MAE Derivative: Size mismatch or empty vector.");
    std::vector<double> grad(y_true.size());
    for (size_t i = 0; i < y_true.size(); ++i) {
        grad[i] = (y_pred[i] > y_true[i]) ? 1.0 : (y_pred[i] < y_true[i]) ? -1.0 : 0.0;
        grad[i] /= y_true.size();
    }
    return grad;
}

double mae_loss_batch(const std::vector<std::vector<double>>& y_true, const std::vector<std::vector<double>>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MAE Batch: Size mismatch or empty batch.");
    double total = 0.0;
    for(size_t i = 0; i < y_true.size(); ++i) {
        if(y_true[i].empty() || y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("MAE Batch: Size mismatch or empty vector at '" + std::to_string(i) + "' index.");
        for(size_t j = 0; j < y_true[i].size(); ++j)
            total += std::abs(y_true[i][j] - y_pred[i][j]);
    }
    return total / (2 * y_true.size() * y_true[0].size());
}

std::vector<std::vector<double>> mae_derivative_batch(const std::vector<std::vector<double>>& y_true,
                                                      const std::vector<std::vector<double>>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("MAE Derivative Batch: Size mismatch or empty batch.");
    std::vector<std::vector<double>> grads(y_true.size());
    for(size_t i = 0; i < y_true.size(); ++i) {
        if(y_true[i].empty() || y_true[i].size() != y_pred[i].size())
            throw std::invalid_argument("MSE Derivative Batch: Size mismatch or empty vector at '" + std::to_string(i) + "' index.");
        std::vector<double> grad_i(y_true.size());
        for(size_t j = 0; j < y_true[j].size(); ++j) {
            grad_i[j] = (y_pred[i] > y_true[i]) ? 1.0 : (y_pred[i] < y_true[i]) ? -1.0 : 0.0;
            grad_i[j] /= y_pred[i].size();
        }   
        grads[i] = grad_i;
    }
    return grads;
}

}
