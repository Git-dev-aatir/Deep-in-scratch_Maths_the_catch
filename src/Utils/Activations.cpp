#include "Utils/Activations.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace Activations {

// Scalar implementations
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double relu(double x) { return (x > 0) ? x : 0; }
double tanh(double x) { return std::tanh(x); }
double softplus(double x) { return std::log(1 + std::exp(x)); }

// Vector implementations
std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(sigmoid(val));
    return result;
}

std::vector<double> relu(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(relu(val));
    return result;
}

std::vector<double> tanh(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(tanh(val));
    return result;
}

std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) throw std::invalid_argument("softmax: Input vector cannot be empty");
    
    double max_elem = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    std::vector<double> exp_vals;
    exp_vals.reserve(x.size());
    
    for (double val : x) {
        double exp_val = std::exp(val - max_elem);
        exp_vals.push_back(exp_val);
        sum += exp_val;
    }
    
    if (sum < 1e-15) return std::vector<double>(x.size(), 1.0/x.size());
    
    std::vector<double> result;
    result.reserve(x.size());
    for (double ev : exp_vals) result.push_back(ev / sum);
    return result;
}

// Batch implementations
std::vector<std::vector<double>> sigmoid_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(sigmoid(vec));
    return result;
}

std::vector<std::vector<double>> relu_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(relu(vec));
    return result;
}

std::vector<std::vector<double>> tanh_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(tanh(vec));
    return result;
}

std::vector<std::vector<double>> softmax_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(softmax(vec));
    return result;
}

// Derivative implementations
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1 - t*t;
}

std::vector<double> sigmoid_derivative(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(sigmoid_derivative(val));
    return result;
}

std::vector<double> relu_derivative(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(relu_derivative(val));
    return result;
}

std::vector<double> tanh_derivative(const std::vector<double>& x) {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x) result.push_back(tanh_derivative(val));
    return result;
}

// Batch derivatives
std::vector<std::vector<double>> sigmoid_derivative_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(sigmoid_derivative(vec));
    return result;
}

std::vector<std::vector<double>> relu_derivative_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(relu_derivative(vec));
    return result;
}

std::vector<std::vector<double>> tanh_derivative_batch(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> result;
    result.reserve(x.size());
    for (const auto& vec : x) result.push_back(tanh_derivative(vec));
    return result;
}

} // namespace Activations
