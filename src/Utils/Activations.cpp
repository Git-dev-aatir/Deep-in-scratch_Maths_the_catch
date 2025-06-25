#include "../../include/Utils/Activations.h"
#include <cmath>
#include <algorithm>

namespace Activations {

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max_elem = *std::max_element(x.begin(), x.end());

    double sum = 0.0;
    for (double val : x) sum += std::exp(val - max_elem);
    for (size_t i = 0; i < x.size(); ++i)
        result[i] = std::exp(x[i] - max_elem) / sum;

    return result;
}

} // namespace Activations
