#include "../../include/Layers/Activation_utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace std;

vector<double> applyActivation(const vector<double>& x, ActivationType act_type,
                               double alpha, double lambda) {
    if (x.empty()) return {};
    
    vector<double> result;
    result.reserve(x.size());
    
    switch (act_type) {
        case ActivationType::RELU:
            for (double xi : x) result.push_back(max(0.0, xi));
            break;
            
        case ActivationType::LEAKY_RELU:
            for (double xi : x) result.push_back((xi > 0) ? xi : alpha * xi);
            break;
            
        case ActivationType::SIGMOID:
            for (double xi : x) result.push_back(1.0 / (1.0 + exp(-xi)));
            break;
            
        case ActivationType::TANH:
            for (double xi : x) result.push_back(tanh(xi));
            break;
            
        case ActivationType::LINEAR:
            result = x;
            break;
            
        case ActivationType::SOFTMAX: {
            double max_elem = *max_element(x.begin(), x.end());
            double sum = 0.0;
            std::vector<double> exps;
            exps.reserve(x.size());
            
            for (double xi : x) {
                double exp_val = exp(xi - max_elem);
                exps.push_back(exp_val);
                sum += exp_val;
            }
            
            // Handle near-zero sum case
            if (sum < 1e-15) {
                double uniform = 1.0 / x.size();
                result = vector<double>(x.size(), uniform);
            } else {
                for (double e : exps) result.push_back(e / sum);
            }
            break;
        }
            
        case ActivationType::SELU:
            for (double xi : x) {
                result.push_back(lambda * ((xi > 0) ? xi : alpha * (exp(xi) - 1)));
            }
            break;
            
        default:
            throw invalid_argument("Unsupported activation type in applyActivation");
    }
    return result;
}

vector<double> activationDerivative(const vector<double>& x, ActivationType act_type,
                                    double alpha, double lambda) {
    if (x.empty()) return {};
    
    vector<double> deriv;
    deriv.reserve(x.size());
    
    switch (act_type) {
        case ActivationType::RELU:
            for (double xi : x) deriv.push_back((xi > 0) ? 1.0 : 0.0);
            break;
            
        case ActivationType::LEAKY_RELU:
            for (double xi : x) deriv.push_back((xi > 0) ? 1.0 : alpha);
            break;
            
        case ActivationType::SIGMOID: {
            for (double xi : x) {
                double sig = 1.0 / (1.0 + exp(-xi));
                deriv.push_back(sig * (1 - sig));
            }
            break;
        }
            
        case ActivationType::TANH:
            for (double xi : x) {
                double t = tanh(xi);
                deriv.push_back(1 - t * t);
            }
            break;
            
        case ActivationType::LINEAR:
            deriv = vector<double>(x.size(), 1.0);
            break;
            
        case ActivationType::SOFTMAX:
            throw logic_error("Softmax derivative should be handled with cross-entropy loss");
            
        case ActivationType::SELU:
            for (double xi : x) {
                deriv.push_back((xi > 0) ? lambda : lambda * alpha * exp(xi));
            }
            break;
            
        default:
            throw invalid_argument("Unsupported activation type in activationDerivative");
    }
    return deriv;
}

string activationTypeToString(ActivationType act_type) {
    switch (act_type) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::LINEAR: return "Linear";
        case ActivationType::SOFTMAX: return "Softmax";
        case ActivationType::SELU: return "SELU";
        default: return "Unknown";
    }
}
