#ifndef LOSSES_H
#define LOSSES_H

#include <vector>
#include <cmath>
#include <stdexcept>
// #include <algorithm>

template <typename T>
inline const T& clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

namespace Losses {

/**
 * @brief Computes Mean Squared Error (MSE) between true and predicted values. \n
 * 
 * @param y_true Ground truth values.
 * @param y_pred Predicted values.
 * @return MSE loss = (y_true - y_pred)^2
 */
inline double mse_loss(const std::vector<std::vector<double>>& y_true,
                       const std::vector<std::vector<double>>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("MSE: Batch sizes do not match.");
    }
    double sum = 0.0;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("MSE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            double diff = y_true[i][j] - y_pred[i][j];
            sum += diff * diff;
        }
    }
    return sum / (batch_size * num_classes);
}

/**
 * @brief Derivative of Mean Squared Error.
 * 
 * @param y_true Ground truth values.
 * @param y_pred Predicted values.
 * @return MSE_gradient = 2 * (y_pred - y_true)
 */
inline std::vector<std::vector<double>> mse_derivative(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("MSE Derivative: Batch sizes do not match.");
    }
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }
    std::vector<std::vector<double>> grad(batch_size, std::vector<double> (num_classes));

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("MSE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            grad[i][j] = 2.0 * (y_pred[i][j] - y_true[i][j]) / (batch_size * num_classes);
        }
    }
    return grad;
}

/**
 * @brief Computes Mean Absolute Error (MAE) between true and predicted values.
 * 
 * @param y_true Ground truth values.
 * @param y_pred Predicted values.
 * @return MAE = |y_pred - y_true|
 */
inline double mae_loss(const std::vector<std::vector<double>>& y_true,
                                  const std::vector<std::vector<double>>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("MAE: Batch sizes do not match.");
    }
    double sum = 0.0;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("MSE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            sum += std::abs(y_true[i][j] - y_pred[i][j]);
        }
    }
    return sum / (batch_size * num_classes);
}

/**
 * @brief Subgradient of Mean Absolute Error (not differentiable at 0).
 * 
 * @param y_true Ground truth values.
 * @param y_pred Predicted values.
 * @return MAE_subgradient = 1 if y_pred > y_true, -1 if y_pred < y_true, 0 otherwise.
 */
inline std::vector<std::vector<double>> mae_derivative(const std::vector<std::vector<double>>& y_true,
                                                       const std::vector<std::vector<double>>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("MAE Derivative: Batch sizes do not match.");
    }
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }
    std::vector<std::vector<double>> grad(batch_size, std::vector<double>(num_classes));

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("MAE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            if (y_pred[i][j] < y_true[i][j]) grad[i][j] = -1.0 / (batch_size * num_classes);
            else if (y_pred[i][j] > y_true[i][j]) grad[i][j] = 1.0 / (batch_size * num_classes);
            else grad[i][j] = 0.0;
        }
    }
    return grad;
}

/**
 * @brief Computes Binary Cross-Entropy (Log Loss) for binary classification.
 * 
 * @param y_true Ground truth labels (0 or 1).
 * @param y_pred Predicted probabilities (0.0 to 1.0).
 * @return BCE_loss = - (y_true) * log(y_pred) 
 */
inline double bce_loss(const std::vector<std::vector<double>>& y_true,
                                   const std::vector<std::vector<double>>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("BCE: Batch sizes do not match.");
    }
    const double epsilon = 1e-12;
    double loss = 0.0;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("BCE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            double pred = clamp(y_pred[i][j], epsilon, 1.0 - epsilon);
            loss += y_true[i][j] * std::log(pred) + (1.0 - y_true[i][j]) * std::log(1.0 - pred);
        }
    }
    return -loss / (batch_size * num_classes);
}

/**
 * @brief Derivative of Binary Cross-Entropy Loss.
 * 
 * @param y_true Ground truth labels (0 or 1).
 * @param y_pred Predicted probabilities (0.0 to 1.0).
 * @return BCE_derivative = - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)
 */
inline std::vector<std::vector<double>> bce_derivative(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("BCE Derivative: Batch sizes do not match.");
    }
    const double epsilon = 1e-12;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }
    std::vector<std::vector<double>> grad(batch_size, std::vector<double>(num_classes));

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("BCE: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            double pred = clamp(y_pred[i][j], epsilon, 1.0 - epsilon);
            grad[i][j] = (pred - y_true[i][j]) / (pred * (1.0 - pred)) / (batch_size * num_classes);
        }
    }
    return grad;
}

/**
 * @brief Computes the Cross-Entropy Loss for softmax outputs over a batch.
 * 
 * Loss = - (1/N) * sum_over_samples(sum_over_classes(y_true * log(pred)))
 * 
 * @param pred 2D vector of predicted softmax probabilities. Shape: [batch_size x num_classes].
 * @param y_true 2D vector of true labels in one-hot encoded form. Shape: [batch_size x num_classes].
 * @return double The computed average Cross-Entropy Loss for the batch.
 */
inline double cross_entropy_loss(const std::vector<std::vector<double>>& y_true,
                                 const std::vector<std::vector<double>>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Cross Entropy: Batch sizes do not match.");
    }
    const double epsilon = 1e-12;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }
    double loss = 0.0;

    for (size_t i = 0; i < batch_size; ++i) {
        if (y_true[i].size() != num_classes || y_pred[i].size() != num_classes) {
            throw std::invalid_argument("Cross Entropy: All rows must have same number of classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            if (y_true[i][j] == 1.0) {
                loss += std::log(std::max(y_pred[i][j], epsilon));
            }
        }
    }
    return -loss / batch_size;
}

/**
 * @brief Computes the derivative of the Cross-Entropy Loss for softmax outputs.
 * 
 * Derivative: grad[i][j] = pred[i][j] - y_true[i][j] (for each sample i, class j).
 * 
 * @param pred 2D vector of predicted softmax probabilities. Shape: [batch_size x num_classes].
 * @param y_true 2D vector of true labels in one-hot encoded form. Shape: [batch_size x num_classes].
 * @param grad Output 2D vector to store computed gradients. Resized to match input shape.
 */
inline std::vector<std::vector<double>> cross_entropy_derivative(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Cross Entropy Derivative: Batch sizes do not match.");
    }
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in MSE.");
    }
    std::vector<std::vector<double>> grad(batch_size, std::vector<double>(num_classes, 0.0));

    for (size_t i = 0; i < batch_size; ++i) {if (y_true[i].size() != num_classes || y_pred[i].size() !=  num_classes) {
            throw std::invalid_argument("Cross Entropy Derivative: All rows must have same number of  classes.");
        }
        for (size_t j = 0; j < num_classes; ++j) {
            grad[i][j] = (y_pred[i][j] - y_true[i][j]) / batch_size;
        }
    }
    return grad;
}

/**
 * @brief Computes Hinge Loss for SVM.
 * 
 * @param y_true True labels (-1 or +1).
 * @param y_pred Predicted raw scores (not probabilities).
 * @return double Hinge Loss.
 */
inline double hinge_loss(const std::vector<std::vector<double>>& y_true,
                         const std::vector<std::vector<double>>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Hinge Loss: Batch sizes do not match.");
    }

    double loss = 0.0;
    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in Hinge Loss.");
    }

    // Check if the input is binary classification (1D y_true and y_pred) or multi-class
    bool is_binary = y_true[0].size() == 1 && y_pred[0].size() == 1;

    // Binary classification case
    if (is_binary) {
        for (size_t i = 0; i < batch_size; ++i) {
            double correct_class_score = y_pred[i][0];
            double incorrect_class_score = 0.0; // only one class, so the incorrect score is always 0
            
            // Hinge loss for binary classification
            double margin = 1.0 - y_true[i][0] * correct_class_score; // y_true is either 0 or 1
            if (margin > 0) {
                loss += margin;
            }
        }
    }
    // Multi-class classification case
    else {
        for (size_t i = 0; i < batch_size; ++i) {
            double correct_class_score = 0.0;

            // Find the score of the correct class (the one-hot encoded class)
            for (size_t j = 0; j < num_classes; ++j) {
                if (y_true[i][j] == 1) {
                    correct_class_score = y_pred[i][j];
                    break;
                }
            }

            // Calculate hinge loss for each incorrect class
            for (size_t j = 0; j < num_classes; ++j) {
                if (y_true[i][j] == 0) {
                    // Calculate margin violation for incorrect classes
                    double margin = 1.0 - (correct_class_score - y_pred[i][j]);
                    if (margin > 0) {
                        loss += margin;
                    }
                }
            }
        }
    }

    // Return the average loss over the batch
    return loss / batch_size;
}

/**
 * @brief Computes derivative of hinge loss for SVM.
 * 
 * For each sample, gradient is -y / n if margin < 1, else 0.
 * 
 * @param pred Raw model outputs (scores).
 * @param y_true True labels, must be -1 or +1.
 * @param grad Output vector to store gradients; resized automatically.
 */
inline std::vector<std::vector<double>> hinge_loss_derivative(
    const std::vector<std::vector<double>>& y_true,
    const std::vector<std::vector<double>>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Hinge Loss Derivative: Batch sizes do not match.");
    }

    size_t batch_size = y_true.size();
    size_t num_classes = y_true[0].size();
    if (batch_size == 0 || num_classes == 0) {
        throw std::invalid_argument("Empty batch or classes in Hinge Loss Derivative.");
    }

    std::vector<std::vector<double>> grad(batch_size, std::vector<double>(num_classes, 0.0));

    // Check if binary classification (1D y_true and y_pred) or multi-class
    bool is_binary = y_true[0].size() == 1 && y_pred[0].size() == 1;

    if (is_binary) {
        // Binary classification case
        for (size_t i = 0; i < batch_size; ++i) {
            double margin = 1.0 - y_true[i][0] * y_pred[i][0]; // Only one score per sample

            if (margin > 0) {
                // Gradient for the correct class
                grad[i][0] = -y_true[i][0] / batch_size;

                // Gradient for the incorrect class (margin > 0)
                if (y_true[i][0] == 0) {
                    grad[i][0] += 1.0 / batch_size;
                }
            }
        }
    } else {
        // Multi-class classification case
        for (size_t i = 0; i < batch_size; ++i) {
            size_t correct_class = 0;

            // Find the correct class (where y_true[i][j] == 1)
            for (size_t j = 0; j < num_classes; ++j) {
                if (y_true[i][j] == 1) {
                    correct_class = j;
                    break;
                }
            }

            // Compute the gradient for each class
            for (size_t j = 0; j < num_classes; ++j) {
                double margin = 1.0 - (y_pred[i][correct_class] - y_pred[i][j]);

                if (j == correct_class) {
                    // Gradient for the correct class: the sum of the margins for incorrect classes
                    for (size_t k = 0; k < num_classes; ++k) {
                        if (y_true[i][k] == 0 && margin > 0) {
                            grad[i][correct_class] -= y_true[i][k] / batch_size;
                        }
                    }
                } else {
                    // Gradient for the incorrect classes
                    if (margin > 0) {
                        grad[i][j] = -y_true[i][j] / batch_size;
                    }
                }
            }
        }
    }

    return grad;
}


} // namespace Losses

#endif // LOSSES_H
