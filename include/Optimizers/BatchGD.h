#ifndef BATCH_GD_H
#define BATCH_GD_H

#include "./Optim.h"

/**
 * @brief Batch Gradient Descent Optimizer.
 */
class BatchGradientDescent : public Optimizer {
public:
    /**
     * @brief Constructor for BatchGradientDescent.
     * @param lr Learning rate.
     */
    BatchGradientDescent(double lr = 0.01) : Optimizer(lr) {}

    /**
     * @brief Perform a batch gradient descent step for a Dense layer.
     * @param layer Vector of pointers to all Dense layers.
     */
    void step(std::vector<Dense*>& layers) override {
        for (Dense* layer : layers) {
            const auto& grad_weights = layer->getGradWeights();
            const auto& grad_biases = layer->getGradBiases();

            auto& weights = const_cast<std::vector<std::vector<double>>&>(layer->getWeights());
            auto& biases = const_cast<std::vector<double>&>(layer->getBiases());

            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    weights[i][j] -= learning_rate * grad_weights[i][j];
                }
                biases[i] -= learning_rate * grad_biases[i];
            }
        }
    }
};

#endif // BATCH_GD_H
