#ifndef SGD_H
#define SGD_H

#include "./Optim.h"

/**
 * @brief Stochastic Gradient Descent (SGD) Optimizer.
 */
class SGD : public Optimizer {
public:
    /**
     * @brief Constructor for SGD.
     * @param lr Learning rate.
     */
    SGD(double lr = 0.01) : Optimizer(lr) {}

    /**
     * @brief Perform an SGD step for a Dense layer.
     * @param layers Vector of pointers to all Dense layers in the model.
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

#endif // SGD_H
