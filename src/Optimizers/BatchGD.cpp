#include "../../include/Optimizers/BatchGD.h"

BatchGD::BatchGD(double lr) : Optimizer(lr) {}

void BatchGD::step(std::vector<DenseLayer*>& layers) {
    for (auto* layer : layers) {
        auto& weights = const_cast<std::vector<std::vector<double>>&>(layer->getWeights());
        auto& biases = const_cast<std::vector<double>&>(layer->getBiases());

        const auto& grad_weights = layer->getGradWeights();
        const auto& grad_biases = layer->getGradBiases();

        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= this->learning_rate * grad_weights[i][j];
            }
            biases[i] -= this->learning_rate * grad_biases[i];
        }
    }
}

void BatchGD::step_per_sample(std::vector<DenseLayer*>& layers) {}

void BatchGD::step_after_batch(std::vector<DenseLayer*>& layers) {
    this->step(layers);
}
