#include "../../include/Optimizers/SGD.h"

SGD::SGD(double lr) : Optimizer(lr) {}

void SGD::step(std::vector<DenseLayer*>& layers) {
    for (auto* layer : layers) {
        auto weights = layer->getWeights();
        auto biases = layer->getBiases();

        const auto& grad_weights = layer->getGradWeights();
        const auto& grad_biases = layer->getGradBiases();

        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= this->learning_rate * grad_weights[i][j];
            }
            biases[i] -= this->learning_rate * grad_biases[i];
        }
        layer->setWeights(weights);
        layer->setBiases(biases);
    }
    this->clear_gradients(layers);
}

void SGD::step_per_sample(std::vector<DenseLayer*>& layers) {
    this->step(layers);
}

void SGD::step_after_batch(std::vector<DenseLayer*>& layers) {}
