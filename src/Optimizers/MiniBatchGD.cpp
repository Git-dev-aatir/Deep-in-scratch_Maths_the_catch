#include "../../include/Optimizers/MiniBatchGD.h"

MiniBatchGD::MiniBatchGD(double lr, size_t mb_size) 
    : Optimizer(lr), mini_batch_size(mb_size), sample_count(0) {}

void MiniBatchGD::step_per_sample(std::vector<DenseLayer*>& layers) {
    this->sample_count++;
    if (this->sample_count % this->mini_batch_size == 0) {
        this->step(layers);
        this->clear_gradients(layers);
        this->sample_count = 0;
    }
}

void MiniBatchGD::step(std::vector<DenseLayer*>& layers) {
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

void MiniBatchGD::step_after_batch(std::vector<DenseLayer*>& layers) {
    if (this->sample_count != 0) {
        this->step(layers);
        this->clear_gradients(layers);
        this->sample_count = 0;
    }
}
